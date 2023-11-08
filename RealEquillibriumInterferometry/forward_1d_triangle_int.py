import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from chord_geometry_int import ChordGeometryInterferometry
from magnetic_equilibrium import MagneticEquilibrium, MagneticEquilibriumSinglePoint
from interferometry import Interferometry, InterferometrySinglePoint
from density_profile import DensityProfile, DensityProfileSinglePoint
import logging

logger = logging.getLogger('forward_model')


def cross_product(v1, v2):
    """
    Get the cross product of two vectors
    """
    return v1[0] * v2[1] - v2[0] * v1[1]


def inside_triangle(xt, yt, x0, x1, x2, y0, y1, y2):
    """
    Whether a point (xt, yt) is inside the triangle defined by nodes (x0, y0), (x1, y1) and (x2, y2)
    """
    x01 = x1 - x0
    y01 = y1 - y0
    x0t = xt - x0
    y0t = yt - y0
    x12 = x2 - x1
    y12 = y2 - y1
    x1t = xt - x1
    y1t = yt - y1
    x20 = x0 - x2
    y20 = y0 - y2
    x2t = xt - x2
    y2t = yt - y2
    return cross_product((x01, y01), (x0t, y0t)) >= 0 and \
           cross_product((x12, y12), (x1t, y1t)) >= 0 and \
           cross_product((x20, y20), (x2t, y2t)) >= 0


class LOSChord(object):
    """
    The necessary information of a single line of sight (chord), which is required for constructing the forward model.
    The radial positions are in flux coordinates (rho_toroidal)
    """
    def __init__(self, start_point, end_point, dl, scipy_tri, rho_nodes, rho_profile):
        """
        To calculate the line integration numerically, radial positions of a series of discrete points on chord are determined through interpolation
        :param start_point: array_like. The start point (x1, y1)
        :param end_point: array_like. The start point (x2, y2)
        :param dl: float. The step size for numerical integration
        :param scipy_tri: A scipy.spatial.Delaunay object containing the triangular elements in the poloidal cross section
        :param rho_nodes: array_like. The radial positions of nodes of triangular elements
        :param rho_profile: array_like. The radial positions for the density profile
        """
        self.start_point = np.array(start_point)
        self.end_point = np.array(end_point)
        self.dl = dl
        self.num_points = 0  # Number of discrete points on the chord (inside the plasma)
        self.points = []  # Euclidean coordinates of the discrete points on LOS
        self.scipy_tri = scipy_tri
        self.triangles_of_points = []   # The indices of triangular elements containing the discrete points
        self.rho_discrete = None    # Radial positions of the discrete points on LOS
        self.rho_nodes = rho_nodes  # The radial positions of nodes of triangular elements
        self.rho_profile = rho_profile    # The radial positions of the density profile
        self.chord_length = np.zeros_like(rho_profile)     # The contribution from each radial position (of the density profile)

        logger.debug('generating_points...')
        self.generate_points()  # Get the discrete points on chord
        logger.debug('get contribution from each radial position...')
        self.get_contribution_from_radial_position()  # Get the contribution from each radial position

    def generate_points(self):
        """
        Get the discrete points on chord
        """
        vec = self.end_point - self.start_point
        vec_norm = np.linalg.norm(vec)
        unit = vec / vec_norm
        p = self.start_point.copy()
        triangle = self.scipy_tri.find_simplex(p).item()
        # Find the first point inside the plasma
        while triangle == -1 and np.linalg.norm(p-self.start_point) < vec_norm:
            p += self.dl * unit
            triangle = self.scipy_tri.find_simplex(p).item()
        # Find all points inside the plasma
        while triangle != -1:
            self.points.append(p.copy())
            self.triangles_of_points.append(triangle)
            p += self.dl * unit
            triangle = self.scipy_tri.find_simplex(p).item()
        self.num_points = len(self.points)
        logger.debug(f'Get {self.num_points} points on chord')

    def get_interpolated_rho(self, x, y, triangle):
        """
        Convert a Euclidean coordinate to flux coordinate using Barycentric interpolation
        """
        x0, x1, x2 = self.scipy_tri.points[self.scipy_tri.simplices[triangle, :], 0]
        y0, y1, y2 = self.scipy_tri.points[self.scipy_tri.simplices[triangle, :], 1]
        f0, f1, f2 = self.rho_nodes[self.scipy_tri.simplices[triangle, :]]
        assert inside_triangle(x, y, x0, x1, x2, y0, y1, y2), f'Point ({x}, {y}) is outside the triangle!'
        w0 = ((y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)) / \
             ((y0 - y1) * (x2 - x1) - (x0 - x1) * (y2 - y1))
        w1 = ((y - y2) * (x0 - x2) - (x - x2) * (y0 - y2)) / \
             ((y1 - y2) * (x0 - x2) - (x1 - x2) * (y0 - y2))
        w2 = 1 - w0 - w1
        return f0 * w0 + f1 * w1 + f2 * w2

    def get_contribution_from_radial_position(self):
        """
        Get the contribution of each radial position (of the density profile) to each discrete point on chord
        """
        if self.num_points > 0:
            self.rho_discrete = np.zeros(self.num_points)
            for i in range(self.num_points):
                # Radial position of each point on LOS
                self.rho_discrete[i] = self.get_interpolated_rho(self.points[i][0], self.points[i][1], self.triangles_of_points[i])
            self.get_linear_interpolated_weights()

    def get_linear_interpolated_weights(self):
        """
        Assign weights for each radial position of density profile, according to the distances
        between self.rho_discrete and two nearest points of self.rho_profile
        """
        rho_dif = self.rho_discrete[:, np.newaxis] - self.rho_profile[np.newaxis, :]
        rho_dif[rho_dif < 0] = 1e5
        idx_left = np.argmin(rho_dif, axis=1)
        idx_right = idx_left + 1
        rho_left = self.rho_profile[idx_left]
        rho_right = self.rho_profile[idx_right]
        interval = rho_right - rho_left
        dist_left = self.rho_discrete - rho_left
        dist_right = rho_right - self.rho_discrete

        # first point
        self.chord_length[idx_left[0]] += dist_right[0] / interval[0] * self.dl / 2
        self.chord_length[idx_right[0]] += dist_left[0] / interval[0] * self.dl / 2

        # middle points
        for i in range(1, self.rho_discrete.shape[0]):
            self.chord_length[idx_left[i]] += dist_right[i] / interval[i] * self.dl
            self.chord_length[idx_right[i]] += dist_left[i] / interval[i] * self.dl

        # last point
        self.chord_length[idx_left[-1]] += dist_right[-1] / interval[-1] * self.dl / 2
        self.chord_length[idx_right[-1]] += dist_left[-1] / interval[-1] * self.dl / 2


def _compute_response_matrix(los_r_start, los_r_end, los_z_start, los_z_end, dl, r_all, z_all, rho_nodes, rho_profile):
    """
    In order to construct the (linear) forward model, we calculate the response matrix R (or sometimes called the transfer matrix)
    The relation between density profile ne and line-integrated measurements d: d = R * ne
    :param los_r_start: The r coordinates of the start points of the lines of sight
    :param los_r_end: The r coordinates of the end points the lines of sight
    :param los_z_start: The z coordinates of the start points of the lines of sight
    :param los_z_end: The z coordinates of the end points of the lines of sight
    :param dl: The step size for numerical integration
    :param r_all: The r coordinates of all nodes
    :param z_all: The z coordinates of all nodes
    :param rho_nodes: The radial positions of nodes of triangular elements
    :param rho_profile: The radial positions for the density profile
    :return: The response matrix R
    """
    chords = []
    num_los = los_r_start.shape[0]
    len_rho_profile = rho_profile.shape[0]
    scipy_tri = Delaunay(np.vstack((r_all, z_all)).T)
    for i in range(num_los):
        logger.debug(f'Computing chord {i+1}...')
        chords.append(LOSChord(start_point=(los_r_start[i], los_z_start[i]),
                               end_point=(los_r_end[i], los_z_end[i]),
                               dl=dl, scipy_tri=scipy_tri, rho_nodes=rho_nodes, rho_profile=rho_profile))
    _response = np.zeros([num_los, len_rho_profile])
    for i in range(num_los):
        _response[i, :] = chords[i].chord_length
    return _response


def compute_response_matrix(geo: ChordGeometryInterferometry, equi: MagneticEquilibriumSinglePoint, rho_profile, dl=0.01):
    """
    Calculate response matrix for a given magnetic equilibrium
    :param geo: Chord geometry
    :param equi: Magnetic equilibrium
    :param rho_tor_norm_1d: Normalized radii (toroidal flux coordinate) of the density profile
    :param dl: The step size for numerical integration
    :return: The response matrix R
    """
    nan_mask = np.isnan(equi.psi)
    valid_mask = np.logical_and(np.logical_not(nan_mask), equi.psi_norm <= 1.0)  # Mask the area inside the plasma
    r_valid = equi.r_equi_2d[valid_mask]
    z_valid = equi.z_equi_2d[valid_mask]
    rho_valid = equi.rho_tor_norm_2d[np.logical_not(nan_mask)]
    response = _compute_response_matrix(geo.los_r_start, geo.los_r_end, geo.los_z_start, geo.los_z_end, dl, r_valid,
                                        z_valid, rho_valid, rho_profile)
    return response


@mpl.rc_context({'font.family': 'Times New Roman', 'xtick.direction': 'in', 'ytick.direction': 'in',
                 'figure.dpi': 100})
def plot_geometry_with_mag_equi(geo: ChordGeometryInterferometry, equi: MagneticEquilibriumSinglePoint, file_name=None):
    """
    Plot the geometry of the chords along with magnetic surfaces
    """
    plt.figure()
    ax = plt.subplot(111)
    r = equi.r_equi_2d[0, :]
    z = equi.z_equi_2d[:, 0]
    num_r_points = r.shape[0]
    num_z_points = z.shape[0]
    # Plot grid lines
    for i in range(num_r_points):
        plt.plot(r[i] * np.ones_like(z), z, linestyle='-', linewidth=0.5, color='grey', alpha=0.3)
    for j in range(num_z_points):
        plt.plot(r, z[j] * np.ones_like(r), linestyle='-', linewidth=0.5, color='grey', alpha=0.3)
    plt.contour(equi.r_equi_2d, equi.z_equi_2d, equi.psi_norm, linestyles='dashed', linewidths=1, colors='black')
    plt.plot(equi.plasma_boundary[0, :], equi.plasma_boundary[1, :], linestyle='--', linewidth=2, color='red')
    for i in range(geo.num_los):
        plt.plot([geo.los_r_start[i], geo.los_r_end[i]], [geo.los_z_start[i], geo.los_z_end[i]], label=f'ch{i + 1}')
    plt.xlabel('R (m)', fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Geometry of the chords from interferometry', fontsize=16)
    if file_name is not None:
        plt.savefig(file_name)
    plt.show()


def forward_1d_triangle_main():
    t = 5.2113
    shot = 53259

    geo = ChordGeometryInterferometry(f'data/WEST/{shot}/los_interferometer_{shot}.mat')

    equi_all = MagneticEquilibrium()
    equi_all.load_from_imas(f'data/WEST/{shot}/imas_equilibrium_{shot}.mat')
    equi = equi_all.get_single_point(t)

    interf_all = Interferometry()
    interf_all.load_from_imas(f'data/WEST/{shot}/imas_interferometer_{shot}.mat')
    interf = interf_all.get_single_point(t)

    dens_prof_all = DensityProfile()
    dens_prof_all.load_from_imas(f'data/WEST/{shot}/imas_core_profiles_{shot}_occ1.mat', shot=shot)
    dens_prof = dens_prof_all.get_single_point(t)
    dens = dens_prof.dens_1d

    plot_geometry_with_mag_equi(geo, equi)

    response = compute_response_matrix(geo, equi, dens_prof.rho_tor_norm_1d)

    # Calculate the line-integrated density using the forward model
    lid_recon = np.matmul(response, dens)

    channels = interf.channels.data
    plt.figure(dpi=100)
    plt.plot(channels, lid_recon, 'b', label='calculated')
    plt.plot(channels, equi.lid_nice, 'g', label='NICE')
    plt.plot(interf.channels.compressed(), interf.lid.compressed(), 'r', label='measured')
    plt.xlabel('channel')
    plt.ylabel('line-integrated density ($10^{19} \mathrm{m}^{-2}$)')
    plt.legend()
    plt.xticks(channels, channels)
    plt.show()


if __name__ == '__main__':
    forward_1d_triangle_main()
