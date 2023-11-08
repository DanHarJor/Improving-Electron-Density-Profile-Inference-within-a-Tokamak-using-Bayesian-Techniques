import scipy.io as scio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from scipy.spatial import distance
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MagneticEquilibriumSinglePoint(object):
    def __init__(self, r_equi_2d=None, z_equi_2d=None, psi_norm=None, 
                 rho_tor_norm_1d=None,
                 rho_tor_norm_2d=None, r_ax=0.0, z_ax=0.0, psi=None, psi_prof=None, psi_boundary=None,
                 plasma_boundary=None, lid_nice=None, t=0.0, shot=0):
        """
        The magnetic equilibrium information of a single time point
        :param r_equi_2d:   r of the meshgrid points of reconstructed magnetic equilibrium
        :param z_equi_2d:   z of the meshgrid points of reconstructed magnetic equilibrium
        :param psi_norm: The normalized magnetic flux (0-1)
        :param rho_tor_norm_1d: Normalized radii (toroidal flux coordinate) of kinetic profiles
        :param rho_tor_norm_2d: Normalized radii of the meshgrid points of reconstructed magnetic equilibrium
        :param r_ax: r of the magnetic axis
        :param z_ax: z of the magnetic axis
        :param psi: The magnetic flux
        :param psi_prof: Magnetic flux profile (1d)
        :param psi_boundary: The value of magnetic flux on plasma boundary
        :param plasma_boundary: The points of the plasma boundary
        :param lid_nice: The line integrated density calculated by NICE (1e19 m-2)
        :param t: Time point
        :param shot: Shot number
        """
        self.r_equi_2d = r_equi_2d
        self.z_equi_2d = z_equi_2d
        self.psi_norm = psi_norm
        self.rho_tor_norm_1d = rho_tor_norm_1d
        self.rho_tor_norm_2d = rho_tor_norm_2d
        self.r_ax = r_ax
        self.z_ax = z_ax
        self.psi = psi
        self.psi_prof = psi_prof
        self.psi_boundary = psi_boundary
        self.plasma_boundary = plasma_boundary
        self.lid_nice = lid_nice
        self.t = t
        self.shot = shot
        # self.r_emp_2d = self.get_empirical_radius()

    def get_empirical_radius(self):
        """
        Get empirical radial coordinates of the meshgrid points of reconstructed magnetic equilibrium
        :return:
        """
        idx_r_emp_start_r = np.argmin(np.abs(self.r_equi_2d[0, :] - self.r_ax))  # Magnetic axis
        idx_r_emp_start_z = np.argmin(np.abs(self.z_equi_2d[:, 0] - self.z_ax))  # Magnetic axis
        plasma_boundary_right = self.plasma_boundary[:, self.plasma_boundary[0, :] > self.r_ax]  # The right direction
        idx_boundary = np.argmin(np.abs(plasma_boundary_right[1, :] - self.z_ax))  # The point where the boundary cross the mid-plane
        r_emp_end_r = plasma_boundary_right[0, idx_boundary]
        idx_r_emp_end_r = np.argmin(np.abs(self.r_equi_2d[0, :] - r_emp_end_r))
        psi_norm_1d = self.psi_norm[idx_r_emp_start_z, idx_r_emp_start_r:idx_r_emp_end_r]
        r_emp_1d = self.r_equi_2d[0, idx_r_emp_start_r:idx_r_emp_end_r] - self.r_ax
        f = interp1d(psi_norm_1d, r_emp_1d, kind='cubic', fill_value='extrapolate')
        r_emp_2d = f(self.psi_norm.flatten()).reshape(self.psi_norm.shape)
        return r_emp_2d

    @mpl.rc_context({'image.cmap': 'jet', 'xtick.direction': 'in', 'ytick.direction': 'in', 'figure.dpi': 100})
    def plot_mag_equi(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        psi_colormesh = ax.pcolormesh(self.r_equi_2d, self.z_equi_2d, self.psi_norm, shading='gouraud')
        fig.colorbar(psi_colormesh, cax=cax)
        ax.contour(self.r_equi_2d, self.z_equi_2d, self.psi_norm, colors='black', linewidths=0.5)
        ax.plot(self.plasma_boundary[0, :], self.plasma_boundary[1, :], linestyle='--', linewidth=2,
                color='red')
        ax.set_xlabel('R (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(f'Normalized poloidal flux of #{self.shot} (t = {self.t:.2f}s)')
        plt.show()

    def to_str_dict(self):
        str_dict = self.__dict__.copy()
        for k, v in str_dict.items():
            if v is None:
                str_dict[k] = 'None'
        return str_dict


class MagneticEquilibrium(object):
    def __init__(self):
        self.t0 = 0.0   # The beginning of the plasma
        self.time_equi = None  # Time points of magnetic equilibrium (relative to t0)
        self.ip = None  # Plasma current
        self.psi = None     # Magnetic flux
        self.psi_prof = None    # Magnetic flux profile (1d)
        self.psi_norm = None    # Normalized magnetic flux
        self.rho_tor_norm_1d = None     # Normalized radii (toroidal flux coordinate) of kinetic profiles
        self.rho_tor_norm_2d = None     # Normalized radii of the meshgrid points of reconstructed magnetic equilibrium
        self.plasma_boundary = None     # Points of the plasma boundary
        self.psi_boundary = None        # The value of magnetic flux on plasma boundary
        self.r_equi_2d = None   # r of the meshgrid points of reconstructed magnetic equilibrium
        self.z_equi_2d = None   # z of the meshgrid points of reconstructed magnetic equilibrium
        self.psi_ax = None  # Magnetic flux at the axis
        self.r_ax = None    # r of the magnetic axis
        self.z_ax = None    # z of the magnetic axis
        self.lid_nice = None    # The line integrated density calculated by NICE (1e19 m-2)
        self.shot = 0

    def load_from_imas(self, file_name, remove_x_point=True, remove_boundary=True, shot=0, Daniel=False):
        """
        Load magnetic equilibrium from IMAS file.
        :param file_name:   Name of the IMAS file (in .mat format)
        :param remove_x_point:  Whether to remove the region outside the X point
        :param remove_boundary: Whether to remove the region outside the boundary
        :return:
        """
        file_equilibrium = scio.loadmat(file_name)
        self.t0 = file_equilibrium['t0'][0, 0]
        self.shot = shot
        if Daniel: equi = file_equilibrium['equi_magonly_ids']
        else: equi = file_equilibrium['ids'] 
        # print(equi)
        # Filter no converged equilibrium computations
        output_flag = equi['code'][0, 0]['output_flag'][0, 0].flatten()
        mask_eq = (output_flag >= 0)
        self.time_equi = equi['time'][0, 0].flatten()[mask_eq] - self.t0
        self.ip = equi['ip'][0, 0].flatten()[mask_eq]
        self.r_equi_2d = equi['interp2D'][0, 0]['r'][0, 0].T
        self.z_equi_2d = equi['interp2D'][0, 0]['z'][0, 0].T
        psi = np.transpose(equi['interp2D'][0, 0]['psi'][0, 0][mask_eq, :, :], (0, 2, 1))
        self.psi_prof = equi['psi_prof'][0, 0][mask_eq, :]
        self.rho_tor_norm_1d = equi['profiles_1d'][0, 0]['rho_tor_norm'][0, 0][mask_eq, :]
        self.rho_tor_norm_2d = np.zeros_like(psi)
        self.plasma_boundary = equi['boundPlasma'][0, 0][mask_eq, :, :]
        self.psi_boundary = equi['boundary'][0, 0]['psi'][0, 0].flatten()[mask_eq]
        otherv = 5
        testv = equi['constraints'][0, 0]['n_e_line'][0, 0]['reconstructed'][0, 0]
        self.lid_nice = equi['constraints'][0, 0]['n_e_line'][0, 0]['reconstructed'][0, 0][mask_eq, :] / 1e19
        self.r_ax = equi['mag_ax_R'][0, 0].flatten()[mask_eq]
        self.z_ax = equi['mag_ax_Z'][0, 0].flatten()[mask_eq]
        for i in range(mask_eq.sum()):
            if remove_x_point:
                # Only keep the region inside the LCFS
                dist_axis_boundary = (self.plasma_boundary[i, 0, :] - self.r_ax[i]) ** 2 + \
                                     (self.plasma_boundary[i, 1, :] - self.z_ax[i]) ** 2
                max_dist_2 = dist_axis_boundary.max()
                psi[i, (self.r_equi_2d - self.r_ax[i]) ** 2 + (self.z_equi_2d - self.z_ax[i]) ** 2 > max_dist_2] = np.nan
            if remove_boundary:
                psi[i, psi[i, :, :] > self.psi_boundary[i]] = np.nan
            # Interpolation for rho_tor_norm
            psi_2d = psi[i, :, :].squeeze()
            psi_flatten = psi_2d.flatten()
            f = interp1d(self.psi_prof[i, :], self.rho_tor_norm_1d[i, :], kind='linear')
            rho_tor_norm_flatten = f(psi_flatten)
            self.rho_tor_norm_2d[i, :, :] = rho_tor_norm_flatten.reshape(self.r_equi_2d.shape)
        self.psi = psi
        self.psi_ax = np.zeros(mask_eq.sum())
        for i in range(mask_eq.sum()):
            psi_2d_min = self.psi[i][np.logical_not(np.isnan(psi[i]))].min()
            psi_1d_min = self.psi_prof[i, :].min()
            self.psi_ax[i] = np.minimum(psi_2d_min, psi_1d_min)
        self.psi_norm = (self.psi - self.psi_ax[:, np.newaxis, np.newaxis]) / \
                        (self.psi_boundary[:, np.newaxis, np.newaxis] - self.psi_ax[:, np.newaxis, np.newaxis])

    @mpl.rc_context({'image.cmap': 'jet', 'xtick.direction': 'in', 'ytick.direction': 'in', 'figure.dpi': 100})
    def plot_mag_contours(self, start, end, file_name=None):
        """
        Plot the (a superposition of) contours for a specified time range
        """
        fig = plt.figure()
        idx_start = np.argmin(np.abs(start - self.time_equi))
        idx_end = np.argmin(np.abs(end - self.time_equi))
        for i in range(idx_start, idx_end+1):
            plt.contour(self.r_equi_2d, self.z_equi_2d, self.psi_norm[i, :, :],
                        levels=np.linspace(0.0, 1.0, 6),
                        linestyles='solid', linewidths=0.2, colors='black')
            plt.plot(self.plasma_boundary[i, 0, :], self.plasma_boundary[i, 1, :], linestyle='--', linewidth=0.5,
                     color='red')
        plt.xlabel('R (m)')
        plt.ylabel('z (m)')
        plt.title(f'Normalized poloidal flux of shot {self.shot} '
                  f'({self.time_equi[idx_start]:.2f} - {self.time_equi[idx_end]:.2f}s)')
        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_name)
        plt.show()

    def get_psi_norm(self, t):
        """
        Get the normalized magnetic flux of a time point
        """
        idx_t = np.argmin(np.abs(t - self.time_equi))
        return self.time_equi[idx_t], self.psi_norm[idx_t, :, :]

    def get_single_point(self, t):
        """
        Get the normalized magnetic flux of a time point (returned as an object)
        """
        idx_t = np.argmin(np.abs(t - self.time_equi))
        r_equi_2d = self.r_equi_2d
        z_equi_2d = self.z_equi_2d
        psi = self.psi[idx_t, :, :]
        psi_prof = self.psi_prof[idx_t, :]
        psi_norm = self.psi_norm[idx_t, :, :]
        plasma_boundary = self.plasma_boundary[idx_t, :, :]
        psi_boundary = self.psi_boundary[idx_t]
        rho_tor_norm_1d = self.rho_tor_norm_1d[idx_t, :]
        rho_tor_norm_2d = self.rho_tor_norm_2d[idx_t, :, :]
        r_ax = self.r_ax[idx_t]
        z_ax = self.z_ax[idx_t]
        lid_nice = self.lid_nice[idx_t, :]
        equi_single = MagneticEquilibriumSinglePoint(r_equi_2d, z_equi_2d, psi_norm, rho_tor_norm_1d, rho_tor_norm_2d,
                                                     r_ax, z_ax, psi, psi_prof, psi_boundary, plasma_boundary, lid_nice,
                                                     self.time_equi[idx_t], self.shot)
        return equi_single

    def get_single_point_interp(self, t, r_res=25, z_res=25):
        """
        Get the normalized magnetic flux of a time point (returned as an object) with custom resolution
        :param t: Time
        :param r_res: R resolution
        :param z_res: Z resolution
        :return: The magnetic equilibrium information of a single time point
        """
        idx_t = np.argmin(np.abs(t - self.time_equi))
        r = np.linspace(self.r_equi_2d[0, 0], self.r_equi_2d[0, -1], r_res)
        z = np.linspace(self.z_equi_2d[0, 0], self.z_equi_2d[-1, 0], z_res)
        r_equi_2d, z_equi_2d = np.meshgrid(r, z)
        psi = self.psi[idx_t, :, :]
        rho_2d = self.rho_tor_norm_2d[idx_t, :, :]

        nan_mask = np.isnan(psi)
        valid_mask = np.logical_not(nan_mask)  # Mask the area inside the plasma
        r_valid = self.r_equi_2d[valid_mask]
        z_valid = self.z_equi_2d[valid_mask]
        points = np.hstack([r_valid.reshape(-1, 1), z_valid.reshape(-1, 1)])
        psi_valid = psi[valid_mask]
        rho_valid = rho_2d[valid_mask]

        # Interpolation
        psi_interp = griddata(points, psi_valid, (r_equi_2d, z_equi_2d), method='cubic')
        rho_2d_interp = griddata(points, rho_valid, (r_equi_2d, z_equi_2d), method='cubic')
        psi_norm = (psi_interp - self.psi_ax[idx_t]) / (self.psi_boundary[idx_t] - self.psi_ax[idx_t])
        psi_prof = self.psi_prof[idx_t, :]
        plasma_boundary = self.plasma_boundary[idx_t, :, :]
        psi_boundary = self.psi_boundary[idx_t]
        rho_tor_norm_1d = self.rho_tor_norm_1d[idx_t, :]
        rho_tor_norm_2d = rho_2d_interp
        r_ax = self.r_ax[idx_t]
        z_ax = self.z_ax[idx_t]
        lid_nice = self.lid_nice[idx_t, :]
        equi_single = MagneticEquilibriumSinglePoint(r_equi_2d, z_equi_2d, psi_norm, rho_tor_norm_1d, rho_tor_norm_2d,
                                                     r_ax, z_ax, psi_interp, psi_prof, psi_boundary, plasma_boundary, lid_nice,
                                                     self.time_equi[idx_t], self.shot)
        return equi_single


def magnetic_equilibrium_main():
    t = 5.2113
    print(t)
    shot = 53259
    print(shot)
    equi_all = MagneticEquilibrium()
    print('eq',type(equi_all))
    # equi_all.load_from_imas(f'data/WEST/{shot}/imas_equilibrium_{shot}.mat', shot=shot)
    # equi = equi_all.get_single_point(t)
    # equi_interp = equi_all.get_single_point_interp(t)
    # equi.plot_mag_equi()
    # equi_interp.plot_mag_equi()


if __name__ == '__main__':
    print('is it okay?')
    magnetic_equilibrium_main()



