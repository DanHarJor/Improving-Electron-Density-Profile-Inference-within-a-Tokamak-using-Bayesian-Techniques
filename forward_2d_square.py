import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from scipy.spatial import distance


mpl.rcParams["image.cmap"] = 'jet'
mpl.rcParams["lines.linewidth"] = 0.5
mpl.rcParams["lines.markersize"] = 4
mpl.rcParams["figure.dpi"] = 200


def line_function_x_2d(y, x1, x2, y1, y2):
    """
    The line function is determined by 2 points (x1, y1) and (x2, y2).
    Given a point on the line with an ordinate of y, return the corresponding abscissa x.

    :return: x or None (if the line is parallel to the x axis)
    """
    return None if y1 == y2 else ((x2 - x1) / (y2 - y1)) * (y - y1) + x1


def line_function_y_2d(x, x1, x2, y1, y2):
    """
    The line function is determined by 2 points (x1, y1) and (x2, y2).
    Given a point on the line with a abscissa of x, return the corresponding ordinate y.

    :return: y or None (if the line is parallel to the y axis)
    """
    return None if x1 == x2 else ((y2 - y1) / (x2 - x1)) * (x - x1) + y1


def get_nearest_normalized_radii(rho_pixel, rho_discrete):
    '''
    Get the nearest discrete normalized radii

    :param rho_pixel: Normalized radii of all pixels
    :param rho_discrete: Discrete normalized radii
    :return: The indices of the discrete values to which the normalized radius of each point belongs.
    '''
    rho_discrete = np.expand_dims(rho_discrete, -1)
    rho_pixel = np.expand_dims(rho_pixel, -1)
    distances = distance.cdist(rho_discrete, rho_pixel)  # Distance between each pair of the two collections of inputs
    digitized = np.argmin(distances, axis=0)
    return digitized


def compute_cross_points(r_pixel, z_pixel, start_angles, end_angles):
    """
    Get the coordinates of points where each LOS intersects with grid lines

    :param r_pixel: r coordinates of (the centers) of pixels
    :param z_pixel: z coordinates of (the centers) of pixels
    :param start_angles: angles that parameterize the start points of lines of sight
    :param end_angles: angles that parameterize the end points of lines of sight
    :return: The coordinates of points where each LOS intersects with grid lines
    """
    dr = r_pixel[1] - r_pixel[0]
    dz = z_pixel[1] - z_pixel[0]
    grid_r = np.hstack([r_pixel[0] - dr / 2, r_pixel + dr / 2])
    grid_z = np.hstack([z_pixel[0] - dz / 2, z_pixel + dz / 2])
    los_r_start = 1.5 + 0.5 * np.cos(start_angles)
    los_r_end = 1.5 + 0.5 * np.cos(end_angles)
    los_z_start = 0.5 * np.sin(start_angles)
    los_z_end = 0.5 * np.sin(end_angles)
    num_los = los_r_start.shape[0]
    cross_points = np.empty(num_los, dtype=object)
    for i in range(num_los):
        get_cross_x = lambda y: line_function_x_2d(y, los_r_start[i], los_r_end[i],
                                                   los_z_start[i], los_z_end[i])
        get_cross_y = lambda x: line_function_y_2d(x, los_r_start[i], los_r_end[i],
                                                   los_z_start[i], los_z_end[i])
        cross_x = list(map(get_cross_x, grid_z))
        cross_y = list(map(get_cross_y, grid_r))
        points = np.vstack((np.hstack((cross_x, grid_r)),
                            np.hstack((grid_z, cross_y))))
        # Remove invalid points
        indices_invalid = []
        for n in range(points.shape[1]):
            if points[0, n] is None or points[1, n] is None \
                    or points[0, n] < grid_r[0] or points[0, n] > grid_r[-1] \
                    or points[1, n] < grid_z[0] or points[1, n] > grid_z[-1]:
                indices_invalid.append(n)
        points = np.delete(points, indices_invalid, axis=1)
        # Sort by x (if x is the same then sort by y)
        sorted_idx = np.lexsort((points[1, :], points[0, :]))
        cross_points[i] = points[:, sorted_idx]
    return cross_points


def compute_all_chord_lengths(r_pixel, z_pixel, cross_points):
    """
    Compute the length of each chord inside each pixel

    :param r_pixel: r coordinates of (the centers) of pixels
    :param z_pixel: z coordinates of (the centers) of pixels
    :param cross_points: The coordinates of points where each LOS intersects with grid lines
    :return: None
    """
    num_r_pixel = r_pixel.shape[0]
    num_pixels = r_pixel.shape[0] * z_pixel.shape[0]
    num_los = cross_points.shape[0]
    chord_length = np.zeros([num_los, num_pixels])
    for i in range(num_los):
        num_cross = cross_points[i].shape[1]
        # Find which cell that each point of intersection belongs to and compute the chord length
        for j in range(num_cross - 1):
            chord_start = cross_points[i][:, j]
            chord_end = cross_points[i][:, j + 1]
            chord_center = (chord_start + chord_end) / 2
            idx_r = np.argmin(np.abs(r_pixel - chord_center[0]))
            idx_z = np.argmin(np.abs(z_pixel - chord_center[1]))
            dist = scipy.linalg.norm(chord_end - chord_start)
            chord_length[i, idx_z * num_r_pixel + idx_r] = dist
    return chord_length


def compute_response_matrix(chord_lengths, rho_pixel, rho_discrete):
    '''
    Sum the contribution of pixels with the same normalized radius

    :param chord_lengths: The length of each chord inside each pixel
    :param rho_pixel: Normalized radii of all pixels
    :param rho_discrete: Discrete normalized radii
    :return: The length of each chord inside pixel groups with the same normalized radii (num_los, len_rho_discrete)
    '''
    digitized = get_nearest_normalized_radii(rho_pixel, rho_discrete)
    num_los = chord_lengths.shape[0]
    len_rho_discrete = rho_discrete.shape[0]
    response = np.zeros([num_los, len_rho_discrete])
    for i in range(num_los):
        for j in range(len_rho_discrete):
            response[i, j] = chord_lengths[i, digitized == j].sum()
    return response


def map_profile_to_2d(rho_1d, dens_1d, rho_2d):
    '''
    Map 1-D electron density profile to 2-D
    '''
    rho_2d_flat = rho_2d.flatten()
    dens_2d_flat = np.zeros_like(rho_2d_flat)
    indices_inside = np.argwhere(rho_2d_flat <= 1.0).flatten()
    dens_2d_flat[indices_inside] = interp1d(rho_1d, dens_1d, kind='cubic')(rho_2d_flat[indices_inside])
    dens_2d_flat[rho_2d_flat > 1.0] = np.nan
    return dens_2d_flat.reshape(rho_2d.shape)


def plot_example_configuration(start_angles, end_angles, rho_1d, dens_1d):
    r = np.linspace(1.0, 2.0, 201)
    z = np.linspace(-0.5, 0.5, 201)
    R, Z = np.meshgrid(r, z)
    rho_all = ((R - 1.5) ** 2 / 0.16 + Z ** 2 / 0.25)
    dens_2d = map_profile_to_2d(rho_1d, dens_1d, rho_all)
    num_los = len(start_angles)

    cross_points = compute_cross_points(r, z, start_angles, end_angles)
    chord_lengths = compute_all_chord_lengths(r, z, cross_points)
    exclude_indices = np.where(rho_all.flatten() > 1)
    chord_lengths[:, exclude_indices] = 0  # Remove the points outside the plasma boundary
    response = compute_response_matrix(chord_lengths, rho_all.flatten(), rho_1d)
    lid = np.matmul(response, dens_1d)
    print(lid)
    channel = np.arange(1, num_los + 1)

    fig = plt.figure(figsize=(10, 4))
    grid = plt.GridSpec(4, 10, wspace=0.5, hspace=0.5)
    ax1 = plt.subplot(grid[0:3, 0:4])
    ax2 = plt.subplot(grid[0:3, 6:10])

    # Plot 2-D electron density profile and the lines of sight
    cs = ax1.pcolormesh(R, Z, dens_2d, shading='auto')
    theta = np.linspace(0, np.pi*2, 100)
    x_circle = 1.5 + 0.5 * np.cos(theta)
    y_circle = 0.5 * np.sin(theta)
    ax1.plot(x_circle, y_circle, 'r')
    for i in range(num_los):
        x_start = 1.5 + 0.5 * np.cos(start_angles[i])
        x_end = 1.5 + 0.5 * np.cos(end_angles[i])
        y_start = 0.5 * np.sin(start_angles[i])
        y_end = 0.5 * np.sin(end_angles[i])
        ax1.plot([x_start, x_end], [y_start, y_end], 'k')
    ax1.set_aspect('equal')
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('2-D distribution of electron density')
    # cb = fig.colorbar(cs, ax=ax1)

    # Plot the line-integrated electron density for each LOS
    ax2.scatter(channel, lid)
    ax2.plot(channel, lid)
    ax2.set_xlabel('Channel number')
    ax2.set_xticks(channel)
    ax2.set_ylabel('Line integrated density')
    ax2.set_title('Calculated signals')
    plt.show()


if __name__ == '__main__':
    rho_1d = np.linspace(0.0, 1.0, 101)
    dens_1d = (1 - rho_1d**2) * 4.0
    plot_example_configuration([0, 0, 0.5 * np.pi, np.pi], [np.pi, 1.25 * np.pi, 1.5*np.pi, 1.5 * np.pi], rho_1d, dens_1d)