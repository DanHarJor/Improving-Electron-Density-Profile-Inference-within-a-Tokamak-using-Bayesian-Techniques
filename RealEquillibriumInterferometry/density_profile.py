import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as scio
import re


class DensityProfileSinglePoint(object):
    def __init__(self, rho_tor_norm_1d, dens_1d, dens_1d_lower, dens_1d_upper, t=0.0, shot=0):
        """
        The density profile of a single time point
        :param rho_tor_norm_1d: Normalized radii (toroidal flux coordinate) of the density profiles
        :param dens_1d: Electron density profile (1e19 m-3)
        :param dens_1d_lower: The lower bound of electron density profile (1e19 m-3)
        :param dens_1d_upper: The upper bound of electron density profile (1e19 m-3)
        :param t: Time
        :param shot: Shot number
        """
        self.rho_tor_norm_1d = rho_tor_norm_1d
        self.dens_1d = dens_1d
        self.dens_1d_lower = dens_1d_lower
        self.dens_1d_upper = dens_1d_upper
        self.t = t
        self.shot = shot

    @mpl.rc_context({'image.cmap': 'jet', 'xtick.direction': 'in', 'ytick.direction': 'in', 'figure.dpi': 100})
    def plot_density_profile(self):
        plt.figure()
        plt.plot(self.rho_tor_norm_1d, self.dens_1d, color='red', linewidth=2)
        plt.plot(self.rho_tor_norm_1d, self.dens_1d_lower, 'r--')
        plt.plot(self.rho_tor_norm_1d, self.dens_1d_upper, 'r--')
        plt.title(f'# {self.shot} electron density profile (t = {self.t:.2f} s)')
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$\mathrm{n_e}(10^{19} \mathrm{m}^{-3})$')
        plt.show()


class DensityProfile(object):
    def __init__(self):
        self.t0 = 0.0             # The beginning of the plasma
        self.time_profiles = None   # Time points of electron density profiles (relative to t0)
        self.rho_tor_norm_1d = None       # Normalized radii (toroidal flux coordinate) of the density profiles
        self.dens_1d = None           # Electron density profile (1e19 m-3)
        self.dens_1d_lower = None     # The lower bound of electron density profile (1e19 m-3)
        self.dens_1d_upper = None     # The lower bound of electron density profile (1e19 m-3)
        self.shot = 0

    def load_from_imas(self, file_name, shot=0):
        """
        Load density profiles from IMAS file
        :param file_name:   Name of the IMAS file (in .mat format)
        :return:
        """
        file_core_profiles = scio.loadmat(file_name)
        self.shot = shot
        self.t0 = file_core_profiles['t0'][0, 0]
        profiles = list({key:val for key,val in file_core_profiles.items() if re.search(f"{'ids'}$", key)}.values())[0] #file_core_profiles['ids']
        output_flag = profiles['code'][0, 0]['output_flag'][0, 0].flatten()
        mask_core = (output_flag >= 0)

        idx_all = np.arange(0, output_flag.shape[0])[mask_core]
        self.time_profiles = profiles['time'][0, 0].flatten()[mask_core] - self.t0
        spatial_res = profiles['profiles_1d'][0, 0][0, 0][0, 0]['grid'][0, 0]['rho_tor_norm'].shape[0]
        self.rho_tor_norm_1d = np.zeros((self.time_profiles.shape[0], spatial_res))
        self.dens_1d = np.zeros((self.time_profiles.shape[0], spatial_res))
        self.dens_1d_lower = np.zeros((self.time_profiles.shape[0], spatial_res))
        self.dens_1d_upper = np.zeros((self.time_profiles.shape[0], spatial_res))
        for i in range(self.time_profiles.shape[0]):
            self.rho_tor_norm_1d[i, :] = profiles['profiles_1d'][0, 0].flatten()[idx_all[i]][0, 0]['grid'][0, 0][
                'rho_tor_norm'].flatten()
            self.dens_1d[i, :] = profiles['profiles_1d'][0, 0].flatten()[idx_all[i]][0, 0]['electrons'][0, 0][
                                     'density'].flatten() / 1e19
            self.dens_1d_lower[i, :] = profiles['profiles_1d'][0, 0].flatten()[idx_all[i]][0, 0]['electrons'][0, 0][
                                           'density_error_lower'].flatten() / 1e19
            self.dens_1d_upper[i, :] = profiles['profiles_1d'][0, 0].flatten()[idx_all[i]][0, 0]['electrons'][0, 0][
                                           'density_error_upper'].flatten() / 1e19

    def get_single_point(self, t):
        """
        Get the density profile of a time point (returned as an object)
        """
        idx_t = np.argmin(np.abs(t - self.time_profiles))
        rho_tor_norm_1d = self.rho_tor_norm_1d[idx_t, :]
        dens_1d = self.dens_1d[idx_t, :]
        dens_1d_lower = self.dens_1d_lower[idx_t, :]
        dens_1d_upper = self.dens_1d_upper[idx_t, :]
        dens_prof_single = DensityProfileSinglePoint(rho_tor_norm_1d, dens_1d, dens_1d_lower, dens_1d_upper,
                                                     self.time_profiles[idx_t], shot=self.shot)
        return dens_prof_single


def main():
    t = 5.2113
    shot = 53259
    dens_prof_all = DensityProfile()
    dens_prof_all.load_from_imas(f'data/WEST/{shot}/imas_core_profiles_{shot}_occ1.mat', shot=shot)
    dens_prof = dens_prof_all.get_single_point(t)
    dens_prof.plot_density_profile()


if __name__ == '__main__':
    main()