import multiprocessing
import logging
import numpy as np
import time
import emcee
from scipy.optimize import differential_evolution
from scipy import stats
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.problems.functional import FunctionalProblem
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.optimize import minimize as pm_minimize
from scipy.interpolate import CubicSpline
import scipy.io as scio
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Pool
from chord_geometry_int import ChordGeometryInterferometry
from magnetic_equilibrium import MagneticEquilibrium, MagneticEquilibriumSinglePoint
from interferometry import Interferometry, InterferometrySinglePoint
from density_profile import DensityProfile, DensityProfileSinglePoint
from forward_1d_triangle_int import compute_response_matrix


# The range of parameters and hyperparameters
E = 6   # number of knots
xi_min = 0.0
xi_max = 1.0
delta_xi = 0.1  # minimum spacing of the knot positions
f_min = np.zeros(E)
f_max = 20.0*np.ones(E)
f_max[-1] = 0.5
d_edge_min = -100.0
d_edge_max = -1.0
sigma_min = 0.05
sigma_max = 0.5
smooth_min = 0.002
smooth_max = 10.0


class Sampler(object):
    def __init__(self, interf: InterferometrySinglePoint, equi: MagneticEquilibriumSinglePoint, response, dens_prof=None):
        """
        Initialization of Sampler object
        :param interf: The interferometry data of a single time point
        :param equi: Magnetic equilibrium
        :param response: The response matrix for the forward model
        :param dens_prof: The electron density profile calculated by NICE (a DensityProfileSinglePoint object)
        """
        self.interf = interf
        self.equi = equi
        self.rho_tor_norm_1d = self.equi.rho_tor_norm_1d
        self.response = response[~self.interf.mask, :]
        self.dens_prof = dens_prof
        self.num_los = self.response.shape[0]
        self.num_points = self.response.shape[1]

        # Results
        self.samples = None
        self.sample_mean = None
        self.sample_median = None
        self.sample_mode = None
        self.profile_lower = None
        self.profile_upper = None

    def log_prior(self, xi, f, sigma, smooth, d_edge):
        """
        Logarithm of the prior distribution
        :param xi: Knot positions
        :param f: Function values at the knots
        :param sigma: Error level of the measurements
        :param smooth: Smoothing factor
        :param d_edge: The derivative at the edge
        :return: The value of log prior
        """
        xi_intervals = np.hstack([xi, xi_max]) - np.hstack([xi_min, xi])
        xi_interval_min = xi_intervals.min()
        within_limits = np.logical_and.reduce([*np.logical_and(f > f_min, f < f_max),
                                               np.logical_and(sigma > sigma_min, sigma < sigma_max),
                                               np.logical_and(smooth > smooth_min, smooth < smooth_max),
                                               np.logical_and(d_edge > d_edge_min, d_edge < d_edge_max)])
        if not within_limits or xi_interval_min < delta_xi:
            return -np.inf

        xi = np.hstack([xi_min, xi, xi_max])
        cs = CubicSpline(xi, f, bc_type=((1, 0), (1, d_edge)))
        second_derivatives = cs.derivative(2)(xi)
        integrand = second_derivatives ** 2
        intervals = xi[1:] - xi[:-1]
        int_bar = (integrand[:-1] + integrand[1:]) / 2
        penalty = np.sum(int_bar * intervals)

        log_prior_xi = np.log(np.math.factorial(E - 2)) - (E - 2) * np.log(xi_max - xi_min - (E - 1) * delta_xi)
        log_prior_f = - np.sum(np.log(f_max))
        log_prior_sigma = -np.log(np.log(sigma_max / sigma_min)) - np.log(sigma)
        log_prior_smooth = -np.log(np.log(smooth_max / smooth_min)) - np.log(smooth)
        return log_prior_xi + log_prior_f + log_prior_sigma + log_prior_smooth - smooth * penalty

    def log_likelihood(self, xi, f, sigma, d_edge):
        """
        Logarithm of the likelihood
        :param xi: Knot positions
        :param f: Function values at the knots
        :param sigma: Error level of the measurements
        :param d_edge: The derivative at the edge
        :return: The value of log likelihood
        """
        xi = np.hstack([xi_min, xi, xi_max])
        cs = CubicSpline(xi, f, bc_type=((1, 0), (1, d_edge)))
        ne = cs(self.rho_tor_norm_1d)
        residual = self.interf.lid.compressed() - np.matmul(self.response, ne)
        log_like = -0.5 * (self.num_los * np.log(2*np.pi) + 2 * self.num_los * np.log(sigma)
                           + np.sum(residual**2/sigma**2))
        return log_like

    def log_prob(self, x):
        """
        Logarithm of the posterior distribution
        :param x: All parameters to be sampled
        :return: The value of log posterior
        """
        xi = np.sort(x[:E - 2])
        f = np.sort(x[E - 2:2 * E - 2])[::-1]
        sigma = x[2 * E - 2]
        smooth = x[2 * E - 1]
        d_edge = x[2 * E]
        log_prior = self.log_prior(xi, f, sigma, smooth, d_edge)
        if not np.isfinite(log_prior):
            return -np.inf
        else:
            return log_prior + self.log_likelihood(xi, f, sigma, d_edge)

    def sample_to_profile(self, x):
        """
        Mapping the sampled parameters to a density profile
        :param x: Sampled parameters
        :return: Electron density profile
        """
        d_edge = x[2 * E]
        xi = np.hstack([xi_min, np.sort(x[:E-2]), xi_max])
        f = np.sort(x[E-2:2*E-2])[::-1]
        cs = CubicSpline(xi, f, bc_type=((1, 0), (1, d_edge)))
        return cs(self.rho_tor_norm_1d)

    def samples_to_profiles(self, raw_samples):
        """
        Mapping all samples of parameters to density profiles
        :param raw_samples: All sampled parameters
        :return: Electron density profiles
        """
        param_chunks = [(self.sample_to_profile, 1, sub_arr)
                        for sub_arr in np.array_split(raw_samples, multiprocessing.cpu_count())]
        with Pool() as pool:
            individual_profiles = pool.starmap(np.apply_along_axis, param_chunks)
        return np.concatenate(individual_profiles)

    def find_map_with_differential_evolution(self):
        """
        Find MAP estimate using differential evolution algorithm
        :return: MAP estimate
        """
        bounds_xi = [(xi_min, xi_max) for _ in range(E - 2)]
        bounds_f = [(f_min[i], f_max[i]) for i in range(E)]
        bounds_sigma = [(sigma_min, sigma_max)]
        bounds_smooth = [(smooth_min, smooth_max)]
        bounds_d_edge = [(d_edge_min, d_edge_max)]
        negative_logpdf = lambda x: -self.log_prob(x)
        x_map = differential_evolution(negative_logpdf,
                                       bounds_xi + bounds_f + bounds_sigma + bounds_smooth + bounds_d_edge).x
        return x_map

    def find_map_with_pattern_search(self):
        """
        Find MAP estimate using differential evolution algorithm
        :return: MAP estimate
        """
        x_start = np.hstack([np.linspace(xi_min+delta_xi, xi_max-delta_xi, E-2),
                             (f_min+f_max)/2,
                             (sigma_min+sigma_max)/2,
                             (smooth_min+smooth_max)/2,
                             (d_edge_min+d_edge_max)/2])
        x_lower = np.hstack([xi_min*np.ones(E-2), f_min, sigma_min, smooth_min, d_edge_min])
        x_upper = np.hstack([xi_max*np.ones(E-2), f_max, sigma_max, smooth_max, d_edge_max])
        negative_logpdf = lambda x: -self.log_prob(x)
        problem = FunctionalProblem(2 * E + 1, [negative_logpdf], xl=x_lower, xu=x_upper)
        algorithm = PatternSearch()
        termination = RobustTermination(DesignSpaceTermination(tol=1e-4), period=10)
        x_map = pm_minimize(problem, algorithm, termination=termination, verbose=False, x0=x_start).X
        return x_map

    def sample(self, burnin=2000, num_samples=2000, opt='de', file_name=None):
        """
        MCMC sampling
        :param burnin: Number of burn-in steps
        :param num_samples: Number of samples
        :param opt: Optimization algorithm, either 'de' for differential evolution or 'ps' for pattern search
        :param file_name: If specified, save the samples to file
        :return:
        """
        # Find MAP estimate
        print("Estimating MAP ...")
        start = time.time()
        if opt == 'de':
            x_map = self.find_map_with_differential_evolution()
        else:
            x_map = self.find_map_with_pattern_search()
        time1 = time.time()
        print(f"MAP point found. Cost time: {time1-start:.3f} s")
        print(f"Negative logpdf: {self.log_prob(x_map)}")
        ne_map = self.sample_to_profile(x_map)
        print(f'sigma = {x_map[2 * E - 2]}')
        print(f'smooth_factor = {x_map[2 * E - 1]}')
        print(f'first derivative at the edge = {x_map[2 * E]}')
        plt.plot(self.rho_tor_norm_1d, ne_map, label='MAP')
        if self.dens_prof is not None:
            plt.plot(self.rho_tor_norm_1d, self.dens_prof.dens_1d, label='NICE')
        plt.title('MAP estimate')
        plt.legend()
        plt.show()
        # Sampling
        ndim = 2 * E + 1
        nwalkers = ndim * 2
        x_start = x_map + 0.01 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)
        print("MCMC sampling starts")
        time1 = time.time()
        print("Burn-in steps ...")
        state = sampler.run_mcmc(x_start, burnin, progress=True)
        sampler.reset()
        print('Sampling ...')
        sampler.run_mcmc(state, num_samples, progress=True)
        time2 = time.time()
        samples = sampler.get_chain(flat=True)
        print(f'Sampling finished. Cost time: {time2 - time1} s')
        print(f'Mapping free parameters to splines ...')
        self.samples = self.samples_to_profiles(samples)
        print(f'Cost time: {time.time() - time2} s')
        self.sample_mean = self.samples.mean(axis=0)
        self.sample_median = np.median(self.samples, axis=0)
        self.sample_mode = stats.mode(self.samples, axis=0, keepdims=False).mode.flatten()
        self.profile_lower = np.percentile(self.samples, 2.5, axis=0)
        self.profile_upper = np.percentile(self.samples, 97.5, axis=0)
        if file_name is not None:
            # file_name = f'{self.interf.shot}_{self.interf.t:.2f}s_mcmc_samples.mat'
            scio.savemat(file_name, {'samples': self.samples,
                                     'ne_mean': self.sample_mean,
                                     'ne_median': self.sample_median,
                                     'ne_mode': self.sample_mode,
                                     'rho': self.rho_tor_norm_1d,
                                     'ne_lower': self.profile_lower,
                                     'ne_upper': self.profile_upper})

    def load_samples(self, file_name):
        """
        Load samples from file
        """
        file = scio.loadmat(file_name)
        self.samples = file['samples']
        self.sample_mean = file['ne_mean'].flatten()
        self.sample_median = file['ne_median'].flatten()
        self.sample_mode = file['ne_mode'].flatten()
        self.profile_lower = file['ne_lower'].flatten()
        self.profile_upper = file['ne_upper'].flatten()

    @mpl.rc_context({'image.cmap': 'jet', 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'figure.dpi': 100, 'lines.linewidth': 1.0})
    def plot_posterior(self, line='mean'):
        """
        Plot the posterior density profile reconstruction of a single time point
        """
        if line == 'mean':
            line_data = self.sample_mean
        elif line == 'median':
            line_data = self.sample_median
        else:
            line_data = self.sample_mode
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # The reconstructed lid
        lid_recon = np.matmul(self.response, line_data)

        # Plot reconstructed density profile
        ax1.plot(self.rho_tor_norm_1d, line_data, color='blue', linewidth=2, label='Bayesian')
        ax1.fill_between(self.rho_tor_norm_1d, self.profile_lower, self.profile_upper, color='blue', alpha=0.3)

        if self.dens_prof is not None:
            ax1.plot(self.rho_tor_norm_1d, self.dens_prof.dens_1d, color='red', label='NICE')
            ax1.plot(self.rho_tor_norm_1d, self.dens_prof.dens_1d_lower, 'r--')
            ax1.plot(self.rho_tor_norm_1d, self.dens_prof.dens_1d_upper, 'r--')
            ax1.legend()
        ax1.set_title(f'# {self.interf.shot} electron density profile (t = {self.interf.t:.2f} s)')
        ax1.set_xlabel(r'$\rho$')
        ax1.set_ylabel(r'$\mathrm{n_e}(10^{19} \mathrm{m}^{-3})$')

        # Plot lid
        ax2.plot(self.interf.channels.compressed(), lid_recon, 'b*', label='Bayesian')
        if self.equi.lid_nice is not None:
            ax2.plot(self.interf.channels.data, self.equi.lid_nice, 'r.', label='NICE')
        ax2.plot(self.interf.channels.compressed(), self.interf.lid.compressed(), 'g^', label='measured')
        ax2.legend(loc='best')
        ax2.set_xlabel('channel')
        ax2.set_ylabel(r'Line integrated density ($10^{19} \mathrm{m}^{-2}$)')
        ax2.set_title('Measured and calculated signals')
        ax2.set_xticks(self.interf.channels.data, self.interf.channels.data)
        plt.show()


def sampler_spline_main():
    t = 5.91
    shot = 53259
    # t = 1.4
    # shot = 55191
    geo = ChordGeometryInterferometry(f'data/WEST/{shot}/los_interferometer_{shot}.mat')

    equi_all = MagneticEquilibrium()
    equi_all.load_from_imas(f'data/WEST/{shot}/imas_equilibrium_{shot}.mat', shot=shot)
    equi = equi_all.get_single_point(t)

    interf_all = Interferometry()
    interf_all.load_from_imas(f'data/WEST/{shot}/imas_interferometer_{shot}.mat', shot=shot)
    interf = interf_all.get_single_point(t)

    dens_prof_all = DensityProfile()
    dens_prof_all.load_from_imas(f'data/WEST/{shot}/imas_core_profiles_{shot}_occ1.mat', shot=shot)
    dens_prof = dens_prof_all.get_single_point(t)

    response = compute_response_matrix(geo, equi, dens_prof.rho_tor_norm_1d)

    sampler = Sampler(interf=interf, equi=equi, response=response, dens_prof=dens_prof)
    sampler.sample(opt='de')
    # sampler.sample(opt='de', file_name='53259_5.21s_mcmc_samples.mat')
    # sampler.load_samples('53259_5.21s_mcmc_samples.mat')
    sampler.plot_posterior(line='mean')


if __name__ == '__main__':
    sampler_spline_main()
