import numpy as np
import numpy.ma as ma
import scipy.io as scio
import re


class InterferometrySinglePoint(object):
    def __init__(self, channels, lid, t=0.0, shot=0, excluded_channels=None):
        """
        The interferometry data of a single time point
        :param channels: Channel number of each line of sight
        :param lid: Line integrated density (measured data) (1e19 m-2)
        :param t: Time
        :param shot: Shot number
        :param excluded_channels: The "outlier" channels to be excluded from analysis
        """
        self.mask = np.zeros_like(channels, dtype=bool)
        lid[lid <= 0] = 1e-8
        if excluded_channels is not None:
            self.mask[excluded_channels] = True
        self.channels = ma.array(channels, mask=self.mask)
        self.lid = ma.array(lid, mask=self.mask)
        self.t = t
        self.shot = shot


class Interferometry(object):
    def __init__(self):
        self.t0 = 0.0             # The beginning of the plasma
        self.time_interf = None   # Time points of interferometry data (relative to t0)
        self.channels = None       # Channel number of each line of sight
        self.empty_channels = []  # Empty channels
        self.channel_mask = None   # Mask of the excluded channels
        self.lid = None           # Line integrated density (measured data) (1e19 m-2)
        self.shot = 0

    def load_from_imas(self, file_name, shot=0):
        """
        Load interferometry data from IMAS file
        :param file_name:   Name of the IMAS file (in .mat format)
        :param shot: shot number
        :return:
        """
        file_interf = scio.loadmat(file_name)
        self.t0 = file_interf['t0'][0, 0]
        self.shot = shot
        interf = list({key:val for key,val in file_interf.items() if re.search(f"{'ids'}$", key)}.values())[0]#file_interf['ids']
        self.time_interf = interf['time'][0, 0].flatten() - self.t0
        channel_data = interf['channel'][0, 0]
        self.channels = np.arange(1, channel_data.shape[0] + 1)
        self.lid = np.zeros((self.time_interf.shape[0], self.channels.shape[0]))
        for i in range(self.channels.shape[0]):
            ne_line = channel_data[i, 0][0, 0]['n_e_line']['data'][0, 0].flatten() / 2e19
            if ne_line.shape[0] > 0:
                self.lid[:, i] = ne_line
            else:
                self.empty_channels.append(i)

    def get_single_point(self, t, excluded_channels=None):
        """
        Get the interferometry data of a time point (returned as an object)
        :param t: Time
        :param excluded_channels: The "outlier" channels to be excluded from analysis
        """
        idx_t = np.argmin(np.abs(t - self.time_interf))
        channels = self.channels
        lid = self.lid[idx_t, :]
        if excluded_channels is None:
            excluded_channels = self.empty_channels
        else:
            excluded_channels = list(set(excluded_channels + self.empty_channels))
        interf_single = InterferometrySinglePoint(channels, lid, self.time_interf[idx_t], self.shot, excluded_channels)
        return interf_single

