import scipy.io as scio


class ChordGeometryInterferometry(object):
    def __init__(self, file_name):
        file_los = scio.loadmat(file_name)
        self.los_r_start = file_los['RPOS1'].flatten()
        self.los_r_end = file_los['RPOS2'].flatten()
        self.los_z_start = file_los['ZPOS1'].flatten()
        self.los_z_end = file_los['ZPOS2'].flatten()
        self.num_los = self.los_r_start.shape[0]
