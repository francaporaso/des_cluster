import numpy as np
from dataclasses import dataclass
from astropy.io import fits
from astropy.table import Table

@dataclass
class DataProfile:
    redshift : np.float64
    Njk : np.float64
    R : np.ndarray
    DSigma_t : np.ndarray
    covDSt : np.ndarray
    DSigma_x : np.ndarray
    covDSx : np.ndarray
    Sigma : np.ndarray|None = None
    covS : np.ndarray|None = None

    def plot_profile(self, observable='sigma', **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('R')
        if observable=='sigma':
            ax.set_ylabel('$\\Sigma$')
            ax.errorbar(self.R, self.Sigma, np.sqrt(np.diag(self.covS)),
                         **kwargs)
        elif observable=='delta_sigma':
            ax.set_ylabel('$\\Delta\\Sigma$')
            ax.errorbar(self.R, self.DSigma_t, np.sqrt(np.diag(self.covDSt)),
                         **kwargs)
            ax.errorbar(self.R, self.DSigma_x, np.sqrt(np.diag(self.covDSx)),
                        fmt='x', **kwargs)
        
        fig.show()
        return fig

    def plot_cov(self, observable='sigma', **kwargs):
        import matplotlib.pyplot as plt
        if observable == 'sigma':
            plt.imshow(self.covS)
        elif observable == 'delta_sigma':
            plt.imshow(self.covDSt)
        else:
            plt.imshow(self.covDSx)

        plt.show()

# the **kwargs requires giving the arg name when calling this function
# ex: data = read_dataprofile_fits(name='myprofile.fits')
# this is not going to work: data = read_dataprofile_fits('myprofile.fits').
def read_dataprofile_fits(*args, **kwargs):
    #binspace = (np.linspace if binning=='lin' else np.geomspace)
    with fits.open(*args, **kwargs) as f:
        hd = f[0].header
        dt = f[1].data
        data = DataProfile(
            R = dt['R'],
            redshift = hd['Z_MEAN'],
            Njk = hd['NJK'],
            DSigma_t = dt['DSigma_t'],
            covDSt = np.identity(hd['NBINS']), #f[2].data,
            DSigma_x = dt['DSigma_x'],
            covDSx = np.identity(hd['NBINS']), # f[1].data,
            #Sigma = dt['Sigma'],
            #covS = f[2].data,
        )
    return data

def save_chains_h5():
    pass