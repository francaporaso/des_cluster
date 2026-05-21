import numpy as np
from dataclasses import dataclass
from astropy.io import fits
from astropy.table import Table

@dataclass
class DataProfile:
    redshift : np.float64
    Njk : np.float64
    R : np.ndarray
    DSigma_t : np.ndarray | None = None
    covDSt : np.ndarray | None = None
    DSigma_x : np.ndarray | None = None
    covDSx : np.ndarray | None = None
    Sigma : np.ndarray | None = None
    covS : np.ndarray | None = None
    Gamma_tcos : np.ndarray | None = None
    covGtc : np.ndarray | None = None
    Gamma_xsin : np.ndarray | None = None
    covGxs : np.ndarray | None = None

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
def read_dataprofile_fits(filename, profile='all'):
    #binspace = (np.linspace if binning=='lin' else np.geomspace)
    DSigma_t = None
    covDSt = None
    DSigma_x = None
    covDSx = None
    Gamma_tcos = None
    covGtc = None
    Gamma_xsin = None
    covGxs = None

    with fits.open(*args, **kwargs) as f:
        hd = f[0].header
        dt = f[1].data

        R = dt['R']
        redshift = hd['Z_MEAN'],
        Njk = hd['NJK'],

        if profile=='monopole':
            DSigma_t = dt['DSigma_t']
            covDSt = f[2].data
            DSigma_x = dt['DSigma_x']
            covDSx = f[3].data

        elif profile=='quadrupole':
            Gamma_tcos = dt['Gamma_tcos']
            covGtc = f[4].data
            Gamma_xsin = dt['Gamma_xsin']
            covGxs = f[5].data

        elif profile=='all':
            DSigma_t = dt['DSigma_t']
            covDSt = f[2].data
            DSigma_x = dt['DSigma_x']
            covDSx = f[3].data
            Gamma_tcos = dt['Gamma_tcos']
            covGtc = f[4].data
            Gamma_xsin = dt['Gamma_xsin']
            covGxs = f[5].data

    data = DataProfile(
        redshift = redshift,
        R = R,
        Njk = Njk,
        DSigma_t = DSigma_t,
        covDSt = covDSt,
        DSigma_x = DSigma_x,
        covDSx = covDSx,
        Gamma_tcos = Gamma_tcos,
        covGtc = covGtc,
        Gamma_xsin = Gamma_xsin,
        covGxs = covGxs
    )

    return data

def save_chains_h5():
    pass
