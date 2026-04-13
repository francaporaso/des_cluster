from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
#from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G,c,M_sun,pc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.interpolate import interp1d

#cosmo = FlatLambdaCDM(Om0=0.3, H0=100)

SC_CONSTANT : float = (c**2.0/(4.0*np.pi*G)).to('Msun/pc').value

def read_nzsource(filename='../../cats/DESY3/nz_source_allrealizations.fits'):
    nzbins = {}
    nzperc = {}
    nzmean = {}

    with fits.open(filename) as f:

        zmid = f[1].data['Z_MID']

        for i in range(4):
            nzmean[f'{i}'] = f[1].data[f'BIN{i+1}']
            nzbins[f'{i}'] = np.array([f[j+2].data[f'BIN{i+1}'] for j in range(1000)])
            nzperc[f'{i}'] = np.percentile(nzbins[f'{i}'], [16,84], axis=0)
    return zmid, nzbins, nzperc, nzmean

def sigma_crit(z_l, z_s):
    '''
    Sigma_crit in M_sun/pc**2. Units of SC_CONSTANT are M_sun/pc.
    Since the units of two of the three angular distances cancel out, 
    only transform one of them to pc (d_ls). 
    
    If needed, to go from physical distance to distance/h is necesary 
    to multiply distance by the value of h.
    '''

    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value * 1e6
    #d_ls[d_ls<=0.0] = 0.0
    d_l  = cosmo.angular_diameter_distance(z_l).value 
    d_s  = cosmo.angular_diameter_distance(z_s).value
    sc = SC_CONSTANT*(d_s/(d_ls*d_l))
    sc[sc<=0.0] = 0.0
    return sc

def lensing_efficiency(z_l, z_s, nz):
    '''
    eq. 6 from Bocquet+2024
    z_l (float) : lens redshift
    z_s (array) : source redshift (zmid of nzsource.fits)
    nz (array) : source redshift distribution
    '''
    normfactor = 1.0/simpson(nz, z_s) # normalization
    integrand = nz/sigma_crit(z_l, z_s)
    return normfactor*simpson(integrand, z_s)


def calculate_median(z, pdf):
    y = np.cumsum(pdf) - 0.5
    for i in range(len(y)-1):
        if np.sign(y[i]) != np.sign(y[i+1]):
            f = interp1d(z[i:i+2], y[i:i+2], kind='linear')
            root = brentq(f, z[i], z[i+1])
            return root

if __name__ == '__main__':

    zmid, nzbins, nzperc, nzmean = read_nzsource()

    # plt.figure()
    # for i in range(4):
    #     plt.fill_between(zmid, nzperc[f'{i}'][0], nzperc[f'{i}'][1], color=f'C{i}', alpha=0.25)
    #     plt.plot(zmid, nzmean[f'{i}'], c=f'C{i}')
    # plt.show()

    zlens = np.linspace(0.0,2.0,50)
    inv_sigma_cb = {f'{j}': np.array([lensing_efficiency(zi, zmid, nzmean[f'{j}']) for zi in zlens]) for j in range(4)}

    plt.figure()
    for j in range(4):
        plt.plot(zlens, inv_sigma_cb[f'{j}'])
    plt.show()
