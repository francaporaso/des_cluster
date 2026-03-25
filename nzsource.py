from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
#from astropy.cosmology import FlatLambdaCDM
from astropy.constants import G,c,M_sun,pc
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

#cosmo = FlatLambdaCDM(Om0=0.3, H0=100)

SC_CONSTANT : float = (c**2.0/(4.0*np.pi*G)).to('Msun/Mpc').value # h cancel out

def read_nzsource():
    nzbins = {}
    nzperc = {}
    nzmean = {}

    with fits.open('../../cats/DESY3/nz_source_allrealizations.fits') as f:
        
        zmid = f[1].data['Z_MID']

        for i in range(4):
            nzmean[f'{i}'] = f[1].data[f'BIN{i+1}']
            nzbins[f'{i}'] = np.array([f[j+2].data[f'BIN{i+1}'] for j in range(1000)])
            nzperc[f'{i}'] = np.percentile(nzbins[f'{i}'], [16,84], axis=0)
    return zmid, nzbins, nzperc, nzmean

def sigma_crit(z_l, z_s):
    
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    d_ls[d_ls<=0.0] = 0.0
    d_l  = cosmo.angular_diameter_distance(z_l).value*cosmo.h # from physical distance to distance/h we need to multiply
    d_s  = cosmo.angular_diameter_distance(z_s).value
    return SC_CONSTANT*(d_s/(d_ls*d_l))

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