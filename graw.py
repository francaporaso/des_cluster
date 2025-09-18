import numpy as np
import h5py
from astropy.cosmology import LambdaCDM
from astropy.constants import c, G, pc, M_sun
from astropy.table import Table
from scipy.integrate import simpson

_cosmo = LambdaCDM(H0=100.0, Om0=0.3, Ode0=0.7)

sompzcat = h5py.File('../cats/DESY3/desy3_sompz_v050.h5')['catalog']['sompz']
sourcecat = Table.read('../cats/DESY3/desy3_unsheared.fits', format='fits')

pz_b = np.array(sompzcat['pzdata']['bin0'])
z_s = np.arange(0.0,6.0,0.01)

def sigma_crit(z_l, z_s):
    d_ls = _cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    d_l  = _cosmo.angular_diameter_distance(z_l).value*pc.value*1.0e6
    d_s  = _cosmo.angular_diameter_distance(z_s).value
    return np.where(d_ls>0, ((c.value**2.0)/(4.0*np.pi*G.value*d_l))*(d_s/d_ls)*(pc.value**2/M_sun.value), 0.0)

def mean_sigma_crit_b(z_l):
    # ec 6 Bocquet+2024 (SPT + DES + HST lensing)
    'pz_b : ndarray = mean source redshift distribution as "function" of z_s'
#    z_s = np.linspace(0.0,1.6,0.1) # source redshift range... CHANGE ACCORDINGLY
    return simpson(pz_b*sigma_crit(z_l, z_s), x=z_s)

if __name__ == '__main__':
    z_l=float(input())
    print(mean_sigma_crit_b(z_l))