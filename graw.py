import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c, G, pc, M_sun
from astropy.table import Table
from scipy.integrate import simpson

from funcs import sompzcat_load, metacal_load

COSMO = FlatLambdaCDM(H0=100, Om0=0.3)
SC_CTE = (c.value**2.0/(4.0*np.pi*G.value))*(pc.value/M_sun.value)*1.0e-6

sompz = sompzcat_load()
metacal = metacal_load()

pz_b = np.array(sompz['bin0'])
z_s = np.arange(0.0,6.0,0.01)

def sigma_crit(z_l, z_s):
    d_ls = COSMO.angular_diameter_distance_z1z2(z_l, z_s).value
    d_l  = COSMO.angular_diameter_distance(z_l).value*pc.value*1.0e6
    d_s  = COSMO.angular_diameter_distance(z_s).value
    return np.where(d_ls>0, ((c.value**2.0)/(4.0*np.pi*G.value*d_l))*(d_s/d_ls)*(pc.value**2/M_sun.value), 0.0)

def inverse_sigma_crit(z_l, z_s):
    # eq 4 bocquet+2024
    d_l  = COSMO.angular_diameter_distance(z_l).value
    d_s  = COSMO.angular_diameter_distance(z_s).value
    d_ls = COSMO.angular_diameter_distance_z1z2(z_l, z_s).value
    return (1.0/SC_CTE)*(d_l/d_s)*np.max([0.0,d_ls])

def mean_sigma_crit_b(z_l, pz):
    # eq 6 Bocquet+2024
    'pz_b : ndarray = mean source redshift distribution as "function" of z_s'
#    z_s = np.linspace(0.0,1.6,0.1) # source redshift range... CHANGE ACCORDINGLY
    return simpson(pz*inverse_sigma_crit(z_l, z_s), x=z_s)

if __name__ == '__main__':
    z_l=float(input())
    print(mean_sigma_crit_b(z_l))