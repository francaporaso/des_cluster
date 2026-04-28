import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import M_sun, pc
from colossus.cosmology import cosmology

pc : float = pc.value # parsec in m
Msun : float = M_sun.value # solar mass in kg

h = 1.0
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.25, Ob0=0.0)
cosmology.fromAstropy(cosmo, ns=0.96, sigma8=0.811, cosmo_name='etc')
SQPI : float = np.sqrt(np.pi)

def rho_mean(z):
    '''densidad media en Msun/(pc**2 Mpc)'''
    p_cr0 = cosmo.critical_density(0).to('Msun/(pc**2 Mpc)').value
    a = cosmo.scale_factor(z)
    out = p_cr0*cosmo.Om0/a**3
    return out