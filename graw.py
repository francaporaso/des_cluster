import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
#from astropy.constants import c, G, pc, M_sun
from astropy.table import Table
from scipy.integrate import simpson

from funcs import eq2p2, get_masked_idx_fast
from io import read_metacal, read_redmapper
from nzsource import calculate_median, sigma_crit, lensing_efficiency, read_nzsource

COSMO = FlatLambdaCDM(H0=100, Om0=0.3)

N = 10
RIN, ROUT = 0.2, 15.0 #Mpc/h
BINNING = 'log'

Source = read_metacal()
Lenses = read_redmapper()

if BINNING=='log':
    binspace = np.geomspace
elif BINNING=='lin':
    binspace = np.linspace
else:
    raise ValueError('BINNING must be "log" or "lin".')

def mask_zbin():
    pass

def partial_profile(inp):

    g_t_raw_num = np.zeros(N)
    g_x_raw_num = np.zeros(N)
    g_a_raw_den = np.zeros(N)
    N_inbin = np.zeros(N)

    ra0, dec0, z0, w_b0, w_b1, w_b2, w_b3 = inp

    DEGxMPC = COSMO.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
    psi = DEGxMPC*ROUT

    # get masked data
    idx = get_masked_idx_fast(psi, ra0, dec0, z0)
    catdata = Source[idx]

    # calculate transformation to polar coords
    rads, theta = eq2p2(
        np.deg2rad(catdata['ra_gal']), np.deg2rad(catdata['dec_gal']),
        np.deg2rad(ra0), np.deg2rad(dec0)
    )

    e1 = catdata['e_1']
    e2 = -catdata['e_2']
    R1 = catdata['r11']
    R2 = catdata['r22']
    #se usa el promedio entre ambos xq son muy similares ((R1-R2)/(0.5*(R1+R2)) < 0.1%)
    R = 0.5*(R1+R2)

    #get tangential ellipticities
    cos2t = np.cos(2.0*theta)
    sin2t = np.sin(2.0*theta)
    et = -(e1*cos2t+e2*sin2t)
    ex = (-e1*sin2t+e2*cos2t)

    #get weights
    w_s = catdata['unsheared_weights']
    w_b = np.array([w_b0, w_b1, w_b2, w_b3])

    #get redshift bins
    zbin = mask_zbin()

    ndots = binspace(RIN, ROUT, N+1)
    dig = np.digitize((np.rad2deg(rads)/DEGxMPC), ndots)

    for n_i in range(N):
        m_i = dig == n_i+1
        for b in range(4):  
            g_t_raw_num[n_i] += w_b[b]*np.sum(et[m_i][zbin[b]]*w_s[m_i][zbin[b]])
            g_x_raw_num[n_i] += w_b[b]*np.sum(ex[m_i][zbin[b]]*w_s[m_i][zbin[b]])
            g_a_raw_den[n_i] += w_b[b]*np.sum(R[m_i][zbin[b]]*w_s[m_i][zbin[b]])
            N_inbin[n_i] += np.count_nonzero(m_i)

    return g_t_raw_num, g_x_raw_num, g_a_raw_den, N_inbin


if __name__ == '__main__':
    print('a')