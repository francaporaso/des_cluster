import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
#from astropy.constants import c, G, pc, M_sun
from astropy.table import Table
from scipy.integrate import simpson

from funcs import eq2p2#, get_masked_idx_fast
#from io import *
from nzsource import calculate_median, sigma_crit, lensing_efficiency, read_nzsource

COSMO = FlatLambdaCDM(H0=100, Om0=0.3)

NBINS = 10
RIN, ROUT = 0.2, 15.0 #Mpc/h
BINNING = 'log'


ZMED = np.array([0.285, 0.476, 0.743, 0.942])

if BINNING=='log':
    binspace = np.geomspace
elif BINNING=='lin':
    binspace = np.linspace
else:
    raise ValueError('BINNING must be "log" or "lin".')


def read_redmapper(filename='../cats/DESY3/desy3_redmapper_cluster-ws.fits'):
    return Table.read(filename)

def read_source(filename='../cats/DESY3/desy3_metacal-unsheared-zbins_25314.fits'):
    return Table.read(filename)

Source = read_source() # metacal file
Lenses = read_redmapper() # redmapper

def get_masked_data(psi, ra0, dec0, z0, wb):
    mask_sky = (Source['ra_gal'] < (ra0+psi))&(Source['ra_gal'] > (ra0-psi))&(Source['dec_gal'] < (dec0+psi))&(Source['dec_gal'] > (dec0-psi))
    for i in range(4):
        if z0 > ZMED[i]:
            wb[i] = 0.0
    return mask_sky, wb

def partial_profile(inp):

    g_t_raw_num = np.zeros(NBINS)
    g_x_raw_num = np.zeros(NBINS)
    g_a_raw_den = np.zeros(NBINS)
    N_inbin = np.zeros(NBINS)

    ra0, dec0, z0, *w_b = inp

    DEGxMPC = COSMO.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
    psi = DEGxMPC*ROUT

    # get masked data
    mask, w_b = get_masked_data(psi, ra0, dec0, z0, w_b)
    catdata = Source[mask]

    # calculate transformation to polar coords
    rads, theta = eq2p2(
        np.deg2rad(catdata['ra_gal']), np.deg2rad(catdata['dec_gal']),
        np.deg2rad(ra0), np.deg2rad(dec0)
    )

    #get weights
    w_s = catdata['weight']

    e1 = catdata['e_1']
    e2 = -catdata['e_2']
    R1 = catdata['r11']
    R2 = catdata['r22']
    #se usa el promedio entre ambos xq son muy similares
    #((R1-R2)/(0.5*(R1+R2)) < 0.1%)
    R = 0.5*(R1+R2)*w_s

    #get weighted tangential ellipticities
    cos2t = np.cos(2.0*theta)
    sin2t = np.sin(2.0*theta)
    et = -(e1*cos2t+e2*sin2t)*w_s
    ex = (-e1*sin2t+e2*cos2t)*w_s

    ndots = binspace(RIN, ROUT, NBINS+1)
    dig = np.digitize((np.rad2deg(rads)/DEGxMPC), ndots)

    for n_i in range(NBINS):
        m_i = dig == n_i+1
        for b in range(4):
            zbin = catdata['bhat'] == b
            g_t_raw_num[n_i] += w_b[b]*np.sum(et[m_i][zbin])
            g_x_raw_num[n_i] += w_b[b]*np.sum(ex[m_i][zbin])
            g_a_raw_den[n_i] += w_b[b]*np.sum(R[m_i][zbin])
            N_inbin[n_i] += np.count_nonzero(m_i)

    return g_t_raw_num, g_x_raw_num, g_a_raw_den, N_inbin


def main():

    l0 = Lenses[Lenses['mem_match_id']==69]
    g_t_raw_num, g_x_raw_num, g_a_raw_den, N_inbin = partial_profile([l0['ra_gal'], l0['dec_gal'], l0['redshift'], l0['wb_0'], l0['wb_1'], l0['wb_2'], l0['wb_3']])

    g_t_raw = g_t_raw_num/g_a_raw_den
    g_x_raw = g_x_raw_num/g_a_raw_den

    r = binspace(RIN, ROUT, NBINS)
    plt.scatter(r, g_t_raw, s=5)
    plt.scatter(r, g_x_raw, s=5, m='x', color='gray')
    plt.show()



if __name__ == '__main__':
    print('a')
    main()

# how to get the bhat for the metacal cat
#ids1 = table1['id']
#ids2 = table2['id']
#
## sort table2
#order2 = np.argsort(ids2)
#ids2_sorted = ids2[order2]
#
## find matches
#idx = np.searchsorted(ids2_sorted, ids1)
#
## valid matches mask
#mask = (idx < len(ids2)) & (ids2_sorted[idx] == ids1)
#
#t1_matched = table1[mask]
#t2_matched = table2[order2[idx[mask]]]
