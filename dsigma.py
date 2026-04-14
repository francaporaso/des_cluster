import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
#from astropy.constants import c, G, pc, M_sun
from astropy.table import Table
import healpy as hp
#from scipy.integrate import simpson
import matplotlib.pyplot as plt
from time import time

from funcs import eq2p2
#from io import *
from nzsource import calculate_median, sigma_crit, lensing_efficiency, read_nzsource

# Fixed globals
COSMO = FlatLambdaCDM(H0=100, Om0=0.3)
NSIDE = 128
ZMED = np.array([0.285, 0.476, 0.743, 0.942]) # median redshift of source distribution
REDSHIFT = 'redshift' # name of the redshift col in source table
SOURCE = None
LENSES = None
PIX_TO_IDX : dict = {}

# Tuning globals
NBINS = 15
RIN, ROUT = 0.2, 15.0 #Mpc/h
BINNING = 'log'
PLOT = False

if BINNING=='log':
    binspace = np.geomspace
elif BINNING=='lin':
    binspace = np.linspace
else:
    raise ValueError('BINNING must be "log" or "lin".')


def read_redmapper(filename='../cats/DESY3/desy3_redmapper_cluster-ws.fits'):
    return Table.read(filename)

def read_source(filename='../cats/DESY3/desy3_metacal-unsheared-zbins_w-pix128_25314.fits'):
    return Table.read(filename)

def init_globals():
    global SOURCE, LENSES
    global PIX_TO_IDX

    # reading catalogs
    SOURCE = read_source() # metacal file
    LENSES = read_redmapper() # redmapper
    
    # making a dict of healpix idx for fast query
    upix, split_idx = np.unique(SOURCE['pix'], return_index=True)
    split_idx = np.append(split_idx, len(SOURCE))
    for i, pix in enumerate(upix):
        PIX_TO_IDX[int(pix)] = np.arange(split_idx[i], split_idx[i+1])

def get_masked_square(psi, ra0, dec0, z0, wb):
    mask_sky = (SOURCE['ra_gal'] < (ra0+psi))&(SOURCE['ra_gal'] > (ra0-psi))&(SOURCE['dec_gal'] < (dec0+psi))&(SOURCE['dec_gal'] > (dec0-psi))
    for i in range(4):
        if z0 > ZMED[i]:
            wb[i] = 0.0
    return mask_sky, wb

def get_masked_idx_fast(psi, ra0, dec0, z0, wb):
    '''
    objects are selected by pixel on a disc of rad=psi+pad where pad = 0.1*psi
    uses prebuilt _PIX_TO_INDEX dict
    returns the indices of _S where to select
    '''

    pix_idx = hp.query_disc(
        NSIDE,
        vec=hp.ang2vec(ra0, dec0, lonlat=True),
        radius=np.deg2rad(psi*1.1)
    )

    idx_arrays = np.concatenate([
        PIX_TO_IDX[p]
        for p in pix_idx
        if p in PIX_TO_IDX
    ])

    ### mask_z = SOURCE[REDSHIFT][idx_arrays] > (z0+0.1) # for source w redshift per source
    for i in range(4):
        if z0 > ZMED[i]:
            wb[i] = 0.0

    return idx_arrays, wb

def partial_profile(inp):
    '''
    Profile of reduced shear g_t(r) as in eq. 6 of Grandis et al. (2024)
    '''

    dsigma_t_num = np.zeros(NBINS)
    dsigma_x_num = np.zeros(NBINS)
    response_sum = np.zeros(NBINS)
    n_bin = np.zeros(NBINS)
    n_sl_sum = np.zeros(NBINS)

    ra0, dec0, z0, *w_b = inp

    DEGxMPC = COSMO.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
    psi = DEGxMPC*ROUT

    # get masked data
    mask, w_b = get_masked_idx_fast(psi, ra0, dec0, z0, w_b)
    catdata = SOURCE[mask]

    # calculate transformation to polar coords
    rads, theta = eq2p2(
        np.deg2rad(catdata['ra_gal']), np.deg2rad(catdata['dec_gal']),
        np.deg2rad(ra0), np.deg2rad(dec0)
    )

    #get weights
    w_s = catdata['weight']

    e1 = -catdata['e_1']
    e2 = catdata['e_2']
    R1 = catdata['r11']
    R2 = catdata['r22']
    #se usa el promedio entre ambos xq son muy similares
    #((R1-R2)/(0.5*(R1+R2)) < 0.1%)
    R = 0.5*(R1+R2)*w_s

    #get weighted tangential ellipticities
    cos2t = np.cos(2.0*theta)
    sin2t = np.sin(2.0*theta)
    et = (-e1*cos2t+e2*sin2t)*w_s
    ex = (e1*sin2t+e2*cos2t)*w_s

    ndots = binspace(RIN, ROUT, NBINS+1)
    dig = np.digitize((np.rad2deg(rads)/DEGxMPC), ndots)

    for n_i in range(NBINS):
        m_i = dig == n_i+1
        for b in range(4):
            zbin = catdata['bhat'] == b
            dsigma_t_num[n_i] += w_b[b]*np.sum(et[m_i & zbin])
            dsigma_x_num[n_i] += w_b[b]*np.sum(ex[m_i & zbin])
            response_sum[n_i] += w_b[b]*np.sum(R[m_i & zbin])
            n_sl_sum[n_i] += w_b[b]**2 * np.sum(R[m_i & zbin]**2)
            n_bin[n_i] += np.count_nonzero(m_i & zbin)

    return dsigma_t_num, dsigma_x_num, response_sum, n_sl_sum, n_bin

def stacking():

    init_globals()

    l = LENSES[ (LENSES['lambda']>38.0) & (LENSES['lambda']<=55) & (LENSES['redshift']>0.19) & (LENSES['redshift']<=0.27) ]
    print(f'nlenses = {len(l)}')

    dsigma_t_num = np.zeros((len(l), NBINS))
    dsigma_x_num = np.zeros((len(l), NBINS))
    response_sum = np.zeros((len(l), NBINS))
    n_sl_sum = np.zeros((len(l), NBINS))
    n_bin_sum = np.zeros((len(l), NBINS))

    for i, li in enumerate(l):
        dsigma_t_num[i,:], dsigma_x_num[i,:], response_sum[i,:], n_sl_sum[i,:], n_bin_sum[i,:] = partial_profile(
            [
                li['ra_gal'],
                li['dec_gal'],
                li['redshift'],
                li['wb_0'],
                li['wb_1'],
                li['wb_2'],
                li['wb_3']
            ]
        )

    response = np.sum(response_sum, axis=0)
    n_eff = np.sum(response_sum**2/n_sl_sum, axis=0)

    dsigma_t = np.sum(dsigma_t_num, axis=0)/(response*n_eff)
    dsigma_x = np.sum(dsigma_x_num, axis=0)/(response*n_eff)
    n_bin = np.sum(n_bin_sum, axis=0)

    r = binspace(RIN, ROUT, NBINS)

    np.savetxt('results/test-des_dsigma.dat', np.vstack([r, dsigma_t, dsigma_x, n_eff, n_bin]))

    if PLOT:
        fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(5,6))

        axes[0].scatter(r[dsigma_t > 0], dsigma_t[dsigma_t > 0], s=5, marker='o')
        axes[0].scatter(r[dsigma_t <= 0], np.abs(dsigma_t[dsigma_t <= 0]), s=5, marker='o', edgecolor='b', facecolor='none')
        axes[1].scatter(r[dsigma_x > 0], dsigma_x[dsigma_x > 0], s=5, marker='o', color='gray')
        axes[1].scatter(r[dsigma_x <= 0], np.abs(dsigma_x[dsigma_x <= 0]), s=5, marker='o', edgecolor='gray', facecolor='none')
        axes[0].loglog()
        #axes[1].loglog()

        # axes[0,1].scatter(r, N_bin, c='green', s=5)
        # axes[1,1].scatter(r, n_eff, c='green', s=5)
        # axes[0,1].loglog()
        # axes[1,1].loglog()

        fig.savefig('results/test-des_dsigma.png')

if __name__ == '__main__':

    print('Start'.center(15,'-'))
    t1 = time()
    stacking()
    print('End'.center(17,'-'))
    print(f'Took {time()-t1:.2f} s')

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

# def partial_profile_DeltaSigma(inp):
#     '''
#     Profile of proj. density contrast \Delta\Sigma(r).
#     No equation found in literature, intuitive derivation:

#     $$
#         \Delta \Sigma = \frac{
#             \sum_{l=0}^{N_L} \sum_{b} \sum_{i \in b} w_{bl} w_{s,il} e_{t,il}/R_{il} }{
#             \sum_{l=0}^{N_L} N_{s,l}
#         }
#     $$

#     where
#     $$
#         N_{s,l} = \frac{
#             (\sum_{b} \sum_{i \in b} w_{bl} w_{s,il} R_{il} )^2 }{
#             \sum_{b} \sum_{i \in b} (w_{bl} w_{s,il} R_{il})^2
#         }
#     $$

#     and $w_{bl}$ is the lensing efficiency of bin $b$ over lens $l$, $w_{s,il}$ is the weight of source $i$ of lens $l$ and $R_{il}$ is the response of source $i$ of lens $l$.
#     '''

#     dsigma_t_sum = np.zeros(NBINS)
#     dsigma_x_sum = np.zeros(NBINS)
#     response_sum = np.zeros(NBINS)
#     n_bin = np.zeros(NBINS)
#     n_sl_sum = np.zeros(NBINS)

#     ra0, dec0, z0, *w_b = inp

#     DEGxMPC = COSMO.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
#     psi = DEGxMPC*ROUT

#     # get masked data
#     mask, w_b = get_masked_data(psi, ra0, dec0, z0, w_b)
#     catdata = SOURCE[mask]

#     # calculate transformation to polar coords
#     rads, theta = eq2p2(
#         np.deg2rad(catdata['ra_gal']), np.deg2rad(catdata['dec_gal']),
#         np.deg2rad(ra0), np.deg2rad(dec0)
#     )

#     #get weights
#     w_s = catdata['weight']

#     e1 = -catdata['e_1']
#     e2 = catdata['e_2']
#     R1 = catdata['r11']
#     R2 = catdata['r22']
#     #se usa el promedio entre ambos xq son muy similares
#     #((R1-R2)/(0.5*(R1+R2)) < 0.1%)
#     R = 0.5*(R1+R2)

#     #get weighted tangential ellipticities
#     cos2t = np.cos(2.0*theta)
#     sin2t = np.sin(2.0*theta)
#     et = (-e1*cos2t+e2*sin2t)*w_s
#     ex = (e1*sin2t+e2*cos2t)*w_s

#     ndots = binspace(RIN, ROUT, NBINS+1)
#     dig = np.digitize((np.rad2deg(rads)/DEGxMPC), ndots)

#     for n_i in range(NBINS):
#         m_i = dig == n_i+1
#         for b in range(4):
#             zbin = catdata['bhat'] == b
#             dsigma_t_sum[n_i] += w_b[b]*np.sum(et[m_i & zbin])#/R[m_i & zbin])
#             dsigma_x_sum[n_i] += w_b[b]*np.sum(ex[m_i & zbin])#/R[m_i & zbin])
#             response_sum[n_i] += w_b[b]*np.sum(R[m_i & zbin])
#             n_sl_sum[n_i] += w_b[b]**2 * np.sum(R[m_i & zbin]**2)
#             n_bin[n_i] += np.count_nonzero(m_i & zbin)

#     return dsigma_t_sum, dsigma_x_sum, response_sum, n_sl_sum, n_bin

# def stack_dsigma():
#     savefile = 'test-des_dsigma'
#     l = LENSES[ (LENSES['lambda']>38.0) & (LENSES['lambda']<=55) & (LENSES['redshift']>0.19) & (LENSES['redshift']<=0.27) ]
#     print(f'nlenses = {len(l)}')

#     dsigma_t_sum = np.zeros((len(l), NBINS))
#     dsigma_x_sum = np.zeros((len(l), NBINS))
#     response_sum = np.zeros((len(l), NBINS))
#     n_sl_sum = np.zeros((len(l), NBINS))
#     n_bin_sum = np.zeros((len(l), NBINS))

#     for i, li in enumerate(l):
#         dsigma_t_sum[i,:], dsigma_x_sum[i,:], response_sum[i,:], n_sl_sum[i,:], n_bin_sum[i,:] = partial_profile_DeltaSigma(
#             [
#                 li['ra_gal'],
#                 li['dec_gal'],
#                 li['redshift'],
#                 li['wb_0'],
#                 li['wb_1'],
#                 li['wb_2'],
#                 li['wb_3']
#             ]
#         )

#     response = np.sum(response_sum, axis=0)
#     n_eff = np.sum(response_sum**2/n_sl_sum, axis=0)

#     dsigma_t = np.sum(dsigma_t_sum, axis=0)/n_eff
#     dsigma_x = np.sum(dsigma_x_sum, axis=0)/n_eff
#     n_bin = np.sum(n_bin_sum, axis=0)

#     r = binspace(RIN, ROUT, NBINS)

#     np.savetxt(savefile+'.dat', np.vstack([r, dsigma_t, dsigma_x, n_eff, n_bin, response]),
#                header='r | dsigma_t | dsigma_x | n_eff | n_bin | response')

#     if PLOT:
#         fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(5,6))

#         axes[0].scatter(r[dsigma_t > 0], dsigma_t[dsigma_t > 0], s=5, marker='o')
#         axes[0].scatter(r[dsigma_t <= 0], np.abs(dsigma_t[dsigma_t <= 0]), s=5, marker='o', edgecolor='b', facecolor='none')
#         axes[1].scatter(r[dsigma_x > 0], dsigma_x[dsigma_x > 0], s=5, marker='o', color='gray')
#         axes[1].scatter(r[dsigma_x <= 0], np.abs(dsigma_x[dsigma_x <= 0]), s=5, marker='o', edgecolor='gray', facecolor='none')
#         axes[0].loglog()
#         axes[1].loglog()

#         #axes[0,1].scatter(r, n_bin, c='green', s=5)
#         #axes[1,1].scatter(r, n_eff, c='green', s=5)
#         #axes[0,1].loglog()
#         #axes[1,1].loglog()

#         fig.savefig(savefile+'.png')

# #t2_matched = table2[order2[idx[mask]]]
