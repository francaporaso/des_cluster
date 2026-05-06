#from argparse import ArgumentParser
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.io import fits
import healpy as hp
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import time, asctime
from tqdm import tqdm
import toml
from itertools import product

from lensing.funcs import eq2p2, cov_matrix, get_jackknife_kmeans
#from io import *
#from nzsource import calculate_median, sigma_crit, lensing_efficiency, read_nzsource

# === Fixed globals
COSMO = FlatLambdaCDM(H0=100, Om0=0.3)
NSIDE = 128
ZMED = np.array([0.285, 0.476, 0.743, 0.942]) # median redshift of source distribution
REDSHIFT = 'redshift' # name of the redshift col in source table
SOURCE = None
#LENSES = None
PIX_TO_IDX : dict = {}
binspace = None

# ==== Input globals
# read from config file
config = toml.load('lensing/config.toml')
lensname=config['RUN']['LENSNAME']
sourcename=config['RUN']['SOURCENAME']
sample=config['RUN']['SAMPLE']

NCORES = config['RUN']['NCORES']
PLOT = config['RUN']['PLOT']
OVERWRITE = config['RUN']['OVERWRITE']
RIN, ROUT = config['PROFILE']['RIN'], config['PROFILE']['ROUT'] #Mpc/h
NBINS = config['PROFILE']['NBINS']
NJK = config['PROFILE']['NJK']
BINNING = config['PROFILE']['BINNING']
#LMIN, LMAX = config['LENSES']['LMIN'], config['LENSES']['LMAX']
#ZMIN, ZMAX = config['LENSES']['ZMIN'], config['LENSES']['ZMAX']

# Lens bin lists — each pair (ZMIN[i], ZMAX[i]) and (LMIN[j], LMAX[j])
# is read as a list; scalars are wrapped so the rest of the code is uniform.
def _to_list(val):
    return val if isinstance(val, list) else [val]

ZMIN_LIST = _to_list(config['LENSES']['ZMIN'])
ZMAX_LIST = _to_list(config['LENSES']['ZMAX'])
LMIN_LIST = _to_list(config['LENSES']['LMIN'])
LMAX_LIST = _to_list(config['LENSES']['LMAX'])


def read_redmapper(filename='../cats/DESY3/desy3_redmapper_cluster-ws.fits',
                   ZMIN=0.2, ZMAX=0.3, LMIN=10, LMAX=50, PCEN=0.5):
    l = Table.read(filename, format='fits', memmap=True)
    mask = (l['redshift'] > ZMIN)&(l['redshift'] <= ZMAX)&(l['lambda'] > LMIN)&(l['lambda'] <= LMAX)&(l['pcen']>PCEN)
    return l[mask]

def read_source(filename='../cats/DESY3/desy3_metacal-unsheared-zbins_w-pix128_25314.fits'):
    return Table.read(filename, format='fits', memmap=True)

def init_globals():
    global binspace
    global SOURCE#, LENSES
    global PIX_TO_IDX

    if BINNING=='log':
        binspace = np.geomspace
    elif BINNING=='lin':
        binspace = np.linspace
    else:
        raise ValueError('BINNING must be "log" or "lin".')

    # reading catalogs
    SOURCE = read_source(sourcename) # metacal file
    #LENSES = read_redmapper(lensname) # redmapper

    # making a dict of healpix idx for fast query
    upix, split_idx = np.unique(SOURCE['pix'], return_index=True)
    split_idx = np.append(split_idx, len(SOURCE))
    for i, pix in enumerate(upix):
        PIX_TO_IDX[int(pix)] = np.arange(split_idx[i], split_idx[i+1])

def get_masked_idx_fast(psi, ra0, dec0, z0, wb):
    '''
    objects are selected by pixel on a disc of rad=psi+pad where pad = 0.1*psi
    uses prebuilt _PIX_TO_INDEX dict
    returns the indices of _S where to select
    '''

    pix_idx = hp.query_disc(
        NSIDE,
        vec=hp.ang2vec(ra0, dec0, lonlat=True),
        radius=np.deg2rad(psi*1.5)
    )

    idx_arrays = np.concatenate([
        PIX_TO_IDX[p]
        for p in pix_idx
        if p in PIX_TO_IDX
    ])

    for i in range(4):
        if z0 > ZMED[i]:
            wb[i] = 0.0

    return idx_arrays, wb

def partial_profile(inp):
    '''
    Profile of reduced shear g_t(r) as in eq. 6 of Grandis et al. (2024)
    '''

    ra0, dec0, z0, *w_b = inp

    dsigma_t_num = np.zeros(NBINS)
    dsigma_x_num = np.zeros(NBINS)
    response_sum = np.zeros(NBINS)
    n_bin = np.zeros(NBINS)
    sq_weight_sum = np.zeros(NBINS)
    weight_sum = np.zeros(NBINS)

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
    r1 = catdata['r11']
    r2 = catdata['r22']
    # uses the mean of the trace bc they are very similar
    #((r1-r2)/(0.5*(r1+r2)) < 0.1%)
    res = 0.5*(r1+r2)*w_s

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

            dsigma_t_num[n_i] += w_b[b]**2 * np.sum(et[m_i & zbin])
            dsigma_x_num[n_i] += w_b[b]**2 * np.sum(ex[m_i & zbin])

            response_sum[n_i] += w_b[b]**3 * np.sum(res[m_i & zbin])

            weight_sum[n_i] += w_b[b] * np.sum(res[m_i & zbin])
            sq_weight_sum[n_i] += w_b[b]**2 * np.sum(res[m_i & zbin]**2)
            n_bin[n_i] += np.count_nonzero(m_i & zbin) if w_b[b] != 0.0 else 0.0

    return dsigma_t_num, dsigma_x_num, response_sum, weight_sum, sq_weight_sum, n_bin

def stacking(zmin, zmax, lmin, lmax, pcen=0.5):

    l = read_redmapper(lensname, zmin, zmax, lmin, lmax, pcen) # redmapper

    nlenses = len(l)
    print(f'{nlenses =}')
    localNJK = NJK
    if localNJK > int(NBINS**(3/2)):
        localNJK = int(NBINS**(3/2))
    print(f'>> Using NJK ={localNJK}')

    dsigma_t_num = np.zeros((localNJK+1, NBINS))
    dsigma_x_num = np.zeros((localNJK+1, NBINS))
    response_sum = np.zeros((localNJK+1, NBINS))

    weight_sl    = np.zeros((localNJK+1, NBINS))
    sq_weight_sl = np.zeros((localNJK+1, NBINS))
    n_bin        = np.zeros((localNJK+1, NBINS))

    with Pool(processes=NCORES) as pool:
        results_map = list(
            tqdm(
                pool.imap(
                    partial_profile,
                    l['ra_cl','dec_cl','redshift','wb_0','wb_1','wb_2','wb_3'].as_array()
                ), total=nlenses
            )
        )

    # === calculating stack

    # reduce
    gt, gx, res, w_sl, sqw_sl, nbin = map(
        lambda x: np.vstack(x),
        zip(*results_map)
    )

    #calculate sum over lenses
    dsigma_t_num[0,:] = gt.sum(axis=0)
    dsigma_x_num[0,:] = gx.sum(axis=0)
    response_sum[0,:] = res.sum(axis=0)
    weight_sl[0,:] = np.sum(w_sl, axis=0)**2
    sq_weight_sl[0,:] = sqw_sl.sum(axis=0)
    n_bin[0,:] = nbin.sum(axis=0)

    # jackknife
    _, kidx = get_jackknife_kmeans(SOURCE['ra_gal'], SOURCE['dec_gal'], nlenses=nlenses, NJK=localNJK)
    kunq = np.unique(kidx)

    for j, k in enumerate(kunq):
        mask = (kidx!=k)

        dsigma_t_num[j+1,:] = gt[mask].sum(axis=0)
        dsigma_x_num[j+1,:] = gx[mask].sum(axis=0)
        response_sum[j+1,:] = res[mask].sum(axis=0)
        weight_sl[j+1,:] = w_sl[mask].sum(axis=0)**2
        sq_weight_sl[j+1,:] = sqw_sl[mask].sum(axis=0)
        n_bin[j+1,:] = nbin[mask].sum(axis=0)

    n_eff = weight_sl/sq_weight_sl
    #response = np.sum(response_sum, axis=0)

    # cluster contaminants correction
    area = np.pi*np.diff(binspace(RIN, ROUT, NBINS+1))**2
    den_n = n_eff/area
    f_cl = 1.0 - den_n[-1]/den_n

    # profiles
    dsigma_t = (1/(1-f_cl))*dsigma_t_num/response_sum
    dsigma_x = (1/(1-f_cl))*dsigma_x_num/response_sum

    # ==== Saving
    outputname = (f'results/lensing_desy3_{sample}_'
                  f'z{100*zmin:03.0f}-{100*zmax:03.0f}_'
                  f'lambda{lmin:02.0f}-{lmax:02.0f}_'
                  f'bin{NBINS}{BINNING}.fits')

    head=fits.Header()
    head.update({
        'nlenses':nlenses,
        'lenscat':lensname,
        'sourcat':sourcename,
        'lam_min':LMIN,
        'lam_max':LMAX,
        'lam_mean':np.mean(l['lambda']),
        'z_min':ZMIN,
        'z_max':ZMAX,
        'z_mean':np.mean(l['redshift']),
        'RIN':RIN,
        'ROUT':ROUT,
        'NBINS':NBINS,
        'NJK':localNJK,
        'binning':BINNING,
        'HISTORY':f'{asctime()}',
    })

    table = Table({
        'R':binspace(RIN, ROUT, NBINS),
        'DSigma_t':dsigma_t[0],
        'DSigma_x':dsigma_x[0],
        'N_eff':n_eff[0],
        'N_raw':n_bin[0]
    })

    cov_hdu = [
        fits.ImageHDU(cov_matrix(dsigma_t[1:,:]), name='cov_DSigma_t'),
        fits.ImageHDU(cov_matrix(dsigma_x[1:,:]), name='cov_DSigma_x'),
        fits.ImageHDU(cov_matrix(n_eff[1:,:]), name='cov_N_eff'),
        fits.ImageHDU(cov_matrix(n_bin[1:,:]), name='cov_N_raw'),
    ]

    jack_hdu = [
        fits.ImageHDU(dsigma_t[1:localNJK+1, :], name='jack_DSigma_t'),
        fits.ImageHDU(dsigma_x[1:localNJK+1, :], name='jack_DSigma_x'),
        fits.ImageHDU(n_eff[1:localNJK+1, :], name='jack_N_eff'),
        fits.ImageHDU(n_bin[1:localNJK+1, :], name='jack_N_raw'),
    ]

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=head),
        fits.BinTableHDU(table, name='profiles'),
        *cov_hdu,
        *jack_hdu
    ])

    hdul.writeto(outputname, overwrite=OVERWRITE)
    print(f' File saved in: {outputname}', flush=True)

    if PLOT:
        plot_profile(binspace(RIN, ROUT, NBINS), dsigma_t, dsigma_x)

def plot_profile(r, dsigma_t, dsigma_x):

    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(5,6))

    axes[0].scatter(r[dsigma_t > 0], dsigma_t[dsigma_t > 0], s=5, marker='o')
    axes[0].scatter(r[dsigma_t <= 0], np.abs(dsigma_t[dsigma_t <= 0]), s=5, marker='o', edgecolor='b', facecolor='none')
    axes[1].scatter(r[dsigma_x > 0], dsigma_x[dsigma_x > 0], s=5, marker='o', color='gray')
    axes[1].scatter(r[dsigma_x <= 0], np.abs(dsigma_x[dsigma_x <= 0]), s=5, marker='o', edgecolor='gray', facecolor='none')
    axes[0].loglog()
    plt.show()
    #axes[1].loglog()

    # axes[0,1].scatter(r, N_bin, c='green', s=5)
    # axes[1,1].scatter(r, n_eff, c='green', s=5)
    # axes[0,1].loglog()
    # axes[1,1].loglog()

    #fig.savefig('results/test-des_dsigma.png')

def main():
    print(' Start '.center(15,'-'))

    t1 = time()

    init_globals()

    # Build the list of (zmin, zmax) pairs and (lmin, lmax) pairs
    zbins = list(zip(ZMIN_LIST, ZMAX_LIST))
    lbins = list(zip(LMIN_LIST, LMAX_LIST))

    total = len(zbins) * len(lbins)
    print(f'>> Running {len(zbins)} redshift bin(s) x {len(lbins)} richness bin(s) = {total} combination(s)')

    for i, ((zmin, zmax), (lmin, lmax)) in enumerate(product(zbins, lbins), start=1):
        print(f'  \n[{i}/{total}]  ', flush=True)
        stacking(zmin, zmax, lmin, lmax)


    print(' End '.center(17,'-'))
    print(f'>>> Took {time()-t1:.2f} s <<<')

if __name__ == '__main__':
    main()


#def get_masked_square(psi, ra0, dec0, z0, wb):
#    '''
#    deprecated
#    square mask using binary comparisons.
#    too slow for big queries. use get_masked_idx_fast instead.
#    '''
#    mask_sky = (SOURCE['ra_gal'] < (ra0+psi))&(SOURCE['ra_gal'] > (ra0-psi))&(SOURCE['dec_gal'] < (dec0+psi))&(SOURCE['dec_gal'] > (dec0-psi))
#    #drop the first redshift bin altogether
#    wb[0] = 0.0
#    for i in range(1,4):
#        if z0 > ZMED[i]:
#            wb[i] = 0.0
#    return mask_sky, wb

