import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
#from astropy.constants import c, G, pc, M_sun
from astropy.table import Table
from astropy.io import fits
import healpy as hp
#from scipy.integrate import simpson
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import time, asctime
from tqdm import tqdm

from lensing.funcs import eq2p2, cov_matrix, get_jackknife_kmeans
#from io import *
#from nzsource import calculate_median, sigma_crit, lensing_efficiency, read_nzsource

# Fixed globals
COSMO = FlatLambdaCDM(H0=100, Om0=0.3)
NSIDE = 128
ZMED = np.array([0.285, 0.476, 0.743, 0.942]) # median redshift of source distribution
REDSHIFT = 'redshift' # name of the redshift col in source table
SOURCE = None
LENSES = None
ANGLES = None # temp file, should be anexed to LENSES cat
PIX_TO_IDX : dict = {}
binspace = None

# Input globals
NCORES = 16
NBINS = 15
RIN, ROUT = 0.1, 5.0 #Mpc/h
LMIN, LMAX = 38.0, 55.0
ZMIN, ZMAX = 0.19, 0.27
NJK = 40 # kmeans_radec allows up to a tenth of the # of lenses
BINNING = 'log'
PLOT = False
OVERWRITE = True
sample='test'
lensname='../cats/DESY3/desy3_redmapper_cluster-ws.fits'
anglename='../cats/DESY3/redmapper_orientation.dat'
sourcename='../cats/DESY3/desy3_metacal-unsheared-zbins_w-pix128_25314.fits'

def read_redmapper(filename='../cats/DESY3/desy3_redmapper_cluster-ws.fits'):
    lens = Table.read(filename, format='fits', memmap=True)
    lens.add_index('mem_match_id')
    return lens

def read_source(filename='../cats/DESY3/desy3_metacal-unsheared-zbins_w-pix128_25314.fits'):
    return Table.read(filename, format='fits', memmap=True)

def init_globals():
    global binspace
    global SOURCE, LENSES, ANGLES
    global PIX_TO_IDX

    if BINNING=='log':
        binspace = np.geomspace
    elif BINNING=='lin':
        binspace = np.linspace
    else:
        raise ValueError('BINNING must be "log" or "lin".')

    # reading catalogs
    SOURCE = read_source(sourcename) # metacal file
    LENSES = read_redmapper(lensname) # redmapper
    ANGLES = np.loadtxt(anglename)

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

    for i in range(4):
        if z0 > ZMED[i]:
            wb[i] = 0.0

    return idx_arrays, wb

def partial_profile(id_cl):
    '''
    Profile of reduced shear g_t(r) as in eq. 6 of Grandis et al. (2024)
    '''

    dsigma_t_num = np.zeros(NBINS)
    dsigma_x_num = np.zeros(NBINS)
    monopole_den = np.zeros(NBINS)
    quadpole_t_num = np.zeros(NBINS)
    quadpole_x_num = np.zeros(NBINS)
    quadpole_t_den = np.zeros(NBINS)
    quadpole_x_den = np.zeros(NBINS)
    n_bin = np.zeros(NBINS)
    n_sl_sum = np.zeros(NBINS)

    #ra0, dec0, z0, *w_b = inp
    #l = LENSES[LENSES['mem_match_id'] == id_cl]
    l = LENSES.loc(id_cl)
    ra0, dec0, z0, *w_b = l['ra_cl', 'dec_cl', 'redshift', 'wb_0', 'wb_1', 'wb_2', 'wb_3']
    _, e0, theta0 = ANGLES.T[ANGLES[0]==id_cl]

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
            dsigma_t_num[n_i] += np.sum(et[m_i & zbin])
            dsigma_x_num[n_i] += np.sum(ex[m_i & zbin])
            monopole_den[n_i] += w_b[b]*np.sum(res[m_i & zbin])

            quadpole_t_num[n_i] += np.sum(et[m_i & zbin])*np.cos(2.0*theta0)
            quadpole_x_num[n_i] += np.sum(ex[m_i & zbin])*np.sin(2.0*theta0)
            quadpole_t_den[n_i] += w_b[b]*np.sum(res[m_i & zbin])*(np.cos(2.0*theta0))**2
            quadpole_x_den[n_i] += w_b[b]*np.sum(res[m_i & zbin])*(np.sin(2.0*theta0))**2
            
            n_sl_sum[n_i] += w_b[b]**2 * np.sum(res[m_i & zbin]**2)
            n_bin[n_i] += np.count_nonzero(m_i & zbin)

    return dsigma_t_num, dsigma_x_num, monopole_den, quadpole_t_num, quadpole_t_num, quadpole_t_den, quadpole_x_den, n_sl_sum, n_bin

def stacking():
    
    l = LENSES[ (LENSES['lambda']>LMIN) & (LENSES['lambda']<=LMAX) & (LENSES['redshift']>ZMIN) & (LENSES['redshift']<=ZMAX) ]
    nlenses = len(l)
    print(f'{nlenses =}')
    localNJK = NJK
    if localNJK > (nlenses//10):
        localNJK = nlenses//10
    print(f'{localNJK =}')

    dsigma_t_num = np.zeros((localNJK+1, NBINS))
    dsigma_x_num = np.zeros((localNJK+1, NBINS))
    monopole_den = np.zeros((localNJK+1, NBINS))
    quadpole_t_num = np.zeros((localNJK+1, NBINS))
    quadpole_x_num = np.zeros((localNJK+1, NBINS))
    quadpole_t_den = np.zeros((localNJK+1, NBINS))
    quadpole_x_den = np.zeros((localNJK+1, NBINS))
    #n_sl_sum = np.zeros((NJK+1, NBINS))
    #n_bin_sum = np.zeros((NJK+1, NBINS))

    #for i, li in enumerate(l):
        # partial_profile([
        #         li['ra_gal','dec_gal','redshift','wb_0','wb_1','wb_2','wb_3']
        #     ])
    with Pool(processes=NCORES) as pool:
        results_map = list(
            tqdm(
                pool.imap(
                    partial_profile, 
                    l['mem_match_id'].data
                ), total=nlenses
            )
        )

    # === calculating stack
    
    # reduce
    gt, gx, mono_den, Gamma_t, Gamma_x, quad_t_den, quad_x_den, _, _ = map(
        lambda x: np.vstack(x),
        zip(*results_map)
    )

    dsigma_t_num[0,:] = gt.sum(axis=0)
    dsigma_x_num[0,:] = gx.sum(axis=0)
    monopole_den[0,:] = mono_den.sum(axis=0)
    quadpole_t_num[0,:] = Gamma_t.sum(axis=0)
    quadpole_x_num[0,:] = Gamma_x.sum(axis=0)
    quadpole_t_den[0,:] = quad_t_den.sum(axis=0)
    quadpole_x_den[0,:] = quad_x_den.sum(axis=0)
    #n_sl_sum[0,:] = nsl.sum(axis=0)
    #n_bin_sum[0,:] = nbin.sum(axis=0)

    # jackknife
    _, kidx = get_jackknife_kmeans(l['ra_gal'], l['dec_gal'], nlenses=nlenses, NJK=localNJK)
    kunq = np.unique(kidx)

    for j, k in enumerate(kunq):
        mask = (kidx!=k)

        dsigma_t_num[j+1,:] = gt[mask].sum(axis=0)
        dsigma_x_num[j+1,:] = gx[mask].sum(axis=0)
        monopole_den[j+1,:] = mono_den[mask].sum(axis=0)
        quadpole_t_num[0,:] = Gamma_t[mask].sum(axis=0)
        quadpole_x_num[0,:] = Gamma_x[mask].sum(axis=0)
        quadpole_t_den[0,:] = quad_t_den[mask].sum(axis=0)
        quadpole_x_den[0,:] = quad_x_den[mask].sum(axis=0)
        #n_sl_sum[j+1,:] = nsl[mask].sum(axis=0)
        #n_bin_sum[j+1,:] = nbin[mask].sum(axis=0)

    dsigma_t = dsigma_t_num/monopole_den
    dsigma_x = dsigma_x_num/monopole_den
    Gamma_quad_t = quadpole_t_num/quadpole_t_den
    Gamma_quad_x = quadpole_x_num/quadpole_x_den

    #n_eff = np.sum(response_sum**2/n_sl_sum, axis=0)
    #n_bin = np.sum(n_bin_sum, axis=0)
    #response = np.sum(response_sum, axis=0)

    # ==== Saving
    
    outputname = (f'results/lensing_desy3_{sample}_'
                  f'lambda{LMIN:02.0f}-{LMAX:02.0f}_'
                  f'z{100*ZMIN:03.0f}-{100*ZMAX:03.0f}_'
                  f'bin{BINNING}.fits')

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
        'res':binspace(RIN, ROUT, NBINS),
        'DSigma_t':dsigma_t[0], 
        'DSigma_x':dsigma_x[0],
        'Gamma_t':Gamma_quad_t[0],
        'Gamma_x':Gamma_quad_x[0],
    })

    cov_hdu = [
        fits.ImageHDU(cov_matrix(dsigma_t[1:,:]), name='cov_DSigma_t'),
        fits.ImageHDU(cov_matrix(dsigma_x[1:,:]), name='cov_DSigma_x'),
        fits.ImageHDU(cov_matrix(Gamma_quad_t[1:,:]), name='cov_Gamma_t'),
        fits.ImageHDU(cov_matrix(Gamma_quad_x[1:,:]), name='cov_Gamma_x'),
    ]

    jack_hdu = [
        fits.ImageHDU(dsigma_t[1:localNJK+1, :], name='jack_DSigma_t'),
        fits.ImageHDU(dsigma_x[1:localNJK+1, :], name='jack_DSigma_x'),
        fits.ImageHDU(Gamma_quad_t[1:localNJK+1, :], name='jack_Gamma_t'),
        fits.ImageHDU(Gamma_quad_x[1:localNJK+1, :], name='jack_Gamma_x'),
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
    print('Start'.center(15,'-'))

    t1 = time()
    init_globals()
    stacking()

    print('End'.center(17,'-'))
    print(f'Took {time()-t1:.2f} s')

if __name__ == '__main__':
    main()
