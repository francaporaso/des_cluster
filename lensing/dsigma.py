from argparse import ArgumentParser
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
from dataclasses import dataclass

from lensing.funcs import eq2p2, cov_matrix, get_jackknife_kmeans
#from io import *
#from nzsource import calculate_median, sigma_crit, lensing_efficiency, read_nzsource

# === Fixed globals
cfg = None
COSMO = FlatLambdaCDM(H0=100, Om0=0.3)
NSIDE = 128
ZMED = np.array([0.285, 0.476, 0.743, 0.942]) # median redshift of source distribution
# REDSHIFT = 'redshift' # name of the redshift col in source table ## not used for now
SOURCE = None
#LENSES = None
PIX_TO_IDX : dict = {}
binspace = None

# ==== Input globals
# read from config file
class Config:

    def __init__(self, configfile:str='lensing/config.toml'):

        config = toml.load(configfile)

        self.lensname=config['RUN']['LENSNAME']
        self.sourcename=config['RUN']['SOURCENAME']
        self.sample=config['RUN']['SAMPLE']

        self.NCORES = config['RUN']['NCORES']
        self.PLOT = config['RUN']['PLOT']
        self.OVERWRITE = config['RUN']['OVERWRITE']
        self.RIN, self.ROUT = config['PROFILE']['RIN'], config['PROFILE']['ROUT'] #Mpc/h
        self.NBINS = config['PROFILE']['NBINS']
        self.NJK = config['PROFILE']['NJK']
        self.BINNING = config['PROFILE']['BINNING']

        self.PCEN = config['LENSES']['PCEN']
        self.ZBINS = self._edges_to_bins(config['LENSES']['ZEDGES'], 'ZEDGES')
        self.LBINS = self._edges_to_bins(config['LENSES']['LEDGES'], 'LEDGES')

    def _edges_to_bins(self, edges, name):
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError(f'[LENSES] {name} must be a list with at least 2 values.')
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo >= hi:
                raise ValueError(f'[LENSES] {name} must be strictly increasing, got {lo} >= {hi}.')
        return list(zip(edges[:-1], edges[1:]))


def _footprint_mask(ra, dec, z, padding=1.5):
    '''
    Returns a boolean mask (True = keep) for an array of lens positions.
    A lens is rejected if any healpix pixel within its search disc
    (ROUT * padding in angular size) is missing from the source footprint.

    This removes lenses at the survey edge whose background source annulus
    would be incomplete, which otherwise causes empty-array crashes and
    biased profiles.

    Parameters
    ----------
    ra, dec  : array-like, degrees
    z        : array-like, lens redshifts (used to convert ROUT to angle)
    padding  : float, same multiplier used in get_masked_idx_fast (default 1.5)
    '''
    keep     = np.ones(len(ra), dtype=bool)
    occupied = set(PIX_TO_IDX.keys())

    for i, (ra0, dec0, z0) in enumerate(zip(ra, dec, z)):
        DEGxMPC     = COSMO.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
        search_rad  = np.deg2rad(DEGxMPC * cfg.ROUT * padding)
        pix_in_disc = hp.query_disc(
            NSIDE,
            vec=hp.ang2vec(ra0, dec0, lonlat=True),
            radius=search_rad
        )
        if not occupied.issuperset(pix_in_disc):
            keep[i] = False

    n_cut = (~keep).sum()
    if n_cut:
        print(f'>> Footprint cut: removed {n_cut}/{len(ra)} edge lenses '
              f'(ROUT={cfg.ROUT} Mpc/h, padding={padding}x)', flush=True)
    return keep

def read_redmapper(filename='../cats/DESY3/desy3_redmapper_cluster-ws.fits',
                   zmin=0.2, zmax=0.3, lmin=10, lmax=50, pcen=0.5):
    l = Table.read(filename, format='fits', memmap=True)
    mask = (
        (l['redshift'] >  zmin) & (l['redshift'] <= zmax) &
        (l['lambda']   >  lmin) & (l['lambda']   <= lmax) &
        (l['pcen']     >  pcen)
    )
    l = l[mask]
    footprint = _footprint_mask(l['ra_cl'], l['dec_cl'], l['redshift'], padding=1.0)
    return l[footprint]

def read_source(filename='../cats/DESY3/desy3_metacal-unsheared-zbins_w-pix128_25314.fits'):
    return Table.read(filename, format='fits', memmap=True)

def init_globals():
    global binspace
    global SOURCE#, LENSES
    global PIX_TO_IDX

    if cfg.BINNING=='log':
        binspace = np.geomspace
    elif cfg.BINNING=='lin':
        binspace = np.linspace
    else:
        raise ValueError('BINNING must be "log" or "lin".')

    # reading catalogs
    SOURCE = read_source(cfg.sourcename) # metacal file
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

    wb[0] = 0.0 # not using bin0 althogether
    for i in range(1,4):
        if z0 > ZMED[i]:
            wb[i] = 0.0

    return idx_arrays, wb

def partial_profile(inp):
    '''
    Profile of reduced shear g_t(r) as in eq. 6 of Grandis et al. (2024)
    '''

    # right ascention | declination | redshift | angle of semimayor axis | lensing eff. weights for redshift bins
    ra0, dec0, z0, phi0, *w_b = inp
    
    # monopole
    dsigma_t_num = np.zeros(cfg.NBINS)
    dsigma_x_num = np.zeros(cfg.NBINS)
    response_sum = np.zeros(cfg.NBINS)
    # quadrupole
    gamma_tcos_num = np.zeros(cfg.NBINS)
    gamma_xsin_num = np.zeros(cfg.NBINS)
    resp_tcos_sum = np.zeros(cfg.NBINS)
    resp_xsin_sum = np.zeros(cfg.NBINS)
    # n_eff
    n_bin = np.zeros(cfg.NBINS)
    sq_weight_sum = np.zeros(cfg.NBINS)
    weight_sum = np.zeros(cfg.NBINS)

    DEGxMPC = COSMO.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
    psi = DEGxMPC*cfg.ROUT

    # get masked data
    mask, w_b = get_masked_idx_fast(psi, ra0, dec0, z0, w_b)
    catdata = SOURCE[mask]
    # calculate transformation to polar coords
    rads, theta = eq2p2(
        np.deg2rad(catdata['ra_gal']), np.deg2rad(catdata['dec_gal']),
        np.deg2rad(ra0), np.deg2rad(dec0)
    )

    ### theta is measured east of north (ie from y towards x)
    ### to transform to usual convention (x towards y):
    theta = np.pi/2 + (2.0*np.pi - theta)

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

    # angle wrt the main axis
     phi = theta - phi0

    ndots = binspace(cfg.RIN, cfg.ROUT, cfg.NBINS+1)
    dig = np.digitize((np.rad2deg(rads)/DEGxMPC), ndots)

    for n_i in range(cfg.NBINS):
        m_i = dig == n_i+1
        for b in range(4):
            zbin = catdata['bhat'] == b

            cos_2phi = np.cos(2.0*phi)
            sin_2phi = np.sin(2.0*phi)

            # monopole
            dsigma_t_num[n_i] += w_b[b]**2 * np.sum(et[m_i & zbin])
            dsigma_x_num[n_i] += w_b[b]**2 * np.sum(ex[m_i & zbin])

            # quadrupole
            gamma_tcos_num[n_i] += w_b[b]**2 * np.sum(et[m_i & zbin] * cos_2phi)
            gamma_xsin_num[n_i] += w_b[b]**2 * np.sum(ex[m_i & zbin] * sin_2phi))

            # counts/denominators
            ## monopole
            response_sum[n_i] += w_b[b]**3 * np.sum(res[m_i & zbin])
            ## quadrupole
            resp_tcos_sum[n_i] +=  w_b[b]**3 * np.sum(res[m_i & zbin] * cos_2phi**2)
            resp_xsin_sum[n_i] +=  w_b[b]**3 * np.sum(res[m_i & zbin] * sin_2phi**2)
            ## n_eff
            weight_sum[n_i] += w_b[b] * np.sum(res[m_i & zbin])
            sq_weight_sum[n_i] += w_b[b]**2 * np.sum(res[m_i & zbin]**2)
            n_bin[n_i] += np.count_nonzero(m_i & zbin) if w_b[b] != 0.0 else 0.0

    return dsigma_t_num, dsigma_x_num, response_sum, gamma_tcos_num, gamma_xsin_num, resp_tcos_sum, resp_xsin_sum, weight_sum, sq_weight_sum, n_bin

def stacking(zmin, zmax, lmin, lmax, pcen=0.5):

    l = read_redmapper(cfg.lensname, zmin, zmax, lmin, lmax, pcen) # redmapper

    nlenses = len(l)
    print(f'>> Z = [{zmin}, {zmax})')
    print(f'>> LAMBDA = [{lmin}, {lmax})')
    print(f'>> NLENSES = {nlenses}')
    localNJK = cfg.NJK
    if localNJK < int(cfg.NBINS**(3/2)):
        localNJK = int(cfg.NBINS**(3/2))
    print(f'>> Using NJK = {localNJK}')

    dsigma_t_num = np.zeros((localNJK+1, cfg.NBINS))
    dsigma_x_num = np.zeros((localNJK+1, cfg.NBINS))
    response_sum = np.zeros((localNJK+1, cfg.NBINS))

    gamma_tcos_num = np.zeros((localNJK+1, cfg.NBINS))
    gamma_xsin_num = np.zeros((localNJK+1, cfg.NBINS))
    resp_tcos_sum = np.zeros((localNJK+1, cfg.NBINS))
    resp_xsin_sum = np.zeros((localNJK+1, cfg.NBINS))

    weight_sl    = np.zeros((localNJK+1, cfg.NBINS))
    sq_weight_sl = np.zeros((localNJK+1, cfg.NBINS))
    n_bin        = np.zeros((localNJK+1, cfg.NBINS))

    with Pool(processes=cfg.NCORES) as pool:
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
    gt, gx, res, gammat, gammax, res_cos, res_sin, w_sl, sqw_sl, nbin = map(
        lambda x: np.vstack(x),
        zip(*results_map)
    )

    # calculate sum over lenses
    # monopole
    dsigma_t_num[0,:] = gt.sum(axis=0)
    dsigma_x_num[0,:] = gx.sum(axis=0)
    response_sum[0,:] = res.sum(axis=0)
    # quadrupole
    gamma_tcos_num[0,:] = gammat.sum(axis=0)
    gamma_xsin_num[0,:] = gammax.sum(axis=0)
    resp_tcos_sum[0,:] = res_cos.sum(axis=0)
    resp_xsin_sum[0,:] = res_sin.sum(axis=0)
    # n_eff
    weight_sl[0,:] = np.sum(w_sl, axis=0)**2
    sq_weight_sl[0,:] = sqw_sl.sum(axis=0)
    n_bin[0,:] = nbin.sum(axis=0)

    # jackknife
    jidx = np.arange(0, len(SOURCE)-1, len(SOURCE)//100_000, dtype=int)
    kidx = get_jackknife_kmeans(
        ra_sample=SOURCE['ra_gal'][jidx],
        dec_sample=SOURCE['dec_gal'][jidx],
        ra_cl=l['ra_cl'],
        dec_cl=l['dec_cl'],
        nlenses=nlenses,
        NJK=localNJK
    )
    for j, k in enumerate(range(localNJK)):
        mask = (kidx!=k)
        # monopole
        dsigma_t_num[j+1,:] = gt[mask].sum(axis=0)
        dsigma_x_num[j+1,:] = gx[mask].sum(axis=0)
        response_sum[j+1,:] = res[mask].sum(axis=0)
        # quadrupole
        gamma_tcos_num[0,:] = gammat[mask].sum(axis=0)
        gamma_xsin_num[0,:] = gammax[mask].sum(axis=0)
        resp_tcos_sum[0,:] = res_cos[mask].sum(axis=0)
        resp_xsin_sum[0,:] = res_sin[mask].sum(axis=0)     
        # n_eff
        weight_sl[j+1,:] = w_sl[mask].sum(axis=0)**2
        sq_weight_sl[j+1,:] = sqw_sl[mask].sum(axis=0)
        n_bin[j+1,:] = nbin[mask].sum(axis=0)

    n_eff = weight_sl/sq_weight_sl

    # cluster contaminants correction
    f_cl = contaminants_fraction(n_eff)

    # profiles
    ## monopole
    dsigma_t = (1/(1-f_cl))*dsigma_t_num/response_sum
    dsigma_x = (1/(1-f_cl))*dsigma_x_num/response_sum

    ## quadrupole
    gamma_tcos = (1/(1-f_cl))*gamma_tcos_num/resp_tcos_sum
    gamma_xsin = (1/(1-f_cl))*gamma_xsin_num/resp_xsin_sum

    # ==== Saving
    outputname = (f'results/lensing_desy3_quad_{cfg.sample}_'
                  f'z{100*zmin:03.0f}-{100*zmax:03.0f}_'
                  f'lambda{lmin:02.0f}-{lmax:02.0f}_'
                  f'bin{cfg.NBINS}{cfg.BINNING}.fits')

    richness_quartile = np.percentile(l['lambda'], [16,50,86])

    head=fits.Header()
    head.update({
        'nlenses':nlenses,
        'lenscat':cfg.lensname,
        'sourcat':cfg.sourcename,
        'l_min':lmin,
        'l_max':lmax,
        'l_mean':np.mean(l['lambda']),
        'l_med':richness_quartile[1],
        'l_low':richness_quartile[0],
        'l_hig':richness_quartile[2],
        'z_min':zmin,
        'z_max':zmax,
        'z_mean':np.mean(l['redshift']),
        'RIN':cfg.RIN,
        'ROUT':cfg.ROUT,
        'NBINS':cfg.NBINS,
        'NJK':localNJK,
        'binning':cfg.BINNING,
        'HISTORY':f'{asctime()}',
    })

    table = Table({
        'R':binspace(cfg.RIN, cfg.ROUT, cfg.NBINS),
        'DSigma_t':dsigma_t[0],
        'DSigma_x':dsigma_x[0],
        'Gamma_tcos':gamma_tcos[0],
        'Gamma_xsin':gamma_xsin[0],
        'N_eff':n_eff[0],
        'N_raw':n_bin[0]
    })

    cov_hdu = [
        fits.ImageHDU(cov_matrix(dsigma_t[1:,:]), name='cov_DSigma_t'),
        fits.ImageHDU(cov_matrix(dsigma_x[1:,:]), name='cov_DSigma_x'),
        fits.ImageHDU(cov_matrix(gamma_tcos[1:,:]), name='cov_Gamma_tcos'),
        fits.ImageHDU(cov_matrix(gamma_xsin[1:,:]), name='cov_Gamma_xsin'),
        fits.ImageHDU(cov_matrix(n_eff[1:,:]), name='cov_N_eff'),
        fits.ImageHDU(cov_matrix(n_bin[1:,:]), name='cov_N_raw'),
    ]

    jack_hdu = [
        fits.ImageHDU(dsigma_t[1:localNJK+1, :], name='jack_DSigma_t'),
        fits.ImageHDU(dsigma_x[1:localNJK+1, :], name='jack_DSigma_x'),
        fits.ImageHDU(gamma_tcos[1:localNJK+1, :], name='jack_Gamma_tcos'),
        fits.ImageHDU(gamma_xsin[1:localNJK+1, :], name='jack_Gamma_xsin'),
        fits.ImageHDU(n_eff[1:localNJK+1, :], name='jack_N_eff'),
        fits.ImageHDU(n_bin[1:localNJK+1, :], name='jack_N_raw'),
    ]

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=head),
        fits.BinTableHDU(table, name='profiles'),
        *cov_hdu,
        *jack_hdu
    ])

    hdul.writeto(outputname, overwrite=cfg.OVERWRITE)
    print(f' File saved in: {outputname}', flush=True)

    if cfg.PLOT:
        plot_profile(
            r=binspace(cfg.RIN, cfg.ROUT, cfg.NBINS),
            dst=dsigma_t[0],
            e_dst=np.sqrt(np.diag(cov_matrix(dsigma_t[1:,:]))),
            dsx=dsigma_x[0],
            e_dsx=np.sqrt(np.diag(cov_matrix(dsigma_t[1:,:])))
        )
        plot_profile(
            r=binspace(cfg.RIN, cfg.ROUT, cfg.NBINS),
            dst=gamma_tcos[0],
            e_dst=np.sqrt(np.diag(cov_matrix(gamma_tcos[1:,:]))),
            dsx=gamma_xsin[0],
            e_dsx=np.sqrt(np.diag(cov_matrix(gamma_xsin[1:,:])))
        )
 

    return 0

def contaminants_fraction(n_eff):
    # cluster contaminants correction
    area = np.pi*np.diff(binspace(cfg.RIN, cfg.ROUT, cfg.NBINS+1))**2
    den_n = n_eff/area
    f_cl = 1.0 - den_n[-1]/den_n

    return f_cl

def plot_profile(r, dst, e_dst, dsx, e_dsx):

    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(5,6))

    axes[0].errorbar(r, dst, e_dst, fmt='ok', capsize=2)
    axes[1].errorbar(r, dsx, e_dsx, fmt='ok', capsize=2)
    axes[0].loglog()
    #axes[1].loglog()
    plt.show()

def main():
    global cfg

    print(' Start '.center(15,'-'))
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='lensing/config.toml', action='store')
    args = parser.parse_args()

    t1 = time()

    cfg = Config(args.config)
    init_globals()

    total = len(cfg.ZBINS) * len(cfg.LBINS)
    print(f'>> Running {len(cfg.ZBINS)} redshift bin(s) x {len(cfg.LBINS)} richness bin(s) = {total} combination(s)')

    print('>> RIN '+f'{"= ": >14}{cfg.RIN:.2f}')
    print('>> ROUT '+f'{"= ": >14}{cfg.ROUT:.2f}')
    print('>> NBINS '+f'{"= ": >17}{cfg.NBINS:<2d}')

    for i, ((zmin, zmax), (lmin, lmax)) in enumerate(product(cfg.ZBINS, cfg.LBINS), start=1):
        print(f'  \n[{i}/{total}]  ', flush=True)
        check = stacking(zmin, zmax, lmin, lmax, cfg.PCEN)
        assert check == 0, '>>> Something went wrong. <<<'

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

