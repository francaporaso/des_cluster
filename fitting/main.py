from argparse import ArgumentParser
from multiprocessing import Pool
import emcee
import toml
import time
import h5py
from itertools import product

from fitting.constants import *
from fitting.inference import *
from fitting.io import *
from fitting.models import *
from fitting.utilfuncs import *
from fitting.plotting import *

# ==== Fix globals
cfg = None

# ====
class Config:

    def __init__(self, configfile:str='fitting/config.toml'):

        config = toml.load(configfile)

        self.DATANAME = config['NAMES']['DATANAME']
        self.FOLDER = config['NAMES']['FOLDER']
        self.SAMPLE = config['NAMES']['SAMPLE']
        self.CHAINNAME = config['NAMES']['CHAINNAME']
        self.ZBINS = self._edges_to_bins(config['LENSES']['ZEDGES'], 'ZEDGES')
        self.LBINS = self._edges_to_bins(config['LENSES']['LEDGES'], 'LEDGES')
        self.NBINS = config['LENSES']['NBINS']
        self.BINNING = config['LENSES']['BINNING']

        self.NCORES = config['RUN']['NCORES']
        self.NIT = config['RUN']['NIT']
        self.NWALKERS = config['RUN']['NWALKERS']
        self.COVMODE = config['RUN']['COVMODE']
        self.MODEL = config['RUN']['MODEL']
        self.FIXPARAM = config['RUN']['FIXPARAM']
        self.OBSERVABLE = config['RUN']['OBSERVABLE']
        self.PLOT = config['RUN']['PLOT']

    def _edges_to_bins(self, edges, name):
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError(f'[LENSES] {name} must be a list with at least 2 values.')
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo >= hi:
                raise ValueError(f'[LENSES] {name} must be strictly increasing, got {lo} >= {hi}.')
        return list(zip(edges[:-1], edges[1:]))

# ===
def run_emcee(
        ncores, nit, nwalkers,
        data_filename, save_filename,
        model_name='NFW',
        observable='delta_sigma',
        fix_params = ['c200'],
        cov_mode='diag',
        ):

    data = read_dataprofile_fits(name=data_filename)

    param_limits = default_limits.get(model_name)
    init_guess = default_guess.get(model_name)
    # for p in fix_params:
    #     init_guess[p] = None

    L = Likelihood(
        data=data,
        model=models_dict.get(model_name)(data.redshift),
        param_limits=param_limits,
        observable=observable,
        cov_mode=cov_mode
    )

    rng = np.random.default_rng(0)
    init_pos = np.zeros((nwalkers, len(init_guess.keys())), dtype=object)
    init_pos = np.array([
       rng.uniform(ig*(1-0.2), ig*(1+0.2), nwalkers) for ig in init_guess.values()
    ]).T #ordering of dict is asserted in python >3.7

    group_name = f'emcee/{model_name}/{cov_mode}'
    backend = emcee.backends.HDFBackend(save_filename, name=group_name)
    with Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, L.nparams, L.log_probability, pool=pool, backend=backend
        )
        sampler.run_mcmc(init_pos, nit, progress=True, store=True)

    return sampler

def main():
    global cfg

    parser = ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--config', type=str, default='fitting/config.toml', action='store')
    args = parser.parse_args()

    cfg = Config(args.config)

    tot = len(cfg.ZBINS)*len(cfg.LBINS)
    print(f'>> Fitting {len(cfg.ZBINS)} redshift bins x {len(cfg.LBINS)} lambda bins = {tot} profiles')

    for i, ((zmin, zmax), (lmin, lmax)) in enumerate(product(cfg.ZBINS, cfg.LBINS), start=1):
        print(f'>> \n[{i}/{tot}]', flush=True)
        zstr = f'z{100*zmin:03.0f}-{100*zmax:03.0f}'
        lstr = f'lambda{lmin:02.0f}-{lmax:02.0f}'

        data_filename = self.FOLDER + f'{cfg.DATANAME}_{zstr}_{lstr}_bin{cfg.NBINS}{cfg.BINNING}.fits'
        chain_filename = self.FOLDER + f'{cfg.CHAINNAME}_{cfg.SAMPLE}_{zstr}_{lstr}.hdf5'

        sampler = run_emcee(
            ncores=cfg.NCORES,
            nit=cfg.NIT,
            nwalkers=cfg.NWALKERS,
            data_filename=data_filename,
            save_filename=chain_filename,
            model_name=cfg.MODEL,
            observable=cfg.OBSERVABLE,
            fix_params=cfg.FIXPARAM,
            cov_mode=cfg.COVMODE,
        )

        param_names = list(default_limits.get(cfg.MODEL).keys())
        # not possible to fix params for now
        fitpar, errpar = get_fitted_params(sampler.get_chain(discard=int(cfg.NIT*0.15)), param_names)

        with h5py.File(chain_filename, 'a') as f:
            group_path = f'fitedparams/{cfg.MODEL}/{cfg.COVMODE}'

            # Overwrite if exists
            if group_path in f:
                del f[group_path]

            grp = f.create_group(group_path)

            for pname in param_names:
                pgrp = grp.create_group(pname)
                pgrp.create_dataset('median', data=fitpar[pname])
                pgrp.create_dataset('errs', data=np.array(errpar[pname]))

        if PLOT:
            plot_chains(sampler.get_chain())
            plt.show()

            plot_corner(sampler, discard=int(cfg.NIT*0.15));
            plt.show()

if __name__ == '__main__':

    print('  Start  '.center(15, '-'))
    t1 = time.time()
    main()
    print('  End   '.center(15, '-'))
    print(f'>> Took {(t1-time.time())/60.0:2.0f} s')
