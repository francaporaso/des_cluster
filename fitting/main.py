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

models_dict = {
    'NFW':NFW,
}

default_limits = {
    'NFW':{
        'M200':(1e10, 1e16),
        # 'c200':(1.0, 10.0),
        'pcc':(0.01,1.0),
        # 's_off':(0.01,1.0)
    },
}

default_guess = {
    'NFW':{
        'M200':1e14,
        # 'c200':4.0,
        'pcc':0.8,
        # 's_off':0.4
    },
}


# ====
class Config:

    def __init__(self, configfile:str='fitting/config.toml'):

        config = toml.load(configfile)

        self.NCORES = config['run']['ncores']
        self.OVERWRITE = config['run']['overwriteh5']
        self.PLOT = config['run']['plot']

        self.NIT = config['mcmc']['nit']
        self.NWALKERS = config['mcmc']['nwalkers']
        self.covmode = config['mcmc']['covmode']
        self.model = config['mcmc']['model']
        self.secondhalo = config['mcmc']['secondhalo']
        self.observable = config['mcmc']['observable']
        self.FIXPARAM = config['mcmc']['fixparam']
        self.CHAINSAMPLE = config['mcmc']['chainsample']

        self.FOLDER = config['profiles']['folder']
        self.DATANAME = config['profiles']['dataname']
        self.SAMPLE = config['profiles']['sample']
        self.ZEDGES = self._edges_to_bins(config['profiles']['zedges'], 'ZEDGES')
        self.LBINS = self._edges_to_bins(config['profiles']['ledges'], 'LEDGES')
        self.PCEN = config['profiles']['pcen']
        self.MEMCUT = config['profiles']['memcut']
        self.WANG = config['profiles']['angweight']
        self.NBINS = config['profiles']['nbins']
        self.BINNING = config['profiles']['binning']

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
        secondhalo=False,
        fix_params = ['c200'],
        cov_mode='diag',
        ):

    if observable == 'delta_sigma':
        data = vread_dataprofile_fits(name=data_filename, profile='monopole')
    elif observable == 'quad_gamma_t':
        data = vread_dataprofile_fits(name=data_filename, profile='quadrupole')

    param_limits = default_limits.get(model_name)
    init_guess = default_guess.get(model_name)
    # for p in fix_params:
    #     init_guess[p] = None

    L = Likelihood(
        data=data,
        model=models_dict.get(model_name)(redshift=data.redshift, secondhalo=secondhalo),
        param_limits=param_limits,
        observable=observable,
        cov_mode=cov_mode
    )

    rng = np.random.default_rng(0)
    init_pos = np.zeros((nwalkers, len(init_guess.keys())), dtype=object)
    init_pos = np.array([
       rng.uniform(ig*(1-0.2), ig*(1+0.2), nwalkers) for ig in init_guess.values()
    ]).T #ordering of dict is asserted in python >3.7

    group_name = f'emcee/{model_name}/{'1h+2h' if secondhalo else '1h'}/{cov_mode}'
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
    parser.add_argument('--config', type=str, default='fitting/config.toml', action='store')
    args = parser.parse_args()

    cfg = Config(args.config)

    tot = len(cfg.ZBINS)*len(cfg.LBINS)
    print(f'>> Fitting {len(cfg.ZBINS)} redshift bins x {len(cfg.LBINS)} lambda bins = {tot} profiles')

    for i, ((zmin, zmax), (lmin, lmax)) in enumerate(product(cfg.ZBINS, cfg.LBINS), start=1):
        print(f'>> \n[{i}/{tot}]', flush=True)
        zstr = f'z{100*zmin:03.0f}-{100*zmax:03.0f}'
        lstr = f'lambda{lmin:02.0f}-{lmax:02.0f}'

        data_filename = self.FOLDER + f'{cfg.DATANAME}_{cfg.SAMPLE}_{zstr}_{lstr}_bin{cfg.NBINS}{cfg.BINNING}.fits'
        chain_filename = self.FOLDER + f'fitting_{cfg.CHAINSAMPLE}_{zstr}_{lstr}.hdf5'

        sampler = run_emcee(
            ncores=cfg.NCORES,
            nit=cfg.NIT,
            nwalkers=cfg.NWALKERS,
            data_filename=data_filename,
            save_filename=chain_filename,
            model_name=cfg.model,
            observable=cfg.observable,
            fix_params=cfg.FIXPARAM,
            cov_mode=cfg.covmode,
        )

        param_names = list(default_limits.get(cfg.model).keys())
        # not possible to fix params for now
        fitpar, errpar = get_fitted_params(
            sampler.get_chain(discard=int(cfg.NIT*0.15)), 
            param_names
        )

        with h5py.File(chain_filename, 'a') as f:
            group_path = f'fitedparams/{cfg.model}/{cfg.covmode}'

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
