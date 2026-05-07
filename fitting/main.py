from argparse import ArgumentParser
from multiprocessing import Pool
import emcee
import toml

from fitting.constants import *
from fitting.inference import *
from fitting.io import *
from fitting.models import *
from fitting.utilfuncs import *
from fitting.plotting import *

# ==== Fix globals

# === Input globals
config = toml.load('fitting/config.toml')
DATANAME = config['NAMES']['DATANME']
FOLDER = config['NAMES']['FOLDER']
ZMIN = config['LENSES']['ZMIN']
ZMAX = config['LENSES']['ZMAX']
LMIN = config['LENSES']['LMIN']
LMAX = config['LENSES']['LMAX']

NCORES = config['RUN']['NCORES']
NIT = config['RUN']['NIT']
NWALKERS = config['RUN']['NWALKERS']
COVMODE = config['RUN']['COVMODE']
MODEL = config['RUN']['MODEL']
FIXPARAM = config['RUN']['FIXPARAM']
OBSERVABLE = config['RUN']['OBSERVABLE']
PLOT = config['RUN']['PLOT']

# ===
def run_emcee(
        NCORES, NIT, NWALKERS,
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
    init_pos = np.zeros((NWALKERS, len(init_guess.keys())), dtype=object)
    # for i, ig in enumerate(init_guess.values()):
    #     if ig is not None:
    #         init_pos[:,i] = rng.uniform(ig*(1.0-0.2), ig*(1.0+0.2), NWALKERS)
    #     else:
    #         init_pos[:,i] = np.full(NWALKERS, None)
    init_pos = np.array([
       rng.uniform(ig*(1-0.2), ig*(1+0.2), NWALKERS) for ig in init_guess.values()
    ]).T #ordering of dict is asserted in python >3.7

    group_name = f'emcee/{model_name}/{cov_mode}'
    backend = emcee.backends.HDFBackend(save_filename, name=group_name)
    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            NWALKERS, L.nparams, L.log_probability, pool=pool, backend=backend
        )
        sampler.run_mcmc(init_pos, NIT, progress=True, store=True)

    return sampler

def main():

    parser = ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--config', type=str, default='config.toml', action='store')
    args = parser.parse_args()

    zbins = list(zip(ZMIN, ZMAX))
    lbins = list(zip(LMIN, LMAX))
    tot = len(zbins)*len(lbins)
    print('>> Fitting {len(zbins)} redshift bins x {len(lbins)} lambda bins = {tot} profiles')

    for i, ((zmin, zmax), (lmin, lmax)) in enumerate(product(zbins,lbins), start=1):
        print(f'>> \n[{i}/{total}]', flush=True)
        zstr = f'z{100*zmin:03.0f}-{100*zmax:03.0f}'
        lstr = f'lambda{lmin:02.0f}-{lmax:02.0f}'

        data_filename = FOLDER + f'{DATANAME}_{SAMPLE}_{zstr}_{lstr}_bin{NBINS}{BINNING}.fits'
        chain_filename = FOLDER + f'{CHAINNAME}_{SAMPLE}_{zstr}_{lstr}_model{MODEL}_nit{NIT}xnw{NWALKERS}.hdf5'

        sampler = run_emcee(
            NCORES=NCORES,
            NIT=NIT,
            NWALKERS=NWALKERS,
            data_filename=data_filename,
            save_filename=chain_filename,
            model_name=MODEL,
            observable=OBSERVABLE,
            fix_params=FIXPARAM,
            cov_mode=COVMODE,
        )
        # TODO: que guarde los valores de mejor ajuste!

        if PLOT:
            plot_chains(sampler.get_chain())
            plt.show()

            plot_corner(sampler, discard=int(args.NIT*0.15));
            plt.show()

if __name__ == '__main__':

    print('  Start  '.center('-',15))
    t1 = time.time()
    main()
    print('  End   '.center('-',15))
    print(f'>> Took {(t1-time.time())/60.0:2.0f} s')
