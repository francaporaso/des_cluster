from argparse import ArgumentParser
from multiprocessing import Pool
import emcee

from fitting.constants import *
from fitting.inference import *
from fitting.io import *
from fitting.models import *
from fitting.utilfuncs import *
from fitting.plotting import *

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
    for p in fix_params:
        init_guess[p] = None

    L = Likelihood(
        data=data,
        model=models_dict.get(model_name)(data.redshift),
        param_limits=param_limits,
        observable=observable,
        cov_mode=cov_mode
    )

    rng = np.random.default_rng(0)
    init_pos = np.zeros((NWALKERS, len(init_guess.keys())), dtype=object)
    for i, ig in enumerae(init_guess.values()):
        if ig is not None:
            init_pos[:,i] = rng.uniform(ig*(1.0-0.2), ig*(1.0+0.2), NWALKERS)
        else:
            init_pos[:,i] = np.full(NWALKERS, None)
    #init_pos = np.array([
    #    rng.uniform(ig*(1-0.2), ig*(1+0.2), NWALKERS) for ig in init_guess.values()
    #]).T #ordering of dict is asserted in python >3.7

    group_name = f'emcee/{model_name}/{cov_mode}'
    backend = emcee.backends.HDFBackend(save_filename, name=group_name)
    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            NWALKERS, L.nparams, L.log_probability, pool=pool, backend=backend
        )
        sampler.run_mcmc(init_pos, NIT, progress=True, store=True)

    return sampler

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataname', type=str, action='store', required=True)
    parser.add_argument('--savename', type=str, action='store', required=True)
    parser.add_argument('-c','--NCORES', type=int, default=32, action='store')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--NIT', type=int, default=1_000, action='store')
    parser.add_argument('--NWALKERS', type=int, default=64, action='store')
    parser.add_argument('--model', type=str, default='NFW', action='store')
    parser.add_argument('--cov', action='store_true')
    parser.add_argument('--fix', nargs='*')
    #parser.add_argument('--config', type=str, default='config.toml', action='store')
    #parser.add_argument('--use08', action='store_true')
    #parser.add_argument('--addnoise', action='store_true')
    #parser.add_argument('--nback', type=float, default=26.9, action='store')
    args = parser.parse_args()

    folder = 'results/'

    data_filename = folder + args.dataname #'lensing_desy3_test_lambda50-150_z020-040_binlog.fits'
    chain_filename = folder + args.savename #'fitting_desy3_misscentering_lambda50-150_z020-040.hdf5'
    model_name = args.model #'NFWMiss'
    observable = 'delta_sigma'
    if args.cov:
        cov_mode = 'full'
    else:
        cov_mode = 'diag'

    if args.fix is None:
        args.fix = []

    sampler = run_emcee(
        NCORES=args.NCORES,
        NIT=args.NIT,
        NWALKERS=args.NWALKERS,
        data_filename=data_filename,
        save_filename=chain_filename,
        model_name=model_name,
        observable=observable,
        fix_params=args.fix,
        cov_mode=cov_mode,
    )
    # TODO: que guarde los valores de mejor ajuste!

    plot_chains(sampler.get_chain())
    plt.show()

    plot_corner(sampler, discard=int(args.NIT*0.15));
    plt.show()
