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
        init_guess=np.array([1e14, 4.0])
        ):
    
    data = read_dataprofile_fits(name=data_filename)

    # param_limits = {'M200':(1e10, 1e16)}
    # if len(params_fit)==2:
    #     param_limits['c200']=(1.0, 10.0)
    param_limits = default_limits.get(model_name)
    for p in fix_params:
        param_limits.pop(p)

    L = Likelihood(
        data=data,
        model=models_dict.get(model_name)(data.redshift),
        param_limits=param_limits,
        observable=observable,
        cov_mode=cov_mode 
    )

    rng = np.random.default_rng(0)
    init_pos = np.array([
        rng.uniform(ig*(1-0.4), ig*(1+0.4), NWALKERS) for ig in init_guess
        #rng.uniform(init_guess[1]*(1-0.4), init_guess[1]*(1+0.4), NWALKERS),
    ]).T
    
    group_name = f'emcee/{model_name}/{cov_mode}'
    backend = emcee.backends.HDFBackend(save_filename, name=group_name)
    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            NWALKERS, L.nparams, L.log_probability, pool=pool, backend=backend
        )
        sampler.run_mcmc(init_pos, NIT, progress=True, store=True)

    return sampler

if __name__ == '__main__':
    
    NCORES = 32
    NIT = 1_000
    NWALKERS = 32

    data_filename = 'results/lensing_desy3_test_lambda38-55_z019-027_binlog.fits'
    chain_filename = 'results/fitting_desy3_misscentering_lambda38-55_z019-027.hdf5'
    model_name = 'NFWMiss'
    observable = 'delta_sigma'
    cov_mode = 'full'

    sampler = run_emcee(
        NCORES=NCORES,NIT=NIT,NWALKERS=NWALKERS,
        data_filename=data_filename,
        save_filename=chain_filename,
        model_name=model_name,
        observable=observable,
        fix_params = ['s_off'],
        cov_mode=cov_mode,
        init_guess=np.array([1e14, 4, 0.8])
    )
    # TODO: que guarde los valores de mejor ajuste!

    plot_chains(sampler.get_chain())
    plt.show()

    plot_corner(sampler, discard=int(NIT*0.35));
    plt.show()
