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
        cov_mode='diag',
        init_guess=[1e14, 3.0]
        ):
    
    data = read_dataprofile_fits(name=data_filename)

    L = Likelihood(
        data=data,
        model=models_dict.get(model_name)(data.redshift),
        param_limits=default_limits.get(model_name),
        observable=observable,
        cov_mode=cov_mode 
    )

    rng = np.random.default_rng(0)
    init_pos = np.array([
        rng.uniform(init_guess[0]*(1-0.15), init_guess[0]*(1+0.15), NWALKERS),
        rng.uniform(init_guess[1]*(1-0.15), init_guess[1]*(1+0.15), NWALKERS),
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
    NIT = 5_000
    NWALKERS = 64

    data_filename = 'results/lensing_desy3_test_lambda38-55_z019-027_binlog.fits'
    chain_filename = 'results/fitting_desy3_test_lambda38-55_z019-027.hdf5'
    model_name = 'NFW'
    observable = 'delta_sigma'
    cov_mode = 'diag'

    sampler = run_emcee(
        NCORES=NCORES,NIT=NIT,NWALKERS=NWALKERS,
        data_filename=data_filename,
        save_filename=chain_filename,
        model_name=model_name,
        observable=observable,
        cov_mode=cov_mode,
        #init_guess=None
    )
    # TODO: que guarde los valores de mejor ajuste!

    plot_chains(sampler.get_chain())
    plt.show()

    plot_corner(sampler);
    plt.show()
