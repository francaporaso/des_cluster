import emcee
import numpy as np
import toml
import h5py
import time
from multiprocessing import Pool
from astropy.io import fits
import matplotlib.pyplot as plt

from fitting.plotting import plot_chains, plot_corner
from fitting.utilfuncs import load_fitted_params, get_fitted_params

def ln_lambda_Mz(mass, redshift, lam0, lamM, lamZ):
    Mpiv = 3e14
    zpiv = 0.6
    return np.log(lam0) + lamM*np.log(mass/Mpiv) + lamZ*np.log((1+redshift)/(1+zpiv))

def ln_M_lambdaz(richness, redshift, M0, F_l, G_z):
    lampiv = 40.0
    zpiv = 0.35
    return np.log(M0) + F_l*np.log(richness/lampiv) + G_z*np.log((1.0+z)/(1.0+zpiv))

funcs_dict = {
    'mass-richness': {
        'func': ln_M_lambdaz,
        'nparams': 3,
        'param_names': ['M0','F_l','G_z','sigma_M'],
        'param_limts':{
            'M0':(1e13, 1e15),
            'F_l':(0.01, 10),
            'G_z':(-1.0, 1.0)

        }
    },
    'richness-mass': {
        'func': ln_lambda_Mz,
        'nparams': 4,
        'param_names': ['lam0', 'lamM', 'lamZ', 'sigma_logl'],
        'param_limits': {
            'lam0':(5.0, 250.0),
            'lamM':(0.01, 10.0),
            'lamZ':(-5.0, 5.0),
            'sigma_logl':(0.01, 1.0)
        }
    }
}

class LamMassRelation:

    def __init__(
            self,
            redshift,
            xdata,
            ydata,
            yerr,
            relation='mass-richness',
            xerr=None,
        ):


        self.redshift = redshift

        self.xdata = xdata
        if xerr is not None:
            self.xerr = xerr
        else:
            self.xerr = np.zeros_like(xdata)
        self.ydata = ydata
        self.yerr = yerr

        self.nparams = funcs_dict[relation]['nparams']
        self.func = funcs_dict[relation]['func']
        self.param_name = funcs_dict[relation]['param_names']
        self.limits = funcs_dict[relation]['param_limits']


    def log_likelihood(self, theta):

        lam0, lamM, lamZ, sigma_logl = theta

        model = self.func(self.xdata, self.redshift, lam0, lamM, lamZ)
        dist = self.ydata - model

        s2 = sigma_logl**2 + (lamM*self.yerr)**2

        return -0.5 * np.sum(dist**2 / s2 + np.log(2.0*np.pi*s2))
        # return -0.5*np.dot(dist, np.dot(err, dist))

    def log_prior(self, theta):
        ### tener cuidado con el orden de lims!
        if np.prod(
            [self.limits[self.param_name[j]][0] < theta[j] < self.limits[self.param_name[j]][1] for j in range(self.nparams)],
            dtype=bool
        ): return 0
        return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

# ==================
def make_init_pos(init_guess, nwalkers, seed=0):

    rng = np.random.default_rng(0)
    init_pos = np.zeros((len(init_guess), nwalkers))
    for i, ig in enumerate(init_guess):
        l1 = ig*(1-0.2)
        l2 = ig*(1+0.2)
        if l1<l2:
            init_pos[i,:] = rng.uniform(l1, l2, nwalkers)
        else:
            init_pos[i,:] = rng.uniform(l2, l1, nwalkers)

    return init_pos

def run_fit(
        mass,
        redshift,
        richness,
        richness_err,
        mass_err,
        init_guess = [50.0, 1.3, -0.3, 0.2],
        nwalkers=10,
        nit=100,
        ncores=2,
        savefilename='test.hdf5'
    ):

    L = LamMassRelation(
        ydata=richness,
        yerr=richness_err,
        redshift=redshift,
        xdata=mass,
        xerr=mass_err,
        relation='richness-mass'
    )

    group_name = 'emcee/'

    backend = emcee.backends.HDFBackend(savefilename, name=group_name)

    init_pos = make_init_pos(init_guess, nwalkers)

    with Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers, ndim=L.nparams, log_prob_fn=L.log_probability, pool=pool, backend=backend
        )
        sampler.run_mcmc(init_pos.T, nit, progress=True, store=True)

    return sampler

def main():
    config = toml.load('fitting/config.toml')
    NCORES = config['RUN']['NCORES']
    NWALKERS = config['RUN']['NWALKERS']
    NIT = config['RUN']['NIT']
    SAMPLE = config['NAMES']['SAMPLE']
    ZMIN = config['LENSES']['ZMIN']
    ZMAX = config['LENSES']['ZMAX']
    LMIN = config['LENSES']['LMIN']
    LMAX = config['LENSES']['LMAX']
    PLOT = config['RUN']['PLOT']
    MODEL = config['RUN']['MODEL']
    COVMODE = config['RUN']['COVMODE']
    savefilename = f'rich-mass_rel_{SAMPLE}.hdf5'

    m200 = np.zeros((3,4))
    e_m200 = np.zeros((2,3,4))
    richness = np.zeros((3,4))
    meanz = np.zeros((3,4))
    for z, (zmin,zmax) in enumerate(zip(ZMIN, ZMAX)):
        zstr = f'z{100*zmin:03.0f}-{100*zmax:03.0f}'
        for l, (lmin, lmax) in enumerate(zip(LMIN, LMAX)):
            lstr = f'lambda{lmin:02.0f}-{lmax:02.0f}'

            datafile = f'results/lensing_desy3_{SAMPLE}_{zstr}_{lstr}_bin15log.fits'
            chainfile = f'results/fitting_desy3_{SAMPLE}_{zstr}_{lstr}.hdf5'

            fit, err = load_fitted_params(chainfile, model_name=MODEL, cov_mode=COVMODE)

            m200[z,l] = fit['M200']
            e_m200[0,z,l] = -err['M200'][0]
            e_m200[1,z,l] = err['M200'][1]

            with fits.open(datafile) as f:
                richness[z,l] = f[0].header['L_MEAN']
                meanz[z,l] = f[0].header['Z_MEAN']

    mass_err_mean = 0.5*(e_m200[0,:,:] + e_m200[1,:,:])

    sampler = run_fit(
        mass=m200,
        mass_err=e_m200,
        redshift=meanz,
        richness=np.log(richness),
        richness_err=None,
        nwalkers=NWALKERS,
        nit=NIT,
        ncores=NCORES,
        savefilename=savefilename
    )

    param_names = ['lam0', 'lamM', 'lamZ', 's_logl']
    # not possible to fix params for now
    fitpar, errpar = get_fitted_params(sampler.get_chain(discard=int(NIT*0.15)), param_names)

    with h5py.File(savefilename, 'a') as f:
        group_path = f'fittedparams/{MODEL}/{COVMODE}'

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

        plot_corner(sampler, discard=int(NIT*0.15));
        plt.show()

if __name__ == '__main__':
    print(' Start '.center(15, '>'))
    t = time.time()
    main()
    print(' End '.center(15, '<'))
    print('>> Took {(time.time()-t)/60:2.0f} s')
