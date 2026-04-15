import numpy as np

from fitting.constants import *
from fitting.io import DataProfile
from fitting.models import BaseModelFast

class Likelihood:
    def __init__(
            self, 
            data : DataProfile, 
            model : BaseModelFast, 
            param_limits : dict, 
            observable='delta_sigma', 
            cov_mode='full'
            ):
        
        self.R = data.R
        self.rhomean = rho_mean(data.redshift)
        self.limits = param_limits
        self.hartlap_factor = (data.Njk-len(self.R)-2)/(data.Njk-1)

        if observable=='sigma':
            self.ydata = data.Sigma
            self.cov = data.covS
            
            self.func = model.sigma
        
        elif observable=='delta_sigma':
            self.ydata = data.DSigma_t
            self.cov = data.covDSt

            self.func = model.delta_sigma

            try:
                ## only if passing sigma0 in the param_limits dict
                self.limits.pop('sigma0')
            except KeyError:
                pass

        else:
            raise ValueError('observable must be either "sigma" or "delta_sigma"')

        if cov_mode == 'full':
            self.yerr = np.linalg.inv(self.cov)*self.hartlap_factor
        elif cov_mode == 'diag':
            # this allows to use log_likelihood with both diag or full covariance!
            self.yerr = np.zeros_like(self.cov)
            np.fill_diagonal(self.yerr, 1.0/np.diag(self.cov))
        else:
            raise ValueError('cov_mode must be either "full" or "diag"')

        self.param_name = list(self.limits.keys())
        self.nparams = len(self.param_name)
        
    def log_likelihood(self, theta):
        model = self.func(self.R, *theta) #*self.rhomean
        dist = self.ydata - model
        return -0.5*np.dot(dist, np.dot(self.yerr, dist))

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

# - easy way to make a joint fit for different data but model with the same parameters.
class JointLikelihood:
    # should be a composition of two or more Likelihood instances
    pass