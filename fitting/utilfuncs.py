import numpy as np

from fitting.models import default_limits

def chi2_red(data, model, invC, ndof):
    d = (data-model)
    return np.sum(np.dot(d, np.dot(invC,d)))/ndof

def make_pos_gaussian(init_guess, NWALKERS, seed=0):
    rng = np.random.default_rng(seed)
    nparam = len(init_guess)
    pos = np.zeros((NWALKERS, nparam))
    for i in range(nparam):
        if init_guess[i]!=0.0:
            pos[:, i] = rng.normal(init_guess[i], np.abs(0.15*init_guess[i]), NWALKERS)
        else:
            pos[:, i] = rng.normal(0.0, 0.15, NWALKERS)
    return pos

def validate_pos(pos, model_name):
    rng = np.random.default_rng(seed=0)
    limits = default_limits.get(model_name)
    for i, (lmin, lmax) in enumerate(limits.values()):
        for j, p in enumerate(pos[:,i]):
            if p<lmin or p>lmax:
                print('Invalid pos, redrawing...')
                pos[j,i] = rng.uniform(lmin, lmax)
    return pos

def get_fitted_params(chain, params):
    '''
    chain : sampler.get_chain(discard=ndiscard) # posteriors
    params : L.param_name
    '''
    fitted_params = {}
    error_params = {}
    for i, p in enumerate(params):
        percentil = np.percentile(chain[:,:,i], [16,50,84])
        fitted_params[p] = percentil[1]
        error_params[p] = tuple(percentil[[0,2]]-percentil[1])

    return fitted_params, error_params