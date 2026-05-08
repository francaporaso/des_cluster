import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from emcee.backends import HDFBackend

from fitting.models import default_limits, models_dict
from fitting.io import read_dataprofile_fits
from fitting.utilfuncs import load_fitted_params

def plot_chains(chain):

    nit, _, nparams = chain.shape
    fig, axes = plt.subplots(nparams,1, sharex=True)
    if nparams!=1:
        for i in range(nparams):
            axes[i].plot(chain[:,:,i], 'k', alpha=0.3)
            axes[i].set_xlim(0.0, nit)
            axes[i].set_ylabel(f'$a_{i}$')
            axes[i].yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel('Step Number')
    else:
        axes.plot(chain[:,:,0], 'k', alpha=0.3)
        axes.set_xlim(0.0, nit)
        axes.set_ylabel('$a_0$')
        axes.yaxis.set_label_coords(-0.1, 0.5)

    plt.show()
    return fig

def plot_corner(sampler, discard=100, fig=None, color=None, **corner_kwargs):

    flat_samples = sampler.get_chain(discard=discard, flat=True)
    if fig==None:
        fig = corner(flat_samples, color=color, **corner_kwargs);
        return fig
    else:
        corner(flat_samples, fig=fig, color=color, **corner_kwargs);

def plot_pos(pos):

    nwalkers, nparams = pos.shape

    fig, axes = plt.subplots(1, nparams, figsize=(2*nparams, 2))
    for i in range(nparams):
        axes[i].hist(pos.T[i], bins=nwalkers//10)
        axes[i].set_xlabel(f'$a_{i}$')

    return fig

def plot_getdist(labels, names, discard, model, samplers, samplename):
    from getdist import plots as gdplots, MCSamples
    log_prob = {}
    chain = {}
    log_prob_list = {}
    chain_list = {}

    samples = {}

    for i,spl in enumerate(samplers):
        log_prob[i] = spl.get_log_prob(discard=discard)
        chain[i] = spl.get_chain(discard=discard)
        log_prob_list[i] = [log_prob[i][:,j] for j in range(log_prob[i].shape[1])]
        chain_list[i] = [chain[i][:,j] for j in range(chain[i].shape[1])]

        samples[i] = MCSamples(
            samples=chain_list[i],
            loglikes=[-lp for lp in log_prob_list[i]],
            ranges=default_limits.get(model),
            labels=labels,
            names=names,
            label=samplename[i],
        )

    g = gdplots.get_subplot_plotter()
    g.triangle_plot(list(samples.values()), filled=True);


def plot_fittedmodel(chainfile, ax=None, model_name='NFW', cov_mode='full', components=True):

    mcmc = HDFBackend(chainfile, name=f'emcee/{model_name}/{cov_mode}', read_only=True)
    fit, err = load_fitted_params(chainfile, model_name=model_name, cov_mode=cov_mode)
    model = models_dict[model_name](redshift=data.redshift)
    r = np.geomspace(data.R.min(), data.R.max(), 100)

    if ax is None:
        fig, ax = plt.subplots(1,1)

    ax.plot(
        r,
        model.delta_sigma(r, **fit),
        c='r'
    )

    if components and model_name=='NFW':
        ax.plot(
            r,
            fit['pcc']*model.dsigma_1h(
                r, fit['M200'], model.c_200(fit['M200'])
            ),
            c='r',
            ls='--'
        )
        ax.plot(
            r,
            (1-fit['pcc'])*model.dsigma_miss(
                r, fit['M200'], model.c_200(fit['M200'])
            ),
            c='r',
            ls='--'
        )
    return ax

def plot_profile(datafile, chainfile, ax=None, model_name='NFW', cov_mode='full', components=True):

    data = read_dataprofile_fits(datafile)

    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax.errorbar(
        data.R,
        data.DSigma_t,
        np.sqrt(np.diag(data.covDSt)),
        fmt='.k',
        capsize=2,
    )
    try:
        plot_fittedmodel(chainfile, ax=ax, model_name=model_name, cov_mode=cov_mode, components=components)
    finally:
        return ax
