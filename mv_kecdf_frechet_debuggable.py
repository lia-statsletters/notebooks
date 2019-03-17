#from __future__ import division

from profilehooks import profile#, timecall
import cProfile
import pstats
import io


import pandas as pd
import numpy as np

import scipy.stats as spst
import statsmodels.api as sm
from scipy import optimize

import re
import time

import matplotlib.pyplot as plt

from statsmodels.nonparametric import kernels

kernel_func = dict(wangryzin=kernels.wang_ryzin,
                   aitchisonaitken=kernels.aitchison_aitken,
                   gaussian=kernels.gaussian,
                   aitchison_aitken_reg = kernels.aitchison_aitken_reg,
                   wangryzin_reg = kernels.wang_ryzin_reg,
                   gauss_convolution=kernels.gaussian_convolution,
                   wangryzin_convolution=kernels.wang_ryzin_convolution,
                   aitchisonaitken_convolution=kernels.aitchison_aitken_convolution,
                   gaussian_cdf=kernels.gaussian_cdf,
                   aitchisonaitken_cdf=kernels.aitchison_aitken_cdf,
                   wangryzin_cdf=kernels.wang_ryzin_cdf,
                   d_gaussian=kernels.d_gaussian)

def lets_be_tidy(rd,frechets):
    v_type = f'{"c"*len(rd)}'
    #threshold: number of violations by the cheapest method.
    dens_u_rot = sm.nonparametric.KDEMultivariate(data=rd, var_type=v_type, bw='normal_reference')
    cdf_dens_u_rot = dens_u_rot.cdf()
    violations_rot = count_frechet_fails(cdf_dens_u_rot, frechets)
    #how can we best use this violations?
    #nviolations= np.sum([np.sum(violations_rot['top']),
    #                     np.sum(violations_rot['bottom'])])

    tst = get_bw(rd, v_type, dens_u_rot.bw, frech_bounds=frechets)  # dens_u_rot.bw)

    #for comparison, call the package and check features and time
    dens_u_ml = sm.nonparametric.KDEMultivariate(data=rd,var_type=v_type, bw='cv_ml')

    print(tst, '\n', dens_u_rot.bw, '\n', dens_u_ml.bw)

def gpke(bwp, dataxx, data_predict, var_type, ckertype='gaussian',
         okertype='wangryzin', ukertype='aitchisonaitken', tosum=True):
    r"""Returns the non-normalized Generalized Product Kernel Estimator"""
    kertypes = dict(c=ckertype, o=okertype, u=ukertype)
    Kval = np.empty(dataxx.shape)
    for ii, vtype in enumerate(var_type):
        func = kernel_func[kertypes[vtype]]
        Kval[:, ii] = func(bwp[ii], dataxx[:, ii], data_predict[ii])

    iscontinuous = np.array([c == 'c' for c in var_type])
    dens = Kval.prod(axis=1) / np.prod(bwp[iscontinuous])
    #dens = np.nanprod(Kval,axis=1) / np.prod(bwp[iscontinuous])
    if tosum:
        return dens.sum(axis=0)
    else:
        return dens

def adjust_shape(dat, k_vars):
    """ Returns an array of shape (nobs, k_vars) for use with `gpke`."""
    dat = np.asarray(dat)
    if dat.ndim > 2:
        dat = np.squeeze(dat)
    if dat.ndim == 1 and k_vars > 1:  # one obs many vars
        nobs = 1
    elif dat.ndim == 1 and k_vars == 1:  # one obs one var
        nobs = len(dat)
    else:
        if np.shape(dat)[0] == k_vars and np.shape(dat)[1] != k_vars:
            dat = dat.T

        nobs = np.shape(dat)[0]  # ndim >1 so many obs many vars

    dat = np.reshape(dat, (nobs, k_vars))
    return dat

def calc_frechet_fails(guinea_cdf,frechets):
    #fails = {'top': [], 'bottom': []}
    N=len(guinea_cdf)
    top=np.full(N,0.)
    bot=np.full(N,0.)
    for n in range(N):
        # n_hyper_point=np.array([x[n] for x in rd])
        xdiff=guinea_cdf[n]-frechets['top'][n]
        if xdiff>0:
            top[n]=xdiff

        xdiff=frechets['bottom'][n] - guinea_cdf[n]
        if xdiff>0:
            bot[n]=xdiff

    return {'top': top,
            'bottom': bot}


def count_frechet_fails(guinea_cdf,frechets):
    fails={'top':[], 'bottom':[]}
    for n in range(len(guinea_cdf)):
        if guinea_cdf[n]>frechets['top'][n]:
            fails['top'].append(True)
        else:
            fails['top'].append(False)

        if guinea_cdf[n]<frechets['bottom'][n]:
            fails['bottom'].append(True)
        else:
            fails['bottom'].append(False)
    return {'top':np.array(fails['top']),
            'bottom':np.array(fails['bottom'])}

def get_frechets(dvars):
    d=len(dvars)
    n=len(dvars[0])
    dimx=np.array(range(d))
    un=np.ones(d,dtype=int)
    bottom_frechet = np.array([max( np.sum( dvars[dimx,un*i] ) +1-d, 0 )
                               for i in range(n) ])
    top_frechet = np.array([min([y[i] for y in dvars]) for i in range(n)])
    return {'top': top_frechet, 'bottom': bottom_frechet}



class LeaveOneOut(object):
    def __init__(self, X):
        self.X = np.asarray(X)

    def __iter__(self):
        X = self.X
        nobs, k_vars = np.shape(X)

        for i in range(nobs):
            index = np.ones(nobs, dtype=np.bool)
            index[i] = False
            yield X[index, :]


def cdf(dataxx, bw, var_type, frech_bounds=None):
    data_predict = dataxx
    nobs=np.shape(data_predict)[0]
    cdf_est = []
    #longer code but faster evaluation
    def ze_cdf_eval(bw, data_predict, i, var_type):
        return gpke(bw, data_predict, data_predict[i, :],
                    var_type,ckertype="gaussian_cdf",
                    ukertype="aitchisonaitken_cdf",okertype='wangryzin_cdf')

    #if not frech_bounds:
    for i in range(nobs):#np.shape(data_predict)[0]):
        ze_value=ze_cdf_eval(bw, data_predict, i, var_type)
        cdf_est.append( ze_value )
    cdf_est = np.squeeze(cdf_est)/ nobs
    return cdf_est

def frechet_likelihood(bww, datax, var_type, frech_bounds, func=None, debug_mode=False,):

    cdf_est = cdf(datax, bww, var_type)  # frech_bounds=frech_bounds)
    d_violations = calc_frechet_fails(cdf_est, frech_bounds)
    width_bound = frech_bounds['top'] - frech_bounds['bottom']
    viols=(d_violations['top']+d_violations['bottom'])/width_bound
    L= np.sum(viols)

    if debug_mode:
        nobs = len(datax)
        print(bww, 'violations (top,bottom):',
          f'({np.sum(~np.isin(d_violations["top"],0))},'
          f'{np.sum(~np.isin(d_violations["bottom"],0))})\n',
          'out of',nobs, 'samples, likelihood:', L, '\n')

    return L


def loo_likelihood(bww, datax, var_type, func=lambda x: x, debug_mode=False):
    #if frechet bounds available, check violations for this bandwidth.

    LOO = LeaveOneOut(datax) #iterable, one sample less for each sample.
    L = 0 #score
    for i, X_not_i in enumerate(LOO):
        f_i = gpke(bww, dataxx=-X_not_i, data_predict=-datax[i, :],
                   var_type=var_type)
        L += func(f_i)
    if debug_mode:
        print('\n',bww,'Log likelihood:',-L)
    return -L


def get_bw(datapfft, var_type, reference, frech_bounds=None):
    # Using leave-one-out likelihood
    # the initial value for the optimization is the normal_reference
    # h0 = normal_reference()

    data = adjust_shape(datapfft, len(var_type))

    if not frech_bounds:
        fmin =lambda bw, funcx: loo_likelihood(bw, data, var_type, func=funcx)
        argsx=(np.log,)
    else:
        fmin = lambda bw, funcx: frechet_likelihood(bw, data, var_type,
                                                    frech_bounds, func=funcx)
        argsx=(None,) #second element of tuple is if debug mode

    h0 = reference
    bw = optimize.fmin(fmin, x0=h0, args=argsx, #feeding logarithm for loo
                       maxiter=1e3, maxfun=1e3, disp=0, xtol=1e-3)
    # bw = self._set_bw_bounds(bw)  # bound bw if necessary
    return bw

#@profile(sort='cumulative',filename='/home/lia/liaProjects/outs/experiments.txt')
def profile_run(rd,frechets,iterx):
    dims=len(rd)
    n=len(rd[0])
    v_type = f'{"c"*dims}'
    # threshold: number of violations by the cheapest method.
    dens_u_rot = sm.nonparametric.KDEMultivariate(data=rd, var_type=v_type, bw='normal_reference')
    cdf_dens_u_rot = dens_u_rot.cdf()
    violations_rot = count_frechet_fails(cdf_dens_u_rot, frechets)


    #profile frechets
    pr = cProfile.Profile()
    pr.enable()
    bw_frechets = get_bw(rd, v_type, dens_u_rot.bw, frech_bounds=frechets)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()
    s = s.getvalue()

    with open(f'/home/lia/liaProjects/outs/frechet-profile-d{dims}-n{n}-iter{iterx}.txt', 'w+') as f:
        f.write(s)


    #profile cv_ml
    pr = cProfile.Profile()
    pr.enable()
    bw_cv_ml = get_bw(rd, v_type, dens_u_rot.bw)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()
    s = s.getvalue()

    with open(f'/home/lia/liaProjects/outs/loo-ml-profile-d{dims}-n{n}-iter{iterx}.txt', 'w+') as f:
        f.write(s)


    return bw_frechets,bw_cv_ml


def generate_experiments(reps,n,params, distr, dims):
    bws_frechet={f'bw_{x}':[] for x in params}
    bws_cv_ml={f'bw_{x}':[] for x in params}


    for iteration in range(reps):
        mvdata = {k: distr.rvs(*params[k], size=n) for k in params}
        rd = np.array(list(mvdata.values()))

        # get frechets and thresholds
        frechets = get_frechets(rd)

        bw_frechets, bw_cv_ml=profile_run(rd, frechets,iteration)

        for ix,x in enumerate(params):
            bws_frechet[f'bw_{x}'].append(bw_frechets[ix])
            bws_cv_ml[f'bw_{x}'].append(bw_cv_ml[ix])

    pd.DataFrame(bws_frechet).to_csv(f'/home/lia/liaProjects/outs/bws_frechet_d{dims}-n{n}-iter{reps}.csv')
    pd.DataFrame(bws_cv_ml).to_csv(f'/home/lia/liaProjects/outs/bws_cv_ml_d{dims}-n{n}-iter{reps}.csv')


def aggregate_experiments():

    fl = f'/home/lia/liaProjects/outs/d{dims}-n{n}-reps{reps}.csv'
    #with open(fl, 'w+') as f:
    #    s = io.StringIO()
    #    k = pstats.Stats('/home/lia/liaProjects/outs/experiments.txt', stream=s).sort_stats('cumtime')
    #    k.print_stats()
    #    f.write(s.getvalue())

    k = pd.read_csv(fl,
                    header=3, delim_whitespace=True,
                    usecols={'ncalls', 'tottime', 'percall',
                             'cumtime', 'filename:lineno(function)'}
                    ).rename(columns={'filename:lineno(function)': 'from_func'}).dropna()
    tokeep = np.where([1 if not not re.search('likelihood', x) else 0 for x in k.from_func.values])[0]
    k = k.iloc[tokeep]
    repl=['frechet' if not not re.search('frechet_likelihood', x) else 'loo_cv_ml' for x in k.from_func.values]
    k['from_func']=np.array(repl)
    k.to_csv(f'/home/lia/liaProjects/outs/summary_d{dims}-n{n}-reps{reps}.csv',
             index=False)

def main():
    reps = 30

    params = {'horns': (0.5, 0.5),
              'horns1': (0.45, 0.55),
              #'shower': (5., 2.),
              #'grower': (2., 5.),
              #'symetric': (2., 2.)
              }
    dims = len(params)
    ns = [300, 600, 1000, 3000, 4000]
    distr = spst.beta
    for n in ns:
        generate_experiments(reps,n,params,distr,dims)

    #lets_be_tidy(rd,frechets)

if __name__ == "__main__":
    main()