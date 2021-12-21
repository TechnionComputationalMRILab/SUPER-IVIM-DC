import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .least_squares import fit_least_squares
from SUPER_IVIM_DC.utils.checkarg import checkarg_lsq
from SUPER_IVIM_DC.utils.stats import empirical_neg_log_prior, neg_log_posterior
from SUPER_IVIM_DC.utils.order import order


def fit_bayesian_array(bvalues, dw_data, paramslsq, arg):
    """
    This is an implementation of the Bayesian IVIM fit for arrays. The fit is taken from Barbieri et al. which was
    initially introduced in http://arxiv.org/10.1002/mrm.25765 and later further improved in
    http://arxiv.org/abs/1903.00095. If found useful, please cite those papers.

    :param bvalues: Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param paramslsq: 2D Array with initial estimates for the parameters. These form the base for the Bayesian prior
    distribution and are typically obtained by least squares fitting of the data
    :param arg: an object with fit options, with attributes:
    arg.fitS0 --> Boolean; False fixes S0 to 1, True fits S0
    arg.jobs --> Integer specifying the number of parallel processes used in fitting. If <2, regular fitting is used instead
    arg.bounds --> 2D Array of fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max])
    :return Dt: Array with D in each voxel
    :return Fp: Array with f in each voxel
    :return Dp: Array with Dp in each voxel
    :return S0: Array with S0 in each voxel
    """
    arg = checkarg_lsq(arg)
    # fill out missing args
    Dt0, Fp0, Dp0, S00 = paramslsq
    # determine prior
    if arg.fitS0:
        neg_log_prior = empirical_neg_log_prior(Dt0, Fp0, Dp0, S00)
    else:
        neg_log_prior = empirical_neg_log_prior(Dt0, Fp0, Dp0)
    single = False
    # determine whether we fit parallel or not
    if arg.jobs > 1:
        try:
            # do parallel bayesian fit
            def parfun(i):
                # starting point
                x0 = [Dt0[i], Fp0[i], Dp0[i], S00[i]]
                return fit_bayesian(bvalues, dw_data[i, :], neg_log_prior, x0, fitS0=arg.fitS0)

            output = Parallel(n_jobs=arg.jobs)(delayed(parfun)(i) for i in tqdm(range(len(dw_data)), position=0,
                                                                                leave=True))
            Dt_pred, Fp_pred, Dp_pred, S0_pred = np.transpose(output)
        except:
            single = True
    else:
        single = True
    if single:
        # do serial; intialising arrays
        Dp_pred = np.zeros(len(dw_data))
        Dt_pred = np.zeros(len(dw_data))
        Fp_pred = np.zeros(len(dw_data))
        S0_pred = np.zeros(len(dw_data))
        # fill in array while looping over voxels
        for i in tqdm(range(len(dw_data)), position=0, leave=True):
            # starting point
            x0 = [Dt0[i], Fp0[i], Dp0[i], S00[i]]
            Dt, Fp, Dp, S0 = fit_bayesian(bvalues, dw_data[i, :], neg_log_prior, x0, fitS0=arg.fitS0)
            Dp_pred[i] = Dp
            Dt_pred[i] = Dt
            Fp_pred[i] = Fp
            S0_pred[i] = S0
    return Dt_pred, Fp_pred, Dp_pred, S0_pred


def fit_bayesian(bvalues, dw_data, neg_log_prior, x0=[0.001, 0.2, 0.05], fitS0=True):
    """
    This is an implementation of the Bayesian IVIM fit. It returns the Maximum a posterior probability.
    The fit is taken from Barbieri et al. which was initially introduced in http://arxiv.org/10.1002/mrm.25765 and
    later further improved in http://arxiv.org/abs/1903.00095. If found useful, please cite those papers.

    :param bvalues: Array with the b-values
    :param dw_data: 1D Array with diffusion-weighted signal at different b-values
    :param neg_log_prior: the prior
    :param x0: 1D array with initial parameter guess
    :param fitS0: boolean, if set to False, S0 is not fitted
    :return Dt: estimated D
    :return Fp: estimated f
    :return Dp: estimated D*
    :return S0: estimated S0 (optional)
    """
    try:
        # define fit bounds
        bounds = [(0, 0.006), (0, 1), (0.006, 0.3), (0, 2)]
        # Find the Maximum a posterior probability (MAP) by minimising the negative log of the posterior
        if fitS0:
            params = minimize(neg_log_posterior, x0=x0, args=(bvalues, dw_data, neg_log_prior), bounds=bounds)
        else:
            params = minimize(neg_log_posterior, x0=x0[:3], args=(bvalues, dw_data, neg_log_prior), bounds=bounds[:3])
        if not params.success:
            raise params.message
        if fitS0:
            Dt, Fp, Dp, S0 = params.x[0], params.x[1], params.x[2], params.x[3]
        else:
            Dt, Fp, Dp = params.x[0], params.x[1], params.x[2]
            S0 = 1
        return order(Dt, Fp, Dp, S0)
    except:
        # if fit fails, return regular lsq-fit result
        # print('a bayes fit failed')
        return fit_least_squares(bvalues, dw_data, S0_output=True)
