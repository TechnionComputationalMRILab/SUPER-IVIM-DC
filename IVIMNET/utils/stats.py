import numpy as np
from scipy import stats
from IVIMNET.utils.ivim_functions import ivim


def empirical_neg_log_prior(Dt0, Fp0, Dp0, S00=None):
    """
    This function determines the negative of the log of the empirical prior probability of the IVIM parameters
    :param Dt0: 1D Array with the initial D estimates
    :param Dt0: 1D Array with the initial f estimates
    :param Dt0: 1D Array with the initial D* estimates
    :param Dt0: 1D Array with the initial S0 estimates (optional)
    """
    # Dp0, Dt0, Fp0 are flattened arrays
    # only take valid voxels along, in which the initial estimates were sensible and successful
    Dp_valid = (1e-8 < np.nan_to_num(Dp0)) & (np.nan_to_num(Dp0) < 1 - 1e-8)
    Dt_valid = (1e-8 < np.nan_to_num(Dt0)) & (np.nan_to_num(Dt0) < 1 - 1e-8)
    Fp_valid = (1e-8 < np.nan_to_num(Fp0)) & (np.nan_to_num(Fp0) < 1 - 1e-8)
    # determine whether we fit S0
    if S00 is not None:
        S0_valid = (1e-8 < np.nan_to_num(S00)) & (np.nan_to_num(S00) < 2 - 1e-8)
        valid = Dp_valid & Dt_valid & Fp_valid & S0_valid
        Dp0, Dt0, Fp0, S00 = Dp0[valid], Dt0[valid], Fp0[valid], S00[valid]
    else:
        valid = Dp_valid & Dt_valid & Fp_valid
        Dp0, Dt0, Fp0 = Dp0[valid], Dt0[valid], Fp0[valid]
    # determine prior's shape. Note that D, D* and S0 are shaped as lognorm distributions whereas f is a beta distribution
    Dp_shape, _, Dp_scale = stats.lognorm.fit(Dp0, floc=0)
    Dt_shape, _, Dt_scale = stats.lognorm.fit(Dt0, floc=0)
    Fp_a, Fp_b, _, _ = stats.beta.fit(Fp0, floc=0, fscale=1)
    if S00 is not None:
        S0_a, S0_b, _, _ = stats.beta.fit(S00, floc=0, fscale=2)

    # define the prior
    def neg_log_prior(p):
        # depends on whether S0 is fitted or not
        if len(p) is 4:
            Dt, Fp, Dp, S0 = p[0], p[1], p[2], p[3]
        else:
            Dt, Fp, Dp = p[0], p[1], p[2]
        # make D*<D very unlikely
        if (Dp < Dt):
            return 1e8
        else:
            eps = 1e-8
            Dp_prior = stats.lognorm.pdf(Dp, Dp_shape, scale=Dp_scale)
            Dt_prior = stats.lognorm.pdf(Dt, Dt_shape, scale=Dt_scale)
            Fp_prior = stats.beta.pdf(Fp, Fp_a, Fp_b)
            # determine and return the prior for D, f and D* (and S0)
            if len(p) is 4:
                S0_prior = stats.beta.pdf(S0 / 2, S0_a, S0_b)
                return -np.log(Dp_prior + eps) - np.log(Dt_prior + eps) - np.log(Fp_prior + eps) - np.log(
                    S0_prior + eps)
            else:
                return -np.log(Dp_prior + eps) - np.log(Dt_prior + eps) - np.log(Fp_prior + eps)

    return neg_log_prior


def neg_log_likelihood(p, bvalues, dw_data):
    """
    This function determines the negative of the log of the likelihood of parameters p, given the data dw_data for the Bayesian fit
    :param p: 1D Array with the estimates of D, f, D* and (optionally) S0
    :param bvalues: 1D array with b-values
    :param dw_data: 1D Array diffusion-weighted data
    :returns: the log-likelihood of the parameters given the data
    """
    if len(p) is 4:
        return 0.5 * (len(bvalues) + 1) * np.log(
            np.sum((ivim(bvalues, p[0], p[1], p[2], p[3]) - dw_data) ** 2))  # 0.5*sum simplified
    else:
        return 0.5 * (len(bvalues) + 1) * np.log(
            np.sum((ivim(bvalues, p[0], p[1], p[2], 1) - dw_data) ** 2))  # 0.5*sum simplified


def neg_log_posterior(p, bvalues, dw_data, neg_log_prior):
    """
    This function determines the negative of the log of the likelihood of parameters p, given the prior likelihood and the data
    :param p: 1D Array with the estimates of D, f, D* and (optionally) S0
    :param bvalues: 1D array with b-values
    :param dw_data: 1D Array diffusion-weighted data
    :param neg_log_prior: prior likelihood function (created with empirical_neg_log_prior)
    :returns: the posterior probability given the data and the prior
    """
    return neg_log_likelihood(p, bvalues, dw_data) + neg_log_prior(p)


def goodness_of_fit(bvalues, Dt, Fp, Dp, S0, dw_data):
    """
    Calculates the R-squared as a measure for goodness of fit.
    input parameters are
    :param b: 1D Array b-values
    :param Dt: 1D Array with fitted D
    :param Fp: 1D Array with fitted f
    :param Dp: 1D Array with fitted D*
    :param S0: 1D Array with fitted S0 (or ones)
    :param dw_data: 2D array containing data, as voxels x b-values

    :return R2: 1D Array with the R-squared for each voxel
    """
    # simulate the IVIM signal given the D, f, D* and S0
    datasim = ivim(np.tile(np.expand_dims(bvalues, axis=0), (len(Dt), 1)),
                   np.tile(np.expand_dims(Dt, axis=1), (1, len(bvalues))),
                   np.tile(np.expand_dims(Fp, axis=1), (1, len(bvalues))),
                   np.tile(np.expand_dims(Dp, axis=1), (1, len(bvalues))),
                   np.tile(np.expand_dims(S0, axis=1), (1, len(bvalues)))).astype('f')

    # calculate R-squared given the estimated IVIM signal and the data
    norm = np.mean(dw_data, axis=1)
    ss_tot = np.sum(np.square(dw_data - norm[:, None]), axis=1)
    ss_res = np.sum(np.square(dw_data - datasim), axis=1)
    R2 = 1 - (ss_res / ss_tot)  # R-squared
    return R2


def MSE(bvalues, Dt, Fp, Dp, S0, dw_data):
    """
    Calculates the MSE as a measure for goodness of fit.
    input parameters are
    :param b: 1D Array b-values
    :param Dt: 1D Array with fitted D
    :param Fp: 1D Array with fitted f
    :param Dp: 1D Array with fitted D*
    :param S0: 1D Array with fitted S0 (or ones)
    :param dw_data: 2D array containing data, as voxels x b-values

    :return MSError: 1D Array with the R-squared for each voxel
    """
    # simulate the IVIM signal given the D, f, D* and S0
    datasim = ivim(np.tile(np.expand_dims(bvalues, axis=0), (len(Dt), 1)),
                   np.tile(np.expand_dims(Dt, axis=1), (1, len(bvalues))),
                   np.tile(np.expand_dims(Fp, axis=1), (1, len(bvalues))),
                   np.tile(np.expand_dims(Dp, axis=1), (1, len(bvalues))),
                   np.tile(np.expand_dims(S0, axis=1), (1, len(bvalues)))).astype('f')

    # calculate R-squared given the estimated IVIM signal and the data
    MSError = np.mean(np.square(dw_data-datasim),axis=1)  # R-squared
    return MSError
