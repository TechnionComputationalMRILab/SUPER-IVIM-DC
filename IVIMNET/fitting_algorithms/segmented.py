import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import curve_fit


def fit_segmented_array(bvalues, dw_data, njobs=4, bounds=([0, 0, 0.005], [0.005, 0.7, 0.3]), cutoff=75):
    """
    This is an implementation of the segmented fit, in which we first estimate D using a curve fit to b-values>cutoff;
    then estimate f from the fitted S0 and the measured S0 and finally estimate D* while fixing D and f. This fit
    is done on an array.
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param njobs: Integer determining the number of parallel processes; default = 4
    :param bounds: 2D Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]).
        Default: ([0.005, 0, 0, 0.8], [0.3, 0.7, 0.005, 1.2])
    :param cutoff: cutoff value for determining which data is taken along in fitting D
    :return Dt: 1D Array with D in each voxel
    :return Fp: 1D Array with f in each voxel
    :return Dp: 1D Array with Dp in each voxel
    :return S0: 1D Array with S0 in each voxel
    """
    # first we normalise the signal to S0
    S0 = np.mean(dw_data[:, bvalues == 0], axis=1)
    dw_data = dw_data / S0[:, None]
    # here we try parallel computing, but if fails, go back to computing one single core.
    single = False
    if njobs > 2:
        try:
            # define the parallel function
            def parfun(i):
                return fit_segmented(bvalues, dw_data[i, :], bounds=bounds, cutoff=cutoff)

            output = Parallel(n_jobs=njobs)(delayed(parfun)(i) for i in tqdm(range(len(dw_data)),
                                                                             position=0, leave=True))
            Dt, Fp, Dp = np.transpose(output)
        except:
            # if fails, retry using single core
            single = True
    else:
        # or, if specified, immediately go to single core
        single = True
    if single:
        # initialize empty arrays
        Dp = np.zeros(len(dw_data))
        Dt = np.zeros(len(dw_data))
        Fp = np.zeros(len(dw_data))
        for i in tqdm(range(len(dw_data)), position=0, leave=True):
            # fill arrays with fit results on a per voxel base:
            Dt[i], Fp[i], Dp[i] = fit_segmented(bvalues, dw_data[i, :], bounds=bounds, cutoff=cutoff)
    return [Dt, Fp, Dp, S0]


def fit_segmented(bvalues, dw_data, bounds=([0, 0, 0.005], [0.005, 0.7, 0.3]), cutoff=75):
    """
    This is an implementation of the segmented fit, in which we first estimate D using a curve fit to b-values>cutoff;
    then estimate f from the fitted S0 and the measured S0 and finally estimate D* while fixing D and f.
    :param bvalues: Array with the b-values
    :param dw_data: Array with diffusion-weighted signal at different b-values
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]).
        Default: ([0.005, 0, 0, 0.8], [0.3, 0.7, 0.005, 1.2])
    :param cutoff: cutoff value for determining which data is taken along in fitting D
    :return Dt: Fitted D
    :return Fp: Fitted f
    :return Dp: Fitted Dp
    :return S0: Fitted S0
    """
    try:
        # determine high b-values and data for D
        high_b = bvalues[bvalues >= cutoff]
        high_dw_data = dw_data[bvalues >= cutoff]
        # correct the bounds. Note that S0 bounds determine the max and min of f
        bounds1 = ([bounds[0][0] * 1000., 1 - bounds[1][1]], [bounds[1][0] * 1000., 1. - bounds[0][
            1]])  # By bounding S0 like this, we effectively insert the boundaries of f
        # fit for S0' and D
        params, _ = curve_fit(lambda b, Dt, int: int * np.exp(-b * Dt / 1000), high_b, high_dw_data,
                              p0=(1, 1),
                              bounds=bounds1)
        Dt, Fp = params[0] / 1000, 1 - params[1]
        # remove the diffusion part to only keep the pseudo-diffusion
        dw_data_remaining = dw_data - (1 - Fp) * np.exp(-bvalues * Dt)
        bounds2 = (bounds[0][2], bounds[1][2])
        # fit for D*
        params, _ = curve_fit(lambda b, Dp: Fp * np.exp(-b * Dp), bvalues, dw_data_remaining, p0=0.1, bounds=bounds2)
        Dp = params[0]
        return Dt, Fp, Dp
    except:
        # if fit fails, return zeros
        # print('segnetned fit failed')
        return 0., 0., 0.
