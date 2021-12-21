import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from .segmented import fit_segmented
from SUPER_IVIM_DC.utils.ivim_functions import ivimN, ivimN_noS0
from SUPER_IVIM_DC.utils.order import order


def fit_least_squares_array(bvalues, dw_data, S0_output=True, fitS0=True, njobs=4,
                            bounds=([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3])):
    """
    This is an implementation of the conventional IVIM fit. It is fitted in array form.
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; default = True
    :param fix_S0: Boolean determining whether to fix S0 to 1; default = False
    :param njobs: Integer determining the number of parallel processes; default = 4
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]).
        Default: ([0.005, 0, 0, 0.8], [0.3, 0.7, 0.005, 1.2])
    :return Dt: 1D Array with D in each voxel
    :return Fp: 1D Array with f in each voxel
    :return Dp: 1D Array with Dp in each voxel
    :return S0: 1D Array with S0 in each voxel
    """
    # normalise the data to S(value=0)
    S0 = np.mean(dw_data[:, bvalues == 0], axis=1)
    dw_data = dw_data / S0[:, None]
    single = False
    # split up on whether we want S0 as output
    if S0_output:
        # check if parallel is desired
        if njobs > 1:
            try:
                # defining parallel function
                def parfun(i):
                    return fit_least_squares(bvalues, dw_data[i, :], S0_output=S0_output, fitS0=fitS0, bounds=bounds)

                output = Parallel(n_jobs=njobs)(delayed(parfun)(i) for i in tqdm(range(len(dw_data)),
                                                                                 position=0, leave=True))
                Dt, Fp, Dp, S0 = np.transpose(output)
            except:
                single = True
        else:
            single = True
        if single:
            # run on single core, instead. Defining empty arrays
            Dp = np.zeros(len(dw_data))
            Dt = np.zeros(len(dw_data))
            Fp = np.zeros(len(dw_data))
            S0 = np.zeros(len(dw_data))
            # running in a single loop and filling arrays
            for i in tqdm(range(len(dw_data)), position=0, leave=True):
                Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], S0_output=S0_output, fitS0=fitS0,
                                                               bounds=bounds)
        return [Dt, Fp, Dp, S0]
    else:
        # if S0 is not exported
        if njobs > 1:
            try:
                def parfun(i):
                    return fit_least_squares(bvalues, dw_data[i, :], fitS0=fitS0, bounds=bounds)

                output = Parallel(n_jobs=njobs)(delayed(parfun)(i) for i in tqdm(range(len(dw_data)),
                                                                                 position=0, leave=True))
                Dt, Fp, Dp = np.transpose(output)
            except:
                single = True
        else:
            single = True
        if single:
            Dp = np.zeros(len(dw_data))
            Dt = np.zeros(len(dw_data))
            Fp = np.zeros(len(dw_data))
            for i in range(len(dw_data)):
                Dt[i], Fp[i], Dp[i] = fit_least_squares(bvalues, dw_data[i, :], S0_output=S0_output, fitS0=fitS0,
                                                        bounds=bounds)
        return [Dt, Fp, Dp]


def fit_least_squares(bvalues, dw_data, S0_output=False, fitS0=True,
                      bounds=([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3])):
    """
    This is an implementation of the conventional IVIM fit. It fits a single curve
    :param bvalues: Array with the b-values
    :param dw_data: Array with diffusion-weighted signal at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; default = True
    :param fix_S0: Boolean determining whether to fix S0 to 1; default = False
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]).
        Default: ([0.005, 0, 0, 0.8], [0.3, 0.7, 0.005, 1.2])
    :return Dt: Array with D in each voxel
    :return Fp: Array with f in each voxel
    :return Dp: Array with Dp in each voxel
    :return S0: Array with S0 in each voxel
    """
    try:
        if not fitS0:
            # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
            bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 10],
                      [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 10])
            params, _ = curve_fit(ivimN_noS0, bvalues, dw_data, p0=[1, 1, 0.1], bounds=bounds)
            S0 = 1
        else:
            # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
            bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 10, bounds[0][3]],
                      [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 10, bounds[1][3]])
            params, _ = curve_fit(ivimN, bvalues, dw_data, p0=[1, 1, 0.1, 1], bounds=bounds)
            S0 = params[3]
        # correct for the rescaling of parameters
        Dt, Fp, Dp = params[0] / 1000, params[1] / 10, params[2] / 10
        # reorder output in case Dp<Dt
        if S0_output:
            return order(Dt, Fp, Dp, S0)
        else:
            return order(Dt, Fp, Dp)
    except:
        # if fit fails, then do a segmented fit instead
        # print('lsq fit failed, trying segmented')
        if S0_output:
            Dt, Fp, Dp = fit_segmented(bvalues, dw_data, bounds=bounds)
            return Dt, Fp, Dp, 1
        else:
            return fit_segmented(bvalues, dw_data)
