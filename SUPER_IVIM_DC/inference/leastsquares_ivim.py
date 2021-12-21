from scipy.optimize import curve_fit
import numpy as np

from SUPER_IVIM_DC.utils.ivim_functions import ivimN_lsq
from SUPER_IVIM_DC.utils import isnan
from SUPER_IVIM_DC.utils.checkarg import checkarg


def infer_leastsquares_IVIM(X_infer, labels, bvalues, arg, verbose=1):
    arg = checkarg(arg)
    n_samples = len(X_infer)

    print(f'The number of samples are: {n_samples} \n' if verbose > 0 else "", end='')

    ## normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_infer[:, bvalues == 100], axis=1).astype('<f')
        X_infer = X_infer / S0[:, None]
        nan_idx = isnan(np.mean(X_infer, axis=1))
        X_infer = np.delete(X_infer, nan_idx, axis=0)
        labels = np.delete(labels, nan_idx, axis=0)  # Dt, f, Dp

        print(f'phantom lsq \n' if verbose > 0 else "", end='')
    else:
        S0 = np.mean(X_infer[:, bvalues == 0], axis=1).astype('<f')
        X_infer = X_infer / S0[:, None]
        nan_idx = isnan(np.mean(X_infer, axis=1))
        X_infer = np.delete(X_infer, nan_idx, axis=0)
        labels = np.delete(labels, nan_idx, axis=0)  # Dt, f, Dp

    # Limiting the percentile threshold
    if arg.key == 'phantom':
        b_less_500_idx = np.percentile(X_infer[:, bvalues < 500], 95,
                                       axis=1) < 1.3
        b_greater_500_idx = np.percentile(X_infer[:, bvalues > 500], 95,
                                          axis=1) < 1.2
        b_greater_1000_idx = np.percentile(X_infer[:, bvalues > 1000], 95,
                                           axis=1) < 1
        thresh_idx = b_less_500_idx & b_greater_500_idx & b_greater_1000_idx
    else:
        b_less_50_idx = np.percentile(X_infer[:, bvalues < 50], 95,
                                      axis=1) < 1.3
        b_greater_50_idx = np.percentile(X_infer[:, bvalues > 50], 95,
                                         axis=1) < 1.2
        b_greater_150_idx = np.percentile(X_infer[:, bvalues > 150], 95,
                                          axis=1) < 1
        thresh_idx = b_less_50_idx & b_greater_50_idx & b_greater_150_idx

    suprevised_data = np.append(X_infer[thresh_idx, ], labels[thresh_idx, ], axis=1)

    # initialise parameters and data
    Dp_lsq_NRMSE = np.array([])
    Dt_lsq_NRMSE = np.array([])
    Fp_lsq_NRMSE = np.array([])
    S0_lsq_NRMSE = np.array([])
    # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
    bound_flag = True

    # calculate least square
    for dw_data, dw_label in zip(X_infer[thresh_idx,], labels[thresh_idx,]):
        if bound_flag:
            bounds = ([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3])

            # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
            bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 10, bounds[0][3]],
                      [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 10, bounds[1][3]])

            params, _ = curve_fit(ivimN_lsq, bvalues, dw_data, p0=[1, 1, 0.1, 1], bounds=bounds)
            S0_lsq = params[3]

            # correct for the rescaling of parameters
            Dt_lsq, Fp_lsq, Dp_lsq = params[0] / 1000, params[1] / 10, params[2] / 10
        else:
            params, _ = curve_fit(ivimN_lsq, bvalues, dw_data, p0=[1, 1, 0.1, 1], maxfev=5000)  # TODO: find the error?
            S0_lsq = params[3]
            # correct for the rescaling of parameters
            Dt_lsq, Fp_lsq, Dp_lsq = params[0], params[1], params[2]

        Dt_orig = dw_label[0]
        Fp_orig = dw_label[1]
        Dp_orig = dw_label[2]

        # calculate normelized mean square error
        Dp_norm_error = np.sqrt(np.square(Dp_orig - Dp_lsq)) / Dp_orig
        Dt_norm_error = np.sqrt(np.square(Dt_orig - Dt_lsq)) / Dt_orig
        Fp_norm_error = np.sqrt(np.square(Fp_orig - Fp_lsq)) / Fp_orig
        S0_norm_error = np.sqrt(np.square(1 - S0_lsq))

        # appened all reaults
        Dp_lsq_NRMSE = np.append(Dp_lsq_NRMSE, Dp_norm_error)
        Dt_lsq_NRMSE = np.append(Dt_lsq_NRMSE, Dt_norm_error)
        Fp_lsq_NRMSE = np.append(Fp_lsq_NRMSE, Fp_norm_error)
        S0_lsq_NRMSE = np.append(S0_lsq_NRMSE, S0_norm_error)

    return [Dp_lsq_NRMSE, Dt_lsq_NRMSE, Fp_lsq_NRMSE, S0_lsq_NRMSE]
