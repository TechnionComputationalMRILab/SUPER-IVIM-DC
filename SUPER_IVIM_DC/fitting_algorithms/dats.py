import numpy as np

from .segmented import fit_segmented_array
from .least_squares import fit_least_squares_array
from .bayesian import fit_bayesian_array
from SUPER_IVIM_DC.utils.checkarg import checkarg_lsq


def fit_dats(bvalues, dw_data, arg, savename=None):
    """
    Wrapper function that selects the right fit depending on what fit is selected in arg.method.
    input:
    :param arg: an object with fit options, with attributes:
    arg.method --> string with the fit method;
        allowed: lsq (least squares fit), seg (segmented fit) and bayes (bayesian fit)
    arg.do_fit --> Boolean; False for skipping the regular fit
    arg.load_lsq --> Boolean; True will load the fit results saved under input parameter "savename"
    arg.fitS0 --> Boolean; False fixes S0 to 1, True fits S0
    arg.jobs --> Integer specifying the number of parallel processes used in fitting.
        If <2, regular fitting is used instead
    arg.bounds --> 2D Array of fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max])
    :param bvalues: 1D Array of b-values used
    :param dw_data: 2D Array containing the dw_data used with dimensions voxels x b-values
    optional:
    :param savename: String with the save name

    :return paramslsq: 2D array containing the fit parameters D, f (Fp), D* (Dp) and, optionally, S0, for each voxel
    """
    # Checking completeness of arg and adding missing values as defaults
    arg = checkarg_lsq(arg)
    if arg.do_fit:
        if not arg.load_lsq:
            # select fit to be run
            if arg.method == 'seg':
                print('running segmented fit\n')
                paramslsq = fit_segmented_array(bvalues, dw_data, njobs=arg.jobs, bounds=arg.bounds)
                # save results if parameter savename is given
                if savename is not None:
                    np.savez(savename, paramslsq=paramslsq)
            elif arg.method == 'lsq':
                print('running conventional fit\n')
                paramslsq = fit_least_squares_array(bvalues, dw_data, S0_output=True, fitS0=arg.fitS0, njobs=arg.jobs,
                                                    bounds=arg.bounds)
                # save results if parameter savename is given
                if savename is not None:
                    np.savez(savename, paramslsq=paramslsq)
            elif arg.method == 'bayes':
                print('running conventional fit to determine Bayesian prior\n')
                # for this Bayesian fit approach, a data-driven prior needs to be defined.
                # Hence, intially we do a regular lsq fit
                paramslsq = fit_least_squares_array(bvalues, dw_data, S0_output=True, fitS0=arg.fitS0, njobs=arg.jobs,
                                                    bounds=arg.bounds)
                print('running Bayesian fit\n')
                Dt_pred, Fp_pred, Dp_pred, S0_pred = fit_bayesian_array(bvalues, dw_data, paramslsq, arg)
                Dt0, Fp0, Dp0, S00 = paramslsq
                # For Bayesian fit, we also give the lsq results as we had to obtain them anyway.
                if arg.fitS0:
                    paramslsq = Dt_pred, Fp_pred, Dp_pred, S0_pred, Dt0, Fp0, Dp0, S00
                else:
                    paramslsq = Dt_pred, Fp_pred, Dp_pred, Dt0, Fp0, Dp0
                if savename is not None:
                    # save results if parameter savename is given
                    np.savez(savename, paramslsq=paramslsq)
            else:
                raise Exception('the choise lsq-fit is not implemented. Try ''lsq'', ''seg'' or ''bayes''')
        else:
            # if we chose to load the fit
            print('loading fit\n')
            loads = np.load(savename)
            paramslsq = loads['paramslsq']
            del loads
        return paramslsq
    # if fit is skipped, we return nothing
    return None
