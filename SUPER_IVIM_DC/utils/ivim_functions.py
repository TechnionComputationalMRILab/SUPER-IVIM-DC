import numpy as np


def ivim(b_values, Dt, Fp, Dp, S0):
    """
    regular IVIM function
    """
    return S0 * (Fp * np.exp(-b_values * Dp) + (1 - Fp) * np.exp(-b_values * Dt))


def ivimN_lsq(bvalues, Dt, Fp, Dp, S0):
    """
    IVIM function
    in which we try to have equal variance in the different IVIM parameters;
    equal variance helps with certain fitting algorithms
    """
    return S0 * (Fp / 10 * np.exp(-bvalues * Dp / 10) + (1 - Fp / 10) * np.exp(-bvalues * Dt / 1000))


def ivimN_noS0_lsq(bvalues, Dt, Fp, Dp):
    # IVIM function in which we try to have equal variance in the different IVIM parameters and S0=1
    return (Fp * np.exp(-bvalues * Dp ) + (1 - Fp) * np.exp(-bvalues * Dt))



def ivimN(bvalues, Dt, Fp, Dp, S0):
    """ Compatibility with fitting_algorithms.py """
    return ivimN_lsq(bvalues, Dt, Fp, Dp, S0)



def ivimN_noS0(bvalues, Dt, Fp, Dp):
    """ Compatibility with fitting_algorithms.py """
    # IVIM function in which we try to have equal variance in the different IVIM parameters and S0=1
    return (Fp / 10 * np.exp(-bvalues * Dp / 10) + (1 - Fp / 10) * np.exp(-bvalues * Dt / 1000))

