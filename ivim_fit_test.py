# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 09:58:17 2025

@author: moti.freiman
"""

"""
Unit tests + script for IVIM fitting routines from Classic_ivim_fit.py.

- Generates synthetic IVIM signals within given bounds.
- Runs three different fitters and asserts reconstructed log-signals
  are close to the originals.
- If run as a script, executes the tests, prints pass/fail, then
  produces a comparison plot (log SI vs b-values) just like your
  original code.
"""

from super_ivim_dc.source.Classsic_ivim_fit import *
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys

# Parameter bounds: [D, D*, f, S0]
bounds = [
    [0.0003, 0.009, 0.3, 50],
    [0.01,   0.04,  0.5, 300],
]

# b-values for IVIM model
b_vector = np.array([0, 25, 50, 75, 100, 200, 400, 800])


def generate_si(b_vector=b_vector):
    """
    Generate one synthetic IVIM signal.

    Samples (D, DStar, f, s0) uniformly within `bounds`,
    computes the IVIM_model at each b, and returns:
      - si: 1D np.array of signal values
      - p0: list of the true parameters [D, DStar, f, s0]
    """
    D      = uniform(bounds[0][0], bounds[1][0])
    DStar  = uniform(bounds[0][1], bounds[1][1])
    f      = uniform(bounds[0][2], bounds[1][2])
    s0     = uniform(bounds[0][3], bounds[1][3])
    p0     = [D, DStar, f, s0]
    si     = IVIM_model(b_vector=b_vector, D=D, DStar=DStar, f=f, s0=s0)
    return np.asarray(si), p0


@pytest.fixture(scope="module")
def synthetic_data():
    """
    Fixture that returns:
      - si_array: shape (2, len(b_vector)) synthetic signals (two replicates)
      - idx: row index (0) to test
    """
    si, _ = generate_si()
    si_array = np.vstack([si, si])
    idx = 0
    return si_array, idx


def _assert_signal_close(orig, est, tol=0.1):
    """Assert that log-domain signals match within ±tol."""
    np.testing.assert_allclose(
        np.log(orig), np.log(est),
        atol=tol,
        err_msg="Reconstructed log-signal deviates beyond tolerance"
    )


def test_sls_fitter(synthetic_data):
    """Simple Least-Squares (SLS) fits the IVIM signal."""
    si_array, idx = synthetic_data
    fit = IVIM_fit_sls(si_array.T, b_vector, bounds)
    est = IVIM_model(b_vector, fit[0][idx], fit[1][idx], fit[2][idx], fit[3][idx])
    _assert_signal_close(si_array[idx], est)


def test_sls_lm_fitter(synthetic_data):
    """Levenberg-Marquardt (SLS_LM) fits the IVIM signal."""
    si_array, idx = synthetic_data
    fit = IVIM_fit_sls_lm(si_array.T, b_vector, bounds)
    est = IVIM_model(b_vector, fit[0][idx], fit[1][idx], fit[2][idx], fit[3][idx])
    _assert_signal_close(si_array[idx], est)


def test_sls_trf_fitter(synthetic_data):
    """Trust-Region Reflective (SLS_TRF) fits the IVIM signal."""
    si_array, idx = synthetic_data
    fit = IVIM_fit_sls_trf(si_array.T, b_vector, bounds)
    est = IVIM_model(b_vector, fit[0][idx], fit[1][idx], fit[2][idx], fit[3][idx])
    _assert_signal_close(si_array[idx], est)


if __name__ == "__main__":
    # Run pytest programmatically
    ret = pytest.main([__file__])
    if ret == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ Tests failed (exit code {ret}).")
        sys.exit(ret)

    # After successful tests, recreate signals & fit for plotting
    si_array = np.asarray([generate_si()[0] for _ in range(2)])
    idx = 0

    fit_sls    = IVIM_fit_sls(si_array.T, b_vector, bounds)
    est_sls    = IVIM_model(b_vector,
                             fit_sls[0][idx],
                             fit_sls[1][idx],
                             fit_sls[2][idx],
                             fit_sls[3][idx])

    fit_lm     = IVIM_fit_sls_lm(si_array.T, b_vector, bounds)
    est_lm     = IVIM_model(b_vector,
                             fit_lm[0][idx],
                             fit_lm[1][idx],
                             fit_lm[2][idx],
                             fit_lm[3][idx])

    fit_trf    = IVIM_fit_sls_trf(si_array.T, b_vector, bounds)
    est_trf    = IVIM_model(b_vector,
                             fit_trf[0][idx],
                             fit_trf[1][idx],
                             fit_trf[2][idx],
                             fit_trf[3][idx])

    # Original vs fitted signals (log‐domain)
    plt.figure()
    plt.plot(b_vector, np.log(si_array[idx]),       'b-', label='Ref. signal')
    plt.plot(b_vector, np.log(est_sls),             'r-', label='SLS est.')
    plt.plot(b_vector, np.log(est_lm),              'g-', label='SLS_LM est.')
    plt.plot(b_vector, np.log(est_trf),             'c-', label='SLS-TRF est.')
    plt.xlabel('b-value')
    plt.ylabel('log S(b)')
    plt.legend()
    plt.title('IVIM fit comparison')
    plt.show()