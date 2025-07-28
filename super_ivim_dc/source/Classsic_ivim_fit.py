# -*- coding: utf-8 -*-
"""
ivim_module.py

Module for IVIM (Intra-Voxel Incoherent Motion) signal simulation and fitting.

Author: Yael Zaffrani (refactored)
Created: 2022-01-26

This module provides:
  - IVIM signal model functions
  - Parameter normalization helpers
  - Nonlinear least squares fitting routines (LM, TRF, BOBYQA)
  - Slice-by-slice signal fitting pipelines (SLS, SLS_LM, SLS_TRF, SLS_BOBYQA)
  - Utility for plotting fitted vs. original signals
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from warnings import warn

# Global scaling factors for parameter normalization
D_factor = 0.1
s0_factor = np.sqrt(1000)

__all__ = [
    "IVIM_model",
    "ivimN",
    "parmNRMSE",
    "ivim_fit_nlls_error",
    "fit_least_squares_lm",
    "fit_least_squares_trf",
    "fit_least_squares_BOBYQA",
    "IVIM_fit_sls",
    "fitMonoExpModel",
    "IVIM_fit_sls_lm",
    "IVIM_fit_sls_trf",
    "IVIM_fit_sls_BOBYQA",
    "plot_signal_fit",
]


def IVIM_model(b_vector: np.ndarray, D: float, DStar: float, f: float, s0: float) -> np.ndarray:
    """
    Basic IVIM signal model.

    SI(b) = s0 * (f * exp(-b * DStar) + (1 - f) * exp(-b * D))

    Parameters:
        b_vector (np.ndarray): Array of b-values.
        D (float): Diffusion coefficient.
        DStar (float): Pseudo-diffusion coefficient.
        f (float): Perfusion fraction.
        s0 (float): Signal at b=0.

    Returns:
        np.ndarray: Simulated signal intensities.
    """
    return s0 * (f * np.exp(-b_vector * DStar) + (1 - f) * np.exp(-b_vector * D))


def ivimN(b_vector: np.ndarray, D: float, DStar: float, f: float, s0: float) -> np.ndarray:
    """
    Normalized IVIM model with variance scaling.

    Applies D_factor and s0_factor to normalize parameters during fitting.

    Parameters:
        b_vector (np.ndarray): Array of b-values.
        D (float): Scaled diffusion coefficient.
        DStar (float): Pseudo-diffusion coefficient.
        f (float): Perfusion fraction.
        s0 (float): Scaled initial signal.

    Returns:
        np.ndarray: Simulated signal intensities.
    """
    D_norm = D / D_factor
    s0_norm = s0 / s0_factor
    return s0_norm * (f * np.exp(-b_vector * (D_norm + DStar))
                      + (1 - f) * np.exp(-b_vector * D_norm))


def parmNRMSE(org_param: np.ndarray, fit_param: np.ndarray,
              del_index: list, dim: int = 0) -> float:
    """
    Compute normalized root mean squared error (NRMSE) between original
    and fitted parameters, excluding indices in del_index.

    Parameters:
        org_param (np.ndarray): Original parameter array.
        fit_param (np.ndarray): Fitted parameter array.
        del_index (list): Indices of removed samples.
        dim (int): Axis along which to delete.

    Returns:
        float: Normalized RMSE.
    """
    org_clean = np.delete(org_param, del_index, axis=dim)
    mse = np.linalg.norm(org_clean - fit_param) / np.sqrt(len(org_clean))
    return mse / np.mean(org_clean)


def ivim_fit_nlls_error(xData: np.ndarray, gradData, b_vector: np.ndarray,
                        si: np.ndarray) -> float:
    """
    Negative log-likelihood error for IVIM nonlinear least squares.

    Used by BOBYQA optimizer.

    Parameters:
        xData (np.ndarray): Parameter vector [D, DStar, f, s0].
        gradData: Gradient (unused).
        b_vector (np.ndarray): Array of b-values.
        si (np.ndarray): Observed signal.

    Returns:
        float: Mean squared error between observed and fitted signal.
    """
    D, DStar, f, s0_h = xData
    si_fit = ivimN(b_vector, D, DStar, f, s0_h)
    return np.mean((si - si_fit) ** 2)


def fit_least_squares_lm(b_vector: np.ndarray, si: np.ndarray,
                         p0: list) -> tuple:
    """
    Fit IVIM parameters using Levenberg-Marquardt algorithm.

    Parameters:
        b_vector (np.ndarray): Array of b-values.
        si (np.ndarray): Signal matrix of shape (len(b_vector), n_samples).
        p0 (list): Initial guess [D, DStar, f, s0].

    Returns:
        tuple: Arrays (D, DStar, f, s0) and list of failed indices.
    """
    p0[0] *= D_factor
    p0[3] *= s0_factor
    del_index = []
    D = np.array([])
    DStar = np.array([])
    f = np.array([])
    s0 = np.array([])

    for i in range(si.shape[1]):
        s = np.squeeze(si[:, i])
        try:
            params, _ = curve_fit(ivimN, b_vector, s, p0, maxfev=2000)
            D = np.append(D, params[0] / D_factor)
            DStar = np.append(DStar, params[1])
            f = np.append(f, params[2])
            s0 = np.append(s0, params[3] / s0_factor)
        except Exception:
            del_index.append(i)

    return D, DStar, f, s0, del_index


def fit_least_squares_trf(b_vector: np.ndarray, si: np.ndarray,
                          bounds: list, p0: list) -> tuple:
    """
    Fit IVIM parameters using Trust-Region Reflective algorithm.

    Parameters:
        b_vector (np.ndarray): Array of b-values.
        si (np.ndarray): Signal matrix of shape (len(b_vector), n_samples).
        bounds (list): [[min_params], [max_params]].
        p0 (list): Initial guess [D, DStar, f, s0].

    Returns:
        tuple: Arrays (D, DStar, f, s0) and list of failed indices.
    """
    p0[0] *= D_factor
    p0[3] *= s0_factor
    bounds_scaled = ([bounds[0][0] * D_factor, bounds[0][1], bounds[0][2], bounds[0][3] * s0_factor],
                     [bounds[1][0] * D_factor, bounds[1][1], bounds[1][2], bounds[1][3] * s0_factor])
    del_index = []
    D = np.array([])
    DStar = np.array([])
    f = np.array([])
    s0 = np.array([])

    for i in range(si.shape[1]):
        s = np.squeeze(si[:, i])
        try:
            params, _ = curve_fit(ivimN, b_vector, s, p0,
                                  bounds=bounds_scaled, maxfev=30000)
            D = np.append(D, params[0] / D_factor)
            DStar = np.append(DStar, params[1])
            f = np.append(f, params[2])
            s0 = np.append(s0, params[3] / s0_factor)
        except Exception:
            del_index.append(i)

    return D, DStar, f, s0, del_index


def fit_least_squares_BOBYQA(b_vector: np.ndarray, si: np.ndarray,
                             bounds: list, p0: list) -> tuple:
    """
    Fit IVIM parameters using BOBYQA optimizer via nlopt.

    Parameters:
        b_vector (np.ndarray): Array of b-values.
        si (np.ndarray): Signal matrix of shape (len(b_vector), n_samples).
        bounds (list): [[min_params], [max_params]].
        p0 (list): Initial guess [D, DStar, f, s0].

    Returns:
        tuple: Arrays (D, DStar, f, s0) and list of failed indices.
    """
    import nlopt

    # Scale initial guess and bounds
    p0[0] *= D_factor
    p0[3] *= s0_factor
    lb = np.asarray([bounds[0][0] * D_factor, bounds[0][1], bounds[0][2], bounds[0][3] * s0_factor])
    ub = np.asarray([bounds[1][0] * D_factor, bounds[1][1], bounds[1][2], bounds[1][3] * s0_factor])

    opt = nlopt.opt(nlopt.LN_BOBYQA, 4)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(2000)
    opt.set_ftol_abs(1e-5)

    del_index = []
    D = np.array([])
    DStar = np.array([])
    f = np.array([])
    s0 = np.array([])

    for i in range(si.shape[1]):
        s = np.squeeze(si[:, i])
        try:
            opt.set_min_objective(
                lambda x, grad: ivim_fit_nlls_error(x, grad, b_vector, s)
            )
            xopt = opt.optimize(p0)
            D = np.append(D, xopt[0] / D_factor)
            DStar = np.append(DStar, xopt[1])
            f = np.append(f, xopt[2])
            s0 = np.append(s0, xopt[3] / s0_factor)
        except Exception:
            del_index.append(i)

    return D, DStar, f, s0, del_index


def IVIM_fit_sls(si: np.ndarray, b_vector: np.ndarray,
                 bounds: list, min_bval_high: float = 200) -> tuple:
    """
    Slice-by-slice sequential linear fitting (SLS) for IVIM.

    Steps:
      1. Fit monoexponential decay at high b-values to estimate D and s0_d.
      2. Compute perfusion fraction f.
      3. Fit DStar via NLLS at all b-values.

    Parameters:
        si (np.ndarray): Signal matrix (len(b_vector) x n_samples).
        b_vector (np.ndarray): Array of b-values.
        bounds (list): [[min_params], [max_params]].
        min_bval_high (float): Threshold b-value for high-range fit.

    Returns:
        tuple: Arrays (D, DStar, f, s0), s0_d, and list of failed indices.
    """
    # High b-values for monoexponential fit
    mask_high = b_vector >= min_bval_high
    s_high = si[mask_high, :]
    b_high = b_vector[mask_high]

    # Monoexponential fit: s0_d and D
    s0_d, D = fitMonoExpModel(s_high, b_high)
    s0 = si[0, :]

    # Perfusion fraction
    f = (s0 - s0_d) / s0

    # DStar fit via NLLS
    bounds_Ds = (bounds[0][1], bounds[1][1])
    p0_Ds = float(np.mean(bounds_Ds))
    del_index = []
    DStar = np.array([])

    for i in range(si.shape[1]):
        s = np.squeeze(si[:, i])
        try:
            params, _ = curve_fit(
                lambda b, DStar: s0[i] * (f[i] * np.exp(-b * (D[i] + DStar))
                                           + (1 - f[i]) * np.exp(-b * D[i])),
                b_vector, s, p0=p0_Ds, bounds=bounds_Ds, maxfev=1000
            )
            DStar = np.append(DStar, params[0])
        except Exception:
            del_index.append(i)

    return D, DStar, f, s0, s0_d, del_index


def fitMonoExpModel(s: np.ndarray, b_vector: np.ndarray) -> tuple:
    """
    Fit a monoexponential decay (ln-linear) to data.

    Model: ln(s) = ln(s0) - ADC * b

    Parameters:
        s (np.ndarray): Signal matrix (len(b_vector) x n_samples).
        b_vector (np.ndarray): Array of b-values.

    Returns:
        tuple: s0 and ADC estimates.
    """
    A = np.column_stack((np.ones(len(b_vector)), -b_vector))
    log_s = np.log(s)
    log_s[np.isinf(log_s)] = 0.0
    x, *_ = np.linalg.lstsq(A, log_s, rcond=None)
    s0 = np.exp(x[0])
    ADC = x[1]
    return s0, ADC


def IVIM_fit_sls_lm(si: np.ndarray, b_vector: np.ndarray,
                    bounds: list, min_bval_high: float = 200) -> tuple:
    """
    Sequential SLS pipeline followed by LM refinement.

    Uses IVIM_fit_sls initial guess then calls fit_least_squares_lm.
    """
    D_sls, DStar_sls, f_sls, s0_sls, s0_d, del_idx = IVIM_fit_sls(
        si, b_vector, bounds, min_bval_high
    )
    si_clean = np.delete(si, del_idx, axis=1)
    D, DStar, f, s0 = np.array([]), np.array([]), np.array([]), np.array([])
    failures = []

    for i in range(si_clean.shape[1]):
        p0 = [D_sls[i], DStar_sls[i], f_sls[i], s0_sls[i]]
        try:
            D_fit, DS_fit, f_fit, s0_fit, del2 = fit_least_squares_lm(
                b_vector, si_clean[:, i, None], p0
            )
            if del2:
                failures.append(i)
            D = np.append(D, D_fit)
            DStar = np.append(DStar, DS_fit)
            f = np.append(f, f_fit)
            s0 = np.append(s0, s0_fit)
        except Exception:
            failures.append(i)

    return D, DStar, f, s0, s0_d, failures


def IVIM_fit_sls_trf(si: np.ndarray, b_vector: np.ndarray, bounds: list,
                      eps: float = 1e-5, min_bval_high: float = 200) -> tuple:
    """
    Sequential SLS pipeline followed by TRF refinement.

    Ensures initial guesses lie within bounds.
    """
    D_sls, DStar_sls, f_sls, s0_sls, s0_d, del_idx = IVIM_fit_sls(
        si, b_vector, bounds, min_bval_high
    )
    si_clean = np.delete(si, del_idx, axis=1)
    D, DStar, f, s0 = np.array([]), np.array([]), np.array([]), np.array([])
    failures = []

    for i in range(si_clean.shape[1]):
        p0 = [D_sls[i], DStar_sls[i], f_sls[i], s0_sls[i]]
        # adjust out-of-bounds p0
        bounds_arr = np.array(bounds)
        for j, val in enumerate(p0):
            if val < bounds_arr[0, j]:
                p0[j] = bounds_arr[0, j] * (1 + eps)
            elif val > bounds_arr[1, j]:
                p0[j] = bounds_arr[1, j] * (1 - eps)
        try:
            D_fit, DS_fit, f_fit, s0_fit, del2 = fit_least_squares_trf(
                b_vector, si_clean[:, i, None], bounds, p0
            )
            if del2:
                failures.append(i)
            D = np.append(D, D_fit)
            DStar = np.append(DStar, DS_fit)
            f = np.append(f, f_fit)
            s0 = np.append(s0, s0_fit)
        except Exception:
            failures.append(i)

    return D, DStar, f, s0, s0_d, failures


def IVIM_fit_sls_BOBYQA(si: np.ndarray, b_vector: np.ndarray, bounds: list,
                         eps: float = 1e-5, min_bval_high: float = 200) -> tuple:
    """
    Sequential SLS pipeline followed by BOBYQA refinement.
    """
    D_sls, DStar_sls, f_sls, s0_sls, s0_d, del_idx = IVIM_fit_sls(
        si, b_vector, bounds, min_bval_high
    )
    si_clean = np.delete(si, del_idx, axis=1)
    D, DStar, f, s0 = np.array([]), np.array([]), np.array([]), np.array([])
    failures = []

    for i in range(si_clean.shape[1]):
        p0 = [D_sls[i], DStar_sls[i], f_sls[i], s0_sls[i]]
        bounds_arr = np.array(bounds)
        for j, val in enumerate(p0):
            if val < bounds_arr[0, j]:
                p0[j] = bounds_arr[0, j] * (1 + eps)
            elif val > bounds_arr[1, j]:
                p0[j] = bounds_arr[1, j] * (1 - eps)
        try:
            D_fit, DS_fit, f_fit, s0_fit, del2 = fit_least_squares_BOBYQA(
                b_vector, si_clean[:, i, None], bounds, p0
            )
            if del2:
                failures.append(i)
            D = np.append(D, D_fit)
            DStar = np.append(DStar, DS_fit)
            f = np.append(f, f_fit)
            s0 = np.append(s0, s0_fit)
        except Exception:
            failures.append(i)

    return D, DStar, f, s0, s0_d, failures


def plot_signal_fit(b_vector: np.ndarray, si: np.ndarray,
                    si_fit: np.ndarray, si_original: np.ndarray,
                    s0: float, D: float, DStar: float, f: float) -> None:
    """
    Plot log-normalized IVIM signal vs. b-values.

    Displays ground truth, noisy, and fitted curves.
    """
    log_org = np.log(si_original / si_original[0])
    log_noisy = np.log(si / si[0])
    log_fit = np.log(si_fit / si_fit[0])

    fig, ax = plt.subplots()
    ax.plot(b_vector, log_org, marker='*', label='GT')
    ax.plot(b_vector, log_noisy, marker='*', label='Si_noisy')
    ax.plot(b_vector, log_fit, marker='*', label='fit')

    ax.set_title(f'IVIM Signal: D={D:.3f}, D*={DStar:.2f}, f={f:.2f}, s0={s0:.2f}')
    ax.set_xlabel('b-value')
    ax.set_ylabel('log(s_i/s_0)')
    ax.legend()
    plt.show()