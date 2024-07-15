"""
Author: Ran Liu <rliu20@stanford.edu>

Implementation of phase-amplitude coupling estimation methods
"""

import numpy as np
import pathlib
from importlib.util import find_spec
from somata.basic_models import StateSpaceModel
from somata.exact_inference import inverse
from numpy import ndarray
from cmdstanpy import CmdStanModel
from scipy.optimize import minimize, Bounds
from typing import List
import logging


def pac_anchor_func():
    """Anchor function for locating this .py file"""
    return


def get_pac_path():
    """Find the path to the directory containing the pac_model.py file"""
    f_path = pathlib.Path(pac_anchor_func.__code__.co_filename)
    if f_path.is_file():
        return f_path.parents[0].resolve()
    else:
        somata_path = pathlib.Path(
            find_spec("somata").origin
        )  # find the package root __init__.py filepath
        return (somata_path.parents[0] / "pac").resolve()  # 1-level up


def fit_pac_regression(phase: ndarray, amplitude: ndarray, verbose=True, **kwargs):
    """
    Fit a constrained regression between phase and amplitude according to Soulat et al. 2022.
    Note that this implementation fixes an error in the manuscript that incorrectly
    uses a truncated multivariate t-distribution as the likelihood for numerical search.

    :param phase: Numpy array containing phase vector, of shape (n,).
    :param amplitude: Numpy array containing amplitude vector of shape (n,).
    :param verbose: Whether to print CmdStanPy output to stdout.
    :param **kwargs: Additional parameters to pass to model.sample() for Stan model fitting.

    :return: Draws from posterior distribution of parameters (beta0, beta1, beta2), given observed data.
    """
    model_data = {"n": len(phase), "phase": phase, "amplitude": amplitude}
    model = CmdStanModel(stan_file=get_pac_path() / "stan/pac.stan")

    if not verbose:
        logger = logging.getLogger('cmdstanpy')
        cmdstanpy_logger_status = logger.disabled
        logger.disabled = True

    fit = model.sample(model_data, show_progress=verbose, **kwargs)

    if not verbose:
        logger.disabled = cmdstanpy_logger_status

    # Empirically, these draws seem to follow a multivariate t-distribution;
    # therefore, posterior mean is a reasonable estimator of posterior mode.
    return fit.draws_pd()[["beta0", "beta1", "beta2"]].to_numpy()


def kmod(beta0, beta1, beta2):
    """
    Computes modulation index, given fitted coefficients, according to Soulat et al. 2022.

    :param beta0: Regression intercept.
    :param beta1: Regression coefficient for cos(phase).
    :param beta2: Regression coefficient for sin(phase).

    :return: Computed modulation index.
    """
    return np.sqrt(beta1 ** 2 + beta2 ** 2) / beta0


def phimod(beta1, beta2):
    """
    Computes mean modulation phase, given fitted coefficients, according to Soulat et al. 2022.

    :param beta1: Regression coefficient for cos(phase).
    :param beta2: Regression coefficient for sin(phase).

    :return: Computed mean modulation phase.
    """
    return np.arctan2(beta2, beta1)


def autocovariances(y: ndarray, p: int, rowvar=True, **kwargs):
    """
    Computes autocovariance matrices for lags from 0 to p (inclusive).

    :param y: An ndarray of n observations of d variables. If rowvars is true, expects shape (d, n); otherwise (n, d).
    :param p: Maximum lag for which to compute autocovariance.
    :param rowvar: Whether observations or variables are in first dimension. Defaults to true.
    :param **kwargs: Additional arguments to pass to numpy.cov.

    :return: A list of length p + 1 of autocovariance matrices of shape (d, d), where index i corresponds to a lag of i.
    """
    if rowvar:
        q, n = y.shape
        return [
            np.cov(y[:, :(n - i)], y[:, i:n], **kwargs)[:q, q:] for i in range(p + 1)
        ]
    else:
        n, q = y.shape
        return [
            np.cov(y[:(n - i), :], y[i:n, :], rowvar=False, **kwargs)[:q, q:]
            for i in range(p + 1)
        ]


def block_toeplitz(acov: List[ndarray]):
    """
    Constructs block-toeplitz matrix from autocovariance matrices.

    :param acov: A list of autocovariance matrices of shape (d,d). Index i corresponds to a lag of i.

    :return: A block-toeplitz matrix of shape (nxd, nxd), where the i,jth dxd block is C_{|i-j|}, and
    upper triangular blocks are transposed.
    """
    k = len(acov)  # maximum lag + 1
    q = acov[0].shape[0]
    result = np.zeros((k * q, k * q), dtype=np.float64)

    for i in range(k):
        for j in range(i + 1):  # fill diagonal and lower triangular blocks
            result[(i * q):((i + 1) * q), (j * q):((j + 1) * q)] = acov[i-j]
        for j in range(i + 1, k):  # transpose upper triangular blocks
            result[(i * q):((i + 1) * q), (j * q):((j + 1) * q)] = acov[j-i].T

    return result


def ar_parameters(acov: List[ndarray], R: float):
    """
    Compute AR parameters from autocovariance sequence and candidate observation noise variance.

    :param acov: A list of length p + 1 of autocovariance matrices of shape (d,d). Index i corresponds to a lag of i.
    :param R: Candidate observation noise variance, assumed to be the same aross all dimensions.

    :return: A tuple of a list of length p AR parameter matrices {A_i} such that
    X_t = \\sum_{k=1}^{p} A_k @ X_{t-k} + U_t, U_t ~ N(0, Q), and the state covariance matrix Q.
    """
    p = len(acov) - 1
    q = acov[0].shape[0]

    C_yule = block_toeplitz(acov)[:-q, :-q] - R * np.eye(p * q, dtype=np.float64)
    A_t = inverse(C_yule, approach='svd') @ np.concatenate(acov[1:], axis=0)
    # Compute the transpose of each block
    A = [A_t[(i * q):((i + 1) * q), :].T for i in range(p)]

    # Compute Q from C_0 = \\sum_{k=1}^{p} A_k @ C_k + Q + R (R here is a diagonal matrix)
    Q = (
        acov[0]  # C_0: autocovariance matrix at lag 0
        - R * np.eye(q, dtype=np.float64)  # diagonal observation noise covariance matrix
        - np.sum(
            np.concatenate(
                [np.expand_dims(A[i] @ acov[i + 1], axis=-1) for i in range(p)],
                axis=-1,
            ),
            axis=-1,
        )  # \\sum_{k=1}^{p} A_k @ C_k
    )

    return A, Q


def mvar_ssm(y: ndarray, A: List[ndarray], Q: ndarray, R: float):
    """
    Initializes StateSpaceModel using multivariate AR(p) parameters.

    :param y: Array of n observations of d variables, of shape (d, n).
    :param A: List of AR coefficients of length p, each element an ndarray of shape (d, d).
    :param Q: State noise covariance matrix of shape (d, d).
    :param R: Candidate observation noise variance, assumed to be the same aross all dimensions.

    :return: StateSpaceModel equivalent to specified multivariate AR model.
    """

    p = len(A)  # Order of AR model
    q = A[0].shape[0]  # Dimensionality of observations

    # Compute F (state transition matrix) from matrix coefficients in the list A
    F = np.zeros((p * q, p * q), dtype=np.float64)
    F[:q, :] = np.concatenate(A, axis=1)
    for i in range(1, p):
        F[(i * q):((i + 1) * q), ((i - 1) * q):(i * q)] = np.eye(q, dtype=np.float64)

    # Augment Q to time-lagged hidden states
    Q_new = np.zeros((p * q, p * q), dtype=np.float64)
    Q_new[:q, :q] = Q

    # Compute mu0
    mu0 = np.zeros(p * q, dtype=np.float64)

    # Compute S0 using the diagonal of Q
    S0 = np.diag(np.tile(np.diag(Q), p))

    # Compute G (observation matrix on augmented hidden states)
    G = np.zeros((q, p * q), dtype=np.float64)
    G[:q, :q] = np.eye(q, dtype=np.float64)

    # Convert R
    R_new = R * np.eye(q, dtype=np.float64)

    return StateSpaceModel(F=F, Q=Q_new, mu0=mu0, S0=S0, G=G, R=R_new, y=y)


def optimize_arp(y: ndarray, p: int, epsilon: float = 1e-7):
    """
    For a set of observations, compute the optimal observation noise variance and resulting AR(p) parameters.

    :param y: Array of n observations of d varibles, of shape (d, n).
    :param p: Order of AR model.
    :param epsilon: Epsilon to prevent singular matrices, as scipy optimizes over closed intervals, defaults to 1e-7.

    :return: AR coefficients A, state noise covariance Q, and optimal observation noise variance R.
    """
    acov = autocovariances(y, p=p, bias=True)
    C = block_toeplitz(acov)

    upper = np.min(np.linalg.eigvals(C))

    def objective_function(R):
        A, Q = ar_parameters(acov, R=R)
        model = mvar_ssm(y, A, Q, R)
        return -model.kalman_filt_smooth(return_dict=True)["logL"].sum()

    R = minimize(
        objective_function, x0=upper / 2, bounds=Bounds(lb=epsilon, ub=upper - epsilon)
        ).x[0]

    A, Q = ar_parameters(acov, R=R)

    return A, Q, R
