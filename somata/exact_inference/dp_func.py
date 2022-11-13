"""
Author: Mingjian He <mh105@mit.edu>

Dynamic programming (also known as belief propagation) algorithms for exact inference
"""

import numpy as np
from scipy.linalg import solve_triangular


def forward_backward(A, py_x, p1=None, compute_edge=False):
    """
    Classical forward-backward algorithm for discrete state HMM with
    step-wise normalization. Note that this ensures the final conditional
    densities are correct, but the alpha and beta vectors are off by a
    normalization constant

    Reference:
        Rabiner, L., & Juang, B. (1986). An introduction to hidden
        Markov models. ieee assp magazine, 3(1), 4-16.

    Inputs:
    :param A: transition probability matrix
    :param py_x: observation probability
    :param p1: initial prior of hidden state at t=1
    :param compute_edge: whether to compute edge marginals
    """
    # Model dimensions
    K = A.shape[0]
    T = py_x.shape[1]
    assert py_x.shape[0] == K, 'Dimension mismatch between A and py_x.'
    logL = np.zeros(T)

    # Initialize prior at t=1
    p1 = np.ones(K) / K if p1 is None else p1

    # Compute the alpha (forward pass)
    norm_a = np.zeros((K, T))
    norm_a[:, 0] = py_x[:, 0] * p1 / (py_x[:, 0] * p1).sum()  # t=1
    logL[0] = np.log((py_x[:, 0] * p1).sum())
    for ii in range(1, T):  # t=2 -> t=T
        a = py_x[:, ii] * (A @ norm_a[:, ii-1])
        norm_a[:, ii] = a / a.sum()
        logL[ii] = np.log(a.sum())  # one-step predictive log likelihood

    p_h_t_v1_t = norm_a  # filtered hidden state posterior density

    # Compute the beta (backward pass)
    norm_b = np.zeros((K, T))
    norm_b[:, -1] = np.ones(K)  # b(h_T) = 1 for all h_T at t=T
    for ii in range(T-2, -1, -1):  # t=T-1 -> t=1
        b = A.T @ (py_x[:, ii+1] * norm_b[:, ii+1])
        norm_b[:, ii] = b / b.sum()

    # norm_a and norm_b are defined for t=1 -> t=T
    ab = norm_a * norm_b
    p_h_t_v1_T = ab / ab.sum(axis=0)  # smoothed hidden state posterior density

    # Compute pairwise edge marginals using a and b (note that they are normalized)
    if compute_edge:
        p_h_t_tmin1_v1_T = np.zeros((K, K, T))
        for ii in range(1, T):  # t=2 -> t=T
            ab_mat = norm_a[:, ii-1] * py_x[:, ii][:, None] * A * norm_b[:, ii][:, None]
            p_h_t_tmin1_v1_T[:, :, ii] = ab_mat / ab_mat.sum()  # P(h_t-1,h_t|v_1:T)
    else:
        p_h_t_tmin1_v1_T = None

    return p_h_t_v1_t, p_h_t_v1_T, p_h_t_tmin1_v1_T, logL


def baum_welch(A_init, py_x, p1_init, maxEM_iter=100):
    """
    Classical Baum-Welch algorithm on hidden markov model parameter estimation.
    In this implementation, the observation probability (also known as
    emission distribution) is fixed and therefore not estimated. Even though
    py_x has explicit M step update equations and is estimable by a general
    Baum-welch for both discrete and Gaussian observation variable y, it often
    has known relations that determine the observation probability thus fixed

    Reference:
        Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). A maximization
        technique occurring in the statistical analysis of probabilistic
        functions of Markov chains. The annals of mathematical statistics, 41(1),
        164-171.

    Inputs:
    :param A_init: transition probability matrix for the first E step
    :param py_x: observation probability (fixed)
    :param p1_init: initial prior of hidden state at t=1 for the first E step
    :param maxEM_iter: maximal number of EM iterations
    """
    # Model dimensions
    K = A_init.shape[0]
    assert py_x.shape[0] == K, 'Dimension mismatch between A_init and py_x.'
    assert p1_init.shape[0] == K, 'Dimension mismatch between A_init and p1_init.'

    # Initialize the EM loop
    A = A_init  # A_ij = P(h_t = i | h_t-1 = j), i.e. rows >> t+1, columns >> t
    p1 = p1_init
    logL_list = [float('-inf')]
    logL_delta = float('inf')
    EM_iter = 0

    # EM iterations
    while logL_delta > 10**-6 and EM_iter <= maxEM_iter:
        EM_iter = EM_iter + 1

        # E step - taking expectation under the hidden state posterior using current parameters {A, p1}
        _, p_h_t_v1_T, p_h_t_tmin1_v1_T, logL = forward_backward(A=A, py_x=py_x, p1=p1, compute_edge=True)

        # M step - MLE of parameters {A, p1}
        p1 = p_h_t_v1_T[:, 0]  # updated p1 = P(h1|v_1:T)
        edges = p_h_t_tmin1_v1_T.sum(axis=2)  # sum over time points
        A = edges / edges.sum(axis=0)  # updated A = \sum_t{P(h_t-1,h_t|v_1:T)} / \sum_t{P(h_t-1|v_1:T)}

        # Check convergence using logL
        logL = logL.sum()
        logL_delta = abs(logL - logL_list[-1])
        logL_list.append(logL)

    return A, p1, EM_iter, logL_list


def viterbi(A, py_x, p1=None):
    """
    Classical Viterbi algorithm for discrete state HMM to solve argmax exact
    inference. Note that we use the log-probability version to cope with
    underflow numerical precision, therefore this is an instance of the
    general Max-Sum algorithm specialized to the tree graphical model of HMM

    Reference:
        Viterbi, A. (1967). Error bounds for convolutional codes and an
        asymptotically optimum decoding algorithm. IEEE transactions on
        Information Theory, 13(2), 260-269.

    Inputs:
    :param A: transition probability matrix
    :param py_x: observation probability
    :param p1: initial prior of hidden state at t=1
    """
    # Model dimensions
    K = A.shape[0]
    T = py_x.shape[1]
    assert py_x.shape[0] == K, 'Dimension mismatch between A and py_x.'

    # Initialize prior at t=1
    p1 = np.ones(K) / K if p1 is None else p1

    # Initialize the trellises
    Tre_p = np.zeros((K, T))  # store max probability
    Tre_p[:, 0] = np.log(p1) + np.log(py_x[:, 0])  # initial state probability
    Tre_h = np.zeros((K, T), dtype=int)  # store argmax state

    # Forward pass to fill the trellises
    for ii in range(1, T):  # t=2 -> t=T
        P = Tre_p[:, ii-1] + np.log(A) + np.log(py_x[:, ii])[:, None]
        Tre_h[:, ii] = np.argmax(P, axis=1)
        Tre_p[:, ii] = P[range(K), Tre_h[:, ii]]

    # Backward pass to identify the global argmax path
    viterbi_path = np.zeros(T, dtype=int)
    viterbi_path[-1] = np.argmax(Tre_p[:, -1])
    for ii in range(T-2, -1, -1):  # t=T-1 -> t=1
        viterbi_path[ii] = Tre_h[viterbi_path[ii+1], ii+1]

    # Output binary hidden state path
    bin_path = np.zeros((K, T), dtype=bool)
    for ii in range(K):
        bin_path[ii, viterbi_path == ii] = 1

    return viterbi_path, bin_path


def logdet(A):
    """ Computes logdet using Schur complement (Non-torch) """
    log_det = np.log(A[0, 0])
    for i in range(0, A.shape[0]-1):
        A = A[1:, 1:] - (A[1:, 0] @ A[0, 1:]) / A[0, 0]
        log_det += np.log(A[0, 0])
    return log_det


# noinspection PyUnusedLocal
def kalman(F, Q, mu0, Q0, G, R, y, R_weights=None, skip_interp=True):
    """
    This is the classical Kalman filter and fixed-interval smoother.
    Multivariate observation data y is supported. This implementation runs
    the kalman filtering and smoothing on CPU. This function has been heavily
    optimized for matrix computations in exchange for more intense memory use
    and arcane syntax.

    Reference:
        Shumway, R. H., & Stoffer, D. S. (1982). An approach to time
        series smoothing and forecasting using the EM algorithm.
        Journal of time series analysis, 3(4), 253-264.

        De Jong, P., & Mackinnon, M. J. (1988). Covariances for
        smoothed estimates in state space models. Biometrika, 75(3),
        601-602.

        Jazwinski, A. H. (2007). Stochastic processes and filtering
        theory. Courier Corporation.

    Inputs
    :param F: transition matrix
    :param Q: state noise covariance matrix
    :param mu0: initial state mean vector
    :param Q0: initial state covariance matrix
    :param G: observation matrix
    :param R: observation noise covariance matrix
    :param y: observed data (can be multivariate)
    :param R_weights: time-varying weights on the observation
            noise covariance (default: uniform unit weights)
    :param skip_interp: whether to skip calculating interpolated density (always True)
    :return: x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp (nan)
    """
    # Vector dimensions
    q, T = y.shape
    p = mu0.shape[0]
    approach_y = 'svd' if q >= 5 else 'gaussian'
    approach_x = 'svd' if p >= 5 else 'gaussian'

    if R_weights is None:
        R = R[:, :, None] * np.ones((q, q, T))
    else:  # (index 1 corresponds to t=1, etc.)
        R = R[:, :, None] * np.reshape(np.tile(R_weights, (q ** 2, 1)), (q, q, T))
        R[np.isnan(R)] = 0  # when there was 0 * inf = nan, set it to be 0

    # Kalman filtering (forward pass)
    x_t_tmin1 = np.zeros((p, T+1))  # (index 1 corresponds to t=0, etc.)
    P_t_tmin1 = np.zeros((p, p, T+1))
    K_t = np.zeros((p, q))
    x_t_t = np.zeros((p, T+1))
    P_t_t = np.zeros((p, p, T+1))
    logL = np.zeros(T)  # (index 1 corresponds to t=1)
    qlog2pi = q * np.log(2 * np.pi)

    # Initialize
    x_t_t[:, 0] = np.squeeze(mu0, axis=1)  # x_0_0
    P_t_t[:, :, 0] = Q0  # P_0_0

    # Recursion of forward pass
    for ii in range(1, T+1):  # t=1 -> t=T
        # One-step prediction
        x_t_tmin1[:, ii] = F @ x_t_t[:, ii-1]
        P_t_tmin1[:, :, ii] = F @ P_t_t[:, :, ii-1] @ F.T + Q

        # Update
        GP = G @ P_t_tmin1[:, :, ii]
        Sigma = GP @ G.T + R[:, :, ii-1]
        invSigma = inverse(Sigma, approach=approach_y)
        dy = y[:, ii-1] - G @ x_t_tmin1[:, ii]
        K_t = GP.T @ invSigma
        x_t_t[:, ii] = x_t_tmin1[:, ii] + K_t @ dy
        P_t_t[:, :, ii] = P_t_tmin1[:, :, ii] - K_t @ GP

        # Innovation form of the log likelihood
        logL[ii-1] = -(qlog2pi + logdet(Sigma) + dy.T @ invSigma @ dy) / 2

    # Kalman smoothing (backward pass)
    x_t_n = np.zeros((p, T+1))  # (index 1 corresponds to t=0, etc.)
    P_t_n = np.zeros((p, p, T+1))
    P_t_tmin1_n = np.zeros((p, p, T+1))  # cross-covariance between t and t-1
    fy_t_interp = float('nan')  # interpolated conditional density is not available in classical kalman filtering

    # Initialize
    x_t_n[:, -1] = x_t_t[:, -1]  # x_T_T
    P_t_n[:, :, -1] = P_t_t[:, :, -1]  # P_T_T

    # Recursion of backward pass
    for ii in range(T, 0, -1):  # t=T -> t=1
        J_t = P_t_t[:, :, ii-1] @ F.T @ inverse(P_t_tmin1[:, :, ii], approach=approach_x)
        JP = J_t @ P_t_n[:, :, ii]
        x_t_n[:, ii-1] = x_t_t[:, ii-1] + J_t @ (x_t_n[:, ii] - x_t_tmin1[:, ii])
        P_t_n[:, :, ii-1] = P_t_t[:, :, ii-1] + (JP - J_t @ P_t_tmin1[:, :, ii]) @ J_t.T
        P_t_tmin1_n[:, :, ii] = JP.T  # Cov(t,t-1) proved in De Jong & Mackinnon (1988)

    return x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp


def djkalman(F, Q, mu0, Q0, G, R, y, R_weights=None, skip_interp=True):
    """
    This is the [De Jong 1989] Kalman filter and fixed-interval smoother.
    Multivariate observation data y is supported. This implementation runs
    the kalman filtering and smoothing on CPU. This function has been heavily
    optimized for matrix computations in exchange for more intense memory use
    and arcane syntax. Note the filtered estimates are skipped since not used
    during recursion. If you need filtered estimate, refer to the equations
    derived in the MATLAB human-readable version (under <archive>) and add to
    the for loop.

    Since De Jong Kalman filtering and smoothing are not defined at t=0, we
    repeat the estimates at t=1 to extend to t=0. This is useful to allow
    updating initial state and covariance estimates in the M step.

    Reference:
        De Jong, P. (1989). Smoothing and interpolation with the
        state-space model. Journal of the American Statistical
        Association, 84(408), 1085-1088.

    Inputs
    :param F: transition matrix
    :param Q: state noise covariance matrix
    :param mu0: initial state mean vector
    :param Q0: initial state covariance matrix
    :param G: observation matrix
    :param R: observation noise covariance matrix
    :param y: observed data (can be multivariate)
    :param R_weights: time-varying weights on the observation
            noise covariance (default: uniform unit weights)
    :param skip_interp: whether to skip calculating interpolated density
    :return: x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp
    """
    # Vector dimensions
    q, T = y.shape
    p = mu0.shape[0]
    approach_y = 'svd' if q >= 5 else 'gaussian'

    if R_weights is None:
        R = R[:, :, None] * np.ones((q, q, T))
    else:  # (index 1 corresponds to t=1, etc.)
        R = R[:, :, None] * np.reshape(np.tile(R_weights, (q ** 2, 1)), (q, q, T))
        R[np.isnan(R)] = 0  # when there was 0 * inf = nan, set it to be 0

    # Kalman filtering (forward pass)
    x_t_tmin1 = np.zeros((p, T+2))  # (index 1 corresponds to t=0, etc.)
    P_t_tmin1 = np.zeros((p, p, T+2))
    K_t = np.zeros((p, q, T+1))  # note that this is different from the classical Kalman gain by pre-multiplying with F
    e_t = np.zeros((q, T+1))
    invD_t = np.zeros((q, q, T+1))
    L_t = np.zeros((p, p, T+1))
    logL = np.zeros(T)  # (index 1 corresponds to t=1)
    qlog2pi = q * np.log(2 * np.pi)

    # Initialize
    x_t_t = mu0  # x_0_0, initialized only to keep return outputs consistent
    P_t_t = Q0  # P_0_0, initialized only to keep return outputs consistent
    x_t_tmin1[:, 1] = F @ np.squeeze(x_t_t, axis=1)  # x_1_0 (W_0*beta in De Jong 1989)
    P_t_tmin1[:, :, 1] = F @ P_t_t @ F.T + Q  # P_1_0 (V_0 in De Jong 1989)

    # Recursion of forward pass
    for ii in range(1, T+1):  # t=1 -> t=T
        # Intermediate vectors
        e_t[:, ii] = y[:, ii-1] - G @ x_t_tmin1[:, ii]  # same as dy in classical kalman
        D_t = G @ P_t_tmin1[:, :, ii] @ G.T + R[:, :, ii-1]  # same as Sigma in classical kalman
        invD_t[:, :, ii] = inverse(D_t, approach=approach_y)
        FP = F @ P_t_tmin1[:, :, ii]
        K_t[:, :, ii] = FP @ G.T @ invD_t[:, :, ii]
        L_t[:, :, ii] = F - K_t[:, :, ii] @ G

        # One-step prediction for the next time point
        x_t_tmin1[:, ii+1] = F @ x_t_tmin1[:, ii] + K_t[:, :, ii] @ e_t[:, ii]
        P_t_tmin1[:, :, ii+1] = FP @ L_t[:, :, ii].T + Q

        # Innovation form of the log likelihood
        logL[ii-1] = -(qlog2pi + logdet(D_t) + e_t[:, ii].T @ invD_t[:, :, ii] @ e_t[:, ii]) / 2

    # remove the extra t=T+1 time point created
    x_t_tmin1 = x_t_tmin1[:, :-1]
    P_t_tmin1 = P_t_tmin1[:, :, :-1]

    # Kalman smoothing (backward pass)
    r_t = np.zeros((p, T+1))  # (index 1 corresponds to t=0, etc.)
    R_t = np.zeros((p, p, T+1))
    x_t_n = np.zeros((p, T+1))
    P_t_n = np.zeros((p, p, T+1))
    P_t_tmin1_n = np.zeros((p, p, T+1))  # cross-covariance between t and t-1
    Ip = np.eye(p)

    # Recursion of backward pass - fixed-interval smoothing
    for ii in range(T, 0, -1):  # t=T -> t=1
        # Intermediate vectors
        GD = G.T @ invD_t[:, :, ii]
        r_t[:, ii-1] = GD @ e_t[:, ii] + L_t[:, :, ii].T @ r_t[:, ii]
        R_t[:, :, ii-1] = GD @ G + L_t[:, :, ii].T @ R_t[:, :, ii] @ L_t[:, :, ii]

        # Smoothed estimates
        x_t_n[:, ii] = x_t_tmin1[:, ii] + P_t_tmin1[:, :, ii] @ r_t[:, ii-1]
        RP = R_t[:, :, ii-1] @ P_t_tmin1[:, :, ii]
        P_t_n[:, :, ii] = P_t_tmin1[:, :, ii] @ (Ip - RP)
        # Cov(t,t-1) is derived using Theorem 1 and Lemma 1, m = t, s = t+1
        P_t_tmin1_n[:, :, ii] = (Ip - RP.T) @ L_t[:, :, ii-1] @ P_t_tmin1[:, :, ii-1].T

    # Set the cross-covariance estimate at t=1. Cov(t=1,t=0) can be computed exactly using J_0.
    # But we use P_t=1_n instead to avoid inverting conditional state noise covariance.
    P_t_tmin1_n[:, :, 1] = P_t_n[:, :, 1]

    # Repeat t=1 to extend the smoothed estimates to t=0
    x_t_n[:, 0] = x_t_n[:, 1]
    P_t_n[:, :, 0] = P_t_n[:, :, 1]

    # Interpolated conditional density of y_t
    if skip_interp:
        fy_t_interp = float('nan')
    else:
        # fy_t_interp = normpdf(y_t | y_1,...,y_t-1,y_t+1,y_T)
        fy_t_interp = np.zeros(T)  # (index 1 corresponds to t=1, etc.)
        for ii in range(0, T):  # t=1 -> t=T
            n_t = invD_t[:, :, ii+1] @ e_t[:, ii+1] - K_t[:, :, ii+1].T @ r_t[:, ii+1]
            N_t = invD_t[:, :, ii+1] + K_t[:, :, ii+1].T @ R_t[:, :, ii+1] @ K_t[:, :, ii+1]
            # See De Jong 1989 Section 5, note that -logdet(N_t) is NOT a typo
            if np.linalg.cond(N_t) < (1 / np.finfo(float).eps):
                invN_t = inverse(N_t, approach=approach_y)
            else:
                invN_t = np.ones(N_t.shape, dtype=np.float64) * float('inf')
            fy_t_interp[ii] = np.exp(-(qlog2pi - logdet(N_t) + n_t.T @ invN_t @ n_t) / 2)

    return x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp


def inverse(A, approach='svd'):
    """
    Custom inverse function for inverting large covariance matrices

    Several approaches are possible:
        - 'gaussian': Gaussian elimination
        - 'cholesky': Cholesky decomposition
        - 'qr': QR decomposition
        - 'svd': Singular value decomposition

    Authors: Proloy Das <pd640@mgh.harvard.edu>
             Mingjian He <mh105@mit.edu>
    """
    if np.isinf(A).any():
        return np.zeros(A.shape, dtype=A.dtype)

    if approach == 'gaussian':
        return np.linalg.inv(A)

    if approach == 'cholesky':
        L = np.linalg.cholesky(A)
        L_inv = solve_triangular(L, np.eye(L.shape[0], dtype=L.dtype), lower=True)
        return L_inv.T @ L_inv

    elif approach == 'qr':
        Q, R = np.linalg.qr(A, mode='complete')
        R_inv = solve_triangular(R, np.eye(R.shape[0], dtype=R.dtype), lower=False)
        return R_inv @ Q.T

    elif approach == 'svd':
        U, S, Vh = np.linalg.svd(A, full_matrices=True)
        S = _reciprocal_pos_vals(S)
        return (Vh.T * S) @ U.T

    else:
        raise ValueError('Specified matrix inversion approach is not recognized.')


def _reciprocal_pos_vals(S, epsilon=1e-5):
    """ Take the reciprocal of positive values and set the rest to zero """
    S_pos_idx = (S / S[0]) > epsilon  # find indices of positive (large enough) singular values
    S[~S_pos_idx] = 0.0  # set the remaining singular values to 0
    S[S_pos_idx] = 1.0 / S[S_pos_idx]  # reciprocal of positive singular values
    return S
