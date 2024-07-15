"""
Author: Mingjian He <mh1@stanford.edu>

traditional module contains traditional methods for switching inference
"""

from somata.basic_models import StateSpaceModel as Ssm
from somata.exact_inference import forward_backward, logdet, inverse
import numpy as np


def switching(ssm_array, y=None, method='static', dwell_prob=0.99, A=None,
              future_steps=0, fix_prior=False, mimic1991=False, HMM_smooth=None):
    """
    Traditional switching inference functions.
    This is a collection of multiple traditional switching state-space
    model inference algorithms. There are a few different approaches,
    each associated with different input parameter options. All the
    methods output posterior model probabilities in the end. The
    references associated with each method will be provided above the
    implementation.

    Inputs:
    :param ssm_array: an array of Ssm class objects or a single object with stacked models
    :param y: observed data
    :param method: which traditional switching inference method to use.
                   Options are:
                    - 'static' - static multiple model using Bayes rule
                    - 'gpb1' - first-order generalized pseudo-Bayesian estimator
                    - 'gpb2' - second-order generalized pseudo-Bayesian estimator
                    - 'imm' - interacting multiple model algorithm
                    - '1991' - Shumway & Stoffer 1991 modified Kalman filtering method
                    - 'a parp' - Parallel method: predicted density for SSM with HMM filtering
                    - 'a pari' - Parallel method: interpolated density for SSM with HMM filtering
                    - 'ab parp' - Parallel method: predicted density for SSM with HMM smoothing
                    - 'ab pari' - Parallel method: interpolated density for SSM with HMM smoothing
                    - 'a pars' and 'ab pars': testing methods, do not use
    :param dwell_prob: dwell probability for HMM transitions to initialize A
    :param A: HMM transition matrix, if provided will override dwell_prob
    :param future_steps: A heuristic pseudo smoothing technique provided for the 1991 method
    :param fix_prior: parameter specific to parallel methods, whether to fix prior in HMM
                      filtering step, if not False will remove temporal continuity and MC model
    :param mimic1991: parameter specific to parallel methods, whether to use fy_t as an
                      approximation to fy_j(t|t-1) in the 1991 method and follow with the
                      same form of model probability filtering
    :param HMM_smooth: an option for all traditional methods (except parallel methods) to
                       apply the forward-backward algorithm to fy_t to obtain filtered
                       and smoothed model probabilities. None -> no additional processing,
                       'a' -> filtering, 'ab' -> smoothing. Parallel methods will set the
                       option automatically depending on what version is specified.

    Outputs:
    :returns:
        - Mprob: posterior model probabilities
        - fy_t: conditional density of y given t=1...t-1
    """
    ssm_array, K, T = Ssm.setup_array(ssm_array, y=y)

    # Initial guess of transition probability matrix A for HMM
    A = abs(np.eye(K, dtype=np.float64) -
            np.ones((K, K), dtype=np.float64) * (1-dwell_prob) / (K-1)) if A is None else A

    """
    Choose among the various switching state-space methods:
        - 'static' is similar to 'a parp'
        - 'gpb1'   is similar to '1991'
        - 'gpb2'   is similar to 'imm'
    """
    if method == 'static':
        Mprob, fy_t = _static(ssm_array, K, T)
    elif method == 'gpb1':
        Mprob, fy_t = _gpb1(ssm_array, K, T, A)
    elif method == 'gpb2':
        Mprob, fy_t = _gpb2(ssm_array, K, T, A)
    elif method == 'imm':
        Mprob, fy_t = _imm(ssm_array, K, T, A)
    elif method == '1991':
        Mprob, fy_t = _1991(ssm_array, K, T, A, future_steps=future_steps)
    elif 'par' in method:
        Mprob, fy_t, HMM_smooth = _parallel(ssm_array, K, T, A, method, fix_prior=fix_prior, mimic1991=mimic1991)
    else:
        raise Exception('Unrecognized switching inference method.')

    # Truncate the first time point to output vectors t=1 -> t=T
    Mprob = Mprob[:, 1:] if 'par' not in method else Mprob

    # Apply forward-backward algorithm to fy_t if specified
    if HMM_smooth is not None:
        if HMM_smooth == 'a':  # alpha filtering
            Mprob, *_ = forward_backward(A, fy_t)
        elif HMM_smooth == 'ab':  # alpha-beta smoothing
            _, Mprob, *_ = forward_backward(A, fy_t)

    return Mprob, fy_t


def _static(ssm_array, K, T):
    """
    This is the naive static multiple model derived using
    a direct application of Bayes rule. It essentially
    uses predicted conditional density and runs an update
    on the prior for each time step. This method is
    obsolete now given the later development of dynamic
    methods, however with an ad-hoc modification of lower
    bounding model probability, it is still used in the
    literature.

    Reference:
        Fabri, S., & Kadirkamanathan, V. (2001). Functional
        adaptive control: an intelligent systems approach.
        Springer Science & Business Media.
    """
    # Parallel run Kalman filtering on the static models
    _, _, _, _, _, _, _, x_t_tmin1_all, P_t_tmin1_all, _ = Ssm.par_kalman(ssm_array, method='kalman')

    # Compute the conditional density (aka. predicted likelihood function of y)
    fy_t = _compute_fy_t(ssm_array, x_t_tmin1_all, P_t_tmin1_all)

    # Initialize posterior model probability
    Mprob = np.zeros((K, T+1), dtype=np.float64)  # (index 1 corresponds to t=0)
    Mprob[:, 0] = np.ones(K, dtype=np.float64) / K

    # Iterate through the time steps
    for ii in range(1, T+1):  # t=1 -> t=T
        bayes_num = fy_t[:, ii-1] * Mprob[:, ii-1]
        Mprob[:, ii] = bayes_num / bayes_num.sum()  # direct application of Bayes rule

        # An ad-hoc solution of lower bounding model probability
        lower_bound = 1e-2
        if any(Mprob[:, ii] < lower_bound):
            Mprob[Mprob[:, ii] < lower_bound, ii] = lower_bound  # lower bound
            Mprob[:, ii] = Mprob[:, ii] / Mprob[:, ii].sum()  # re-normalize

    return Mprob, fy_t


def _gpb1(ssm_array, K, T, A, y=None):
    """
    This implements the generalized pseudo-Bayesian
    estimator of switching multiple models. The name
    comes from the fact that we combine at the end of
    each filtering + update step the K separate Gaussian
    distributions into a single approximating Gaussian.

    The merging of Gaussian is simply weighted averaging
    of the first two moments of the filtered state
    estimates using a pseudo-Bayesian update rule. It is
    not strictly Bayesian because the prior involves an
    averaging over transition matrix of discrete HMM.

    Note that this method is intimately related to the
    S&S 1991 approach. The only difference is that S&S
    1991 takes the weighted average of innovations while
    gpb1 takes the weighted average of filtered state
    estimates. The former is exact solution derived under
    the assumption that only the observation matrices
    switch, while the latter is approximating a mixture
    of K Gaussian by merging their first two moments.

    If K models have the same model parameters (1991),
    at every iteration, the two methods produce the same
    first moment for the merged filtered estimate, but
    gpb1 has an additional covariance matrix coming from
    the deviation of mean of each of K Gaussian from the
    mean of the merged Gaussian. Similarly, one can use
    the 1991 method to implement gpb1 by using augmented
    hidden states and making F and Q block diagonal.

    Reference:
        Gordon, K., & Smith, A. F. M. (1990). Modeling and
        monitoring biomedical time series. Journal of the
        American Statistical Association, 85(410), 328-337.
    """
    # gpb1 requires all the models to have the same dimensions
    # of the hidden states in order to merge Gaussian - this
    # is verified here
    first_ssm = ssm_array[0]
    for m in range(1, K):
        assert first_ssm.mu0.shape[0] == ssm_array[m].G.shape[1], 'Model hidden state dimensions are inconsistent.'

    # Array dimensions
    y = first_ssm.y if y is None else y
    p = first_ssm.nstate
    q = first_ssm.nchannel
    approach = 'svd' if q >= 5 else 'gaussian'
    qlog2pi = q * np.log(2 * np.pi)

    # Forward filtering trellis variables
    x_t_tmin1 = np.zeros((p, K, T+1), dtype=np.float64)  # (index 1 corresponds to t=0, etc.)
    P_t_tmin1 = np.zeros((p, p, K, T+1), dtype=np.float64)
    K_t = np.zeros((p, q, K, T+1), dtype=np.float64)
    x_t_t = np.zeros((p, T+1), dtype=np.float64)
    P_t_t = np.zeros((p, p, T+1), dtype=np.float64)
    Mprob = np.zeros((K, T+1), dtype=np.float64)
    fy_t = np.zeros((K, T), dtype=np.float64)  # conditional density of y given t=1...t-1 (index 1 corresponds to t=1)

    # Initial model probability at t=0
    Mprob[:, 0] = np.ones(K, dtype=np.float64) / K

    # Initialize x_0_0
    x_t_t[:, 0] = np.array([ssm_array[m].mu0 * Mprob[m, 0] for m in range(K)]).sum(axis=0)[:, 0]

    # Initialize P_0_0
    for m in range(K):
        P_t_t[:, :, 0] += (ssm_array[m].S0 +
                           (ssm_array[m].mu0 - x_t_t[:, 0][:, None]) @
                           (ssm_array[m].mu0 - x_t_t[:, 0][:, None]).T) * Mprob[m, 0]

    # Initialize log likelihood of filtering for each model
    logL = np.zeros(K, dtype=np.float64)

    # Forward iterations
    for ii in range(1, T+1):  # t=1 -> t=T
        # Store filtered estimates for each model to later combine
        # as one Gaussian at the end of the step
        x_t_t_all = np.zeros((p, K), dtype=np.float64)
        P_t_t_all = np.zeros((p, p, K), dtype=np.float64)

        # Separate filter for each model
        for m in range(K):
            # Current model parameters
            F = ssm_array[m].F
            Q = ssm_array[m].Q
            G = ssm_array[m].G
            R = ssm_array[m].R

            # One-step prediction
            x_t_tmin1[:, m, ii] = F @ x_t_t[:, ii-1]
            P_t_tmin1[:, :, m, ii] = F @ P_t_t[:, :, ii-1] @ F.T + Q
            Sigma = G @ P_t_tmin1[:, :, m, ii] @ G.T + R
            K_t[:, :, m, ii] = P_t_tmin1[:, :, m, ii] @ G.T @ inverse(Sigma, approach=approach)

            # Current time step Gaussian pdf of y_t
            e_t = y[:, ii-1] - G @ x_t_tmin1[:, m, ii]
            fy_t[m, ii-1] = np.exp(-(qlog2pi + logdet(Sigma) + e_t.T @ inverse(Sigma, approach=approach) @ e_t) / 2)

            # Innovation form of the log likelihood for the current model
            logL[m] += np.log(fy_t[m, ii-1])

            # Update equation to get filtered estimates for each model
            x_t_t_all[:, m] = x_t_tmin1[:, m, ii] + K_t[:, :, m, ii] @ e_t
            P_t_t_all[:, :, m] = P_t_tmin1[:, :, m, ii] - K_t[:, :, m, ii] @ G @ P_t_tmin1[:, :, m, ii]

        # Update model probability from last step
        Mprob_prior = A @ Mprob[:, ii-1]  # prior model probability
        bayes_num = fy_t[:, ii-1] * Mprob_prior
        Mprob[:, ii] = bayes_num / bayes_num.sum()  # filtered model probability

        # Merge the K filtered Gaussian into one Gaussian
        x_t_t[:, ii] = (x_t_t_all * Mprob[:, ii].T).sum(axis=1)
        for m in range(K):
            P_t_t[:, :, ii] += (P_t_t_all[:, :, m] +
                                (x_t_t_all[:, m] - x_t_t[:, ii])[:, None] @
                                (x_t_t_all[:, m] - x_t_t[:, ii])[:, None].T) * Mprob[m, ii]

    return Mprob, fy_t


def _gpb2(ssm_array, K, T, A, y=None):
    """
    This implements the second-order generalized pseudo
    Bayesian approach as described in Bar-Shalom & Li
    1993. The method is an extension of gpb1 but differs
    by maintaining K different hidden state estimate
    distributions similar to the IMM method.

    The computations are similar to IMM but have K^2
    number of filters. Merging of K Gaussian into single
    approximating Gaussian is done for each of the model
    at the end of each time step. This is different from
    IMM, which mixes distributions at the beginning of
    every step which requires K filters instead of K^2.

    It can be shown that IMM of second order is identical
    to gpb2, therefore this method also serves as a
    second order extension of the IMM method.

    Reference:
        Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2004).
        Estimation with applications to tracking and
        navigation: theory algorithms and software. John
        Wiley & Sons.
    """
    # gpb2 requires all the models to have the same dimensions
    # of the hidden states in order to merge Gaussian - this
    # is verified here
    first_ssm = ssm_array[0]
    for m in range(1, K):
        assert first_ssm.mu0.shape[0] == ssm_array[m].G.shape[1], 'Model hidden state dimensions are inconsistent.'

    # Array dimensions
    y = first_ssm.y if y is None else y
    p = first_ssm.nstate
    q = first_ssm.nchannel
    approach = 'svd' if q >= 5 else 'gaussian'
    qlog2pi = q * np.log(2 * np.pi)

    # Forward filtering trellis variables
    x_t_tmin1 = np.zeros((p, K, K, T+1), dtype=np.float64)  # (index 1 corresponds to t=0, etc.)
    P_t_tmin1 = np.zeros((p, p, K, K, T+1), dtype=np.float64)
    K_t = np.zeros((p, q, K, K, T+1), dtype=np.float64)
    x_t_t = np.zeros((p, K, T+1), dtype=np.float64)
    P_t_t = np.zeros((p, p, K, T+1), dtype=np.float64)
    Mprob = np.zeros((K, T+1), dtype=np.float64)
    fy_t = np.zeros((K, T), dtype=np.float64)  # conditional density of y given t=1...t-1 (index 1 corresponds to t=1)
    fy_t_all = np.zeros((K, K, T), dtype=np.float64)  # used for model probability update equation

    # Initial model probability at t=0
    Mprob[:, 0] = np.ones(K, dtype=np.float64) / K

    # Initialize x_0_0 and P_0_0
    for m in range(K):
        x_t_t[:, m, 0] = ssm_array[m].mu0[:, 0]
        P_t_t[:, :, m, 0] = ssm_array[m].S0

    # Initialize log likelihood of filtering for each model
    logL = np.zeros(K, dtype=np.float64)

    # Forward iterations
    for ii in range(1, T+1):  # t=1 -> t=T
        # Separate filter for each model
        for m in range(K):
            # Store filtered estimates for each model to later combine
            # as one Gaussian at the end of the step
            x_t_t_all = np.zeros((p, K), dtype=np.float64)
            P_t_t_all = np.zeros((p, p, K), dtype=np.float64)

            # Current model parameters
            F = ssm_array[m].F  # in state m at current time step
            Q = ssm_array[m].Q
            G = ssm_array[m].G
            R = ssm_array[m].R

            # Evaluate each model separately
            for j in range(K):
                # One-step prediction
                x_t_tmin1[:, m, j, ii] = F @ x_t_t[:, j, ii-1]  # in state j at last time step
                P_t_tmin1[:, :, m, j, ii] = F @ P_t_t[:, :, j, ii-1] @ F.T + Q
                Sigma = G @ P_t_tmin1[:, :, m, j, ii] @ G.T + R
                K_t[:, :, m, j, ii] = P_t_tmin1[:, :, m, j, ii] @ G.T @ inverse(Sigma, approach=approach)

                # Current time step Gaussian pdf of y_t
                e_t = y[:, ii-1] - G @ x_t_tmin1[:, m, j, ii]
                fy_t_all[m, j, ii-1] = np.exp(-(qlog2pi + logdet(Sigma) + e_t.T @ inverse(Sigma,
                                                                                          approach=approach) @ e_t) / 2)

                # Update equation to get filtered estimates
                x_t_t_all[:, j] = x_t_tmin1[:, m, j, ii] + K_t[:, :, m, j, ii] @ e_t
                P_t_t_all[:, :, j] = P_t_tmin1[:, :, m, j, ii] - K_t[:, :, m, j, ii] @ G @ P_t_tmin1[:, :, m, j, ii]

            # Compute the merging probability
            bayes_num = fy_t_all[m, :, ii-1] * A[m, :] * Mprob[:, ii-1].T
            Mprob_merge = bayes_num / bayes_num.sum()

            # Merge the K Gaussian into one Gaussian for each model
            x_t_t[:, m, ii] = (x_t_t_all * Mprob_merge).sum(axis=1)
            for j in range(K):
                P_t_t[:, :, m, ii] += (P_t_t_all[:, :, j] +
                                       (x_t_t_all[:, j] - x_t_t[:, m, ii])[:, None] @
                                       (x_t_t_all[:, j] - x_t_t[:, m, ii])[:, None].T) * Mprob_merge[j]

            # Weighted averaging of conditional density to get fy_t
            fy_t[m, ii-1] = fy_t_all[m, :, ii-1] @ Mprob_merge

            # Innovation form of the log likelihood for the current model
            logL[m] += np.log(fy_t[m, ii-1])

        # Update model probability from last step
        bayes_mat = fy_t_all[:, :, ii-1] * A * Mprob[:, ii-1].T
        Mprob[:, ii] = bayes_mat.sum(axis=1) / bayes_mat.sum()

    return Mprob, fy_t


def _imm(ssm_array, K, T, A, y=None):
    """
    This implements the interacting multiple model (IMM)
    approach as sketched out in Bar-Shalom & Li 1993. The
    method is distinguished by the key feature of mixing
    Gaussian at beginning of a step in Kalman filtering
    for each of the K models. Therefore, this method
    maintains K different hidden state estimate
    distributions, which interact at the beginning of
    every step using mixing probabilities that are
    model probabilities at time step t-1 conditioned on
    being at a specific state at time step t. Note that
    this is different from the posterior model
    probabilities used for mixing filtered estimates
    (gpb1) or innovations (1991), which occur at the end
    of a time step in Kalman filtering.

    Reference:
        Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2004).
        Estimation with applications to tracking and
        navigation: theory algorithms and software. John
        Wiley & Sons.
    """
    # imm requires all the models to have the same dimensions
    # of the hidden states in order to merge Gaussian - this
    # is verified here
    first_ssm = ssm_array[0]
    for m in range(1, K):
        assert first_ssm.mu0.shape[0] == ssm_array[m].G.shape[1], 'Model hidden state dimensions are inconsistent.'

    # Array dimensions
    y = first_ssm.y if y is None else y
    p = first_ssm.nstate
    q = first_ssm.nchannel
    approach = 'svd' if q >= 5 else 'gaussian'
    qlog2pi = q * np.log(2 * np.pi)

    # Forward filtering trellis variables
    x_t_tmin1 = np.zeros((p, K, T+1), dtype=np.float64)  # (index 1 corresponds to t=0, etc.)
    P_t_tmin1 = np.zeros((p, p, K, T+1), dtype=np.float64)
    K_t = np.zeros((p, q, K, T+1), dtype=np.float64)
    x_t_t = np.zeros((p, K, T+1), dtype=np.float64)
    P_t_t = np.zeros((p, p, K, T+1), dtype=np.float64)
    Mprob = np.zeros((K, T+1), dtype=np.float64)
    fy_t = np.zeros((K, T), dtype=np.float64)  # conditional density of y given t=1...t-1 (index 1 corresponds to t=1)

    # Initial model probability at t=0
    Mprob[:, 0] = np.ones(K, dtype=np.float64) / K

    # Initialize x_0_0 and P_0_0
    for m in range(K):
        x_t_t[:, m, 0] = ssm_array[m].mu0[:, 0]
        P_t_t[:, :, m, 0] = ssm_array[m].S0

    # Initialize log likelihood of filtering for each model
    logL = np.zeros(K, dtype=np.float64)

    # Forward iterations
    for ii in range(1, T+1):  # t=1 -> t=T
        # Compute the mixing probability
        bayes_num = A * Mprob[:, ii-1].T
        # each row is the mixing probability of other models for model m
        Mprob_mix = bayes_num / bayes_num.sum(axis=1)[:, None]

        # Separate filter for each model
        for m in range(K):
            # Current model parameters
            F = ssm_array[m].F
            Q = ssm_array[m].Q
            G = ssm_array[m].G
            R = ssm_array[m].R

            # Mix the last step hidden states from K models
            x0_t_t = (x_t_t[:, :, ii-1] * Mprob_mix[m, :]).sum(axis=1)
            P0_t_t = np.zeros((p, p), dtype=np.float64)
            for j in range(K):
                P0_t_t += (P_t_t[:, :, j, ii-1] +
                           (x_t_t[:, j, ii-1] - x0_t_t)[:, None] @
                           (x_t_t[:, j, ii-1] - x0_t_t)[:, None].T) * Mprob_mix[m, j]

            # One-step prediction using the merged Gaussian
            x_t_tmin1[:, m, ii] = F @ x0_t_t
            P_t_tmin1[:, :, m, ii] = F @ P0_t_t @ F.T + Q
            Sigma = G @ P_t_tmin1[:, :, m, ii] @ G.T + R
            K_t[:, :, m, ii] = P_t_tmin1[:, :, m, ii] @ G.T @ inverse(Sigma, approach=approach)

            # Current time step Gaussian pdf of y_t
            e_t = y[:, ii-1] - G @ x_t_tmin1[:, m, ii]
            fy_t[m, ii-1] = np.exp(-(qlog2pi + logdet(Sigma) + e_t.T @ inverse(Sigma, approach=approach) @ e_t) / 2)

            # Innovation form of the log likelihood for the current model
            logL[m] += np.log(fy_t[m, ii-1])

            # Update equation to get filtered estimates for each model
            x_t_t[:, m, ii] = x_t_tmin1[:, m, ii] + K_t[:, :, m, ii] @ e_t
            P_t_t[:, :, m, ii] = P_t_tmin1[:, :, m, ii] - K_t[:, :, m, ii] @ G @ P_t_tmin1[:, :, m, ii]

        # Update model probability from last step
        Mprob_prior = (A * Mprob[:, ii-1].T).sum(axis=1)
        bayes_num = fy_t[:, ii-1] * Mprob_prior
        Mprob[:, ii] = bayes_num / bayes_num.sum()  # filtered model probability

    return Mprob, fy_t


def _1991(ssm_array, K, T, A, y=None, future_steps=0):
    """
    This implements the original Shumway & Stoffer 1991
    modified Kalman filtering approach for switching
    state space models. A distinguishing feature of this
    method from the gpb1 method is that only the
    observation matrices switch in S&S 1991, therefore
    the Kalman filtering equations are exact by weighting
    the innovations instead of merging Gaussian as
    approximation. Thus, unless using pseudo-EM to learn
    the SSM parameters (in S&S 1991 Appendix), this
    method does not involve approximating Gaussian like
    all the other traditional switching inference methods.

    Reference:
        Shumway, R. H., & Stoffer, D. S. (1991). Dynamic
        linear models with switching. Journal of the American
        Statistical Association, 86(415), 763-769.

    There is an optional pseudo-smoothing available by
    specifying <future_steps> to be greater than 0. It
    is a heuristic modification of the 1991 method by
    replacing the conditional density of y with the
    summation of the conditional density of y under each
    alternative model after filtering into the future for
    <future_steps> steps. This is not strictly derivable
    under Bayes rule and therefore only serves as a
    heuristic solution.
    """
    # 1991 requires all the models to have the same properties
    # except the observation matrices - this is verified here
    first_ssm = ssm_array[0]
    for m in range(1, K):
        assert (first_ssm.F == ssm_array[m].F).all(), 'Models have different F matrices.'
        assert (first_ssm.Q == ssm_array[m].Q).all(), 'Models have different Q matrices.'
        assert (first_ssm.mu0 == ssm_array[m].mu0).all(), 'Models have different mu0 vectors.'
        assert (first_ssm.S0 == ssm_array[m].S0).all(), 'Models have different S0 matrices.'
        assert (first_ssm.R == ssm_array[m].R).all(), 'Models have different R matrices.'
        assert first_ssm.mu0.shape[0] == ssm_array[m].G.shape[1], 'Model hidden state dimensions are inconsistent.'

    # Array dimensions
    y = first_ssm.y if y is None else y
    p = first_ssm.nstate
    q = first_ssm.nchannel
    approach = 'svd' if q >= 5 else 'gaussian'
    qlog2pi = q * np.log(2 * np.pi)

    # Forward filtering trellis variables
    x_t_tmin1 = np.zeros((p, T+1), dtype=np.float64)  # (index 1 corresponds to t=0, etc.)
    P_t_tmin1 = np.zeros((p, p, T+1), dtype=np.float64)
    K_t = np.zeros((p, q, K, T+1), dtype=np.float64)
    x_t_t = np.zeros((p, T+1), dtype=np.float64)
    P_t_t = np.zeros((p, p, T+1), dtype=np.float64)
    Mprob = np.zeros((K, T+1), dtype=np.float64)
    fy_t = np.zeros((K, T), dtype=np.float64)  # conditional density of y given t=1...t-1 (index 1 corresponds to t=1)

    # Initial model probability at t=0
    Mprob[:, 0] = np.ones(K, dtype=np.float64) / K  # pi_0_0

    # Initialize hidden states at t=0
    x_t_t[:, 0] = first_ssm.mu0[:, 0]  # x_0_0
    P_t_t[:, :, 0] = first_ssm.S0  # P_0_0

    # Initialize log likelihood of filtering for each model
    logL = np.zeros(K, dtype=np.float64)

    # Set up model parameters
    F = first_ssm.F
    Q = first_ssm.Q
    R = first_ssm.R

    # Forward iterations
    for ii in range(1, T+1):  # t=1 -> t=T
        # One-step prediction
        x_t_tmin1[:, ii] = F @ x_t_t[:, ii-1]
        P_t_tmin1[:, :, ii] = F @ P_t_t[:, :, ii-1] @ F.T + Q

        # Compute innovations from K models
        fy_t_tmin1 = np.zeros((K, min(T+1-ii, future_steps+1)), dtype=np.float64)
        for m in range(K):
            G = ssm_array[m].G  # current model observation matrix
            Sigma = G @ P_t_tmin1[:, :, ii] @ G.T + R
            K_t[:, :, m, ii] = P_t_tmin1[:, :, ii] @ G.T @ inverse(Sigma, approach=approach)

            # Current time step Gaussian pdf of y_t
            e_t = y[:, ii-1] - G @ x_t_tmin1[:, ii]
            fy_t_tmin1[m, 0] = np.exp(-(qlog2pi + logdet(Sigma) + e_t.T @ inverse(Sigma, approach=approach) @ e_t) / 2)

            # Innovation form of the log likelihood for the current model
            logL[m] += np.log(fy_t_tmin1[m, 0])

            # Filter into the future if future_steps > 0
            temp_x_t_tmin1 = np.zeros((p, future_steps+1), dtype=np.float64)
            temp_x_t_tmin1[:, 0] = x_t_tmin1[:, ii]
            temp_P_t_tmin1 = np.zeros((p, p, future_steps+1), dtype=np.float64)
            temp_P_t_tmin1[:, :, 0] = P_t_tmin1[:, :, ii]
            temp_K_t = np.zeros((p, q, future_steps+1), dtype=np.float64)
            temp_K_t[:, :, 0] = K_t[:, :, m, ii]
            temp_x_t_t = np.zeros((p, future_steps+1), dtype=np.float64)
            temp_P_t_t = np.zeros((p, p, future_steps+1), dtype=np.float64)

            # Filter forward in the current model for <future_steps> steps
            for k in range(1, min(T+1-ii, future_steps+1)):
                # Finish off the filtering for the last time step
                temp_x_t_t[:, k-1] = temp_x_t_tmin1[:, k-1] + temp_K_t[:, :, k-1] @ (y[:, ii-1+k-1] -
                                                                                     G @ temp_x_t_tmin1[:, k-1])
                temp_P_t_t[:, :, k-1] = temp_P_t_tmin1[:, :, k-1] - temp_K_t[:, :, k-1] @ G @ temp_P_t_tmin1[:, :, k-1]

                # Compute the prediction for the current step
                temp_x_t_tmin1[:, k] = F @ temp_x_t_t[:, k-1]
                temp_P_t_tmin1[:, :, k] = F @ temp_P_t_t[:, :, k-1] @ F.T + Q
                temp_Sigma = G @ temp_P_t_tmin1[:, :, k] @ G.T + R
                temp_K_t[:, :, k] = temp_P_t_tmin1[:, :, k] @ G.T @ inverse(temp_Sigma, approach=approach)

                # Calculate the conditional density of f_y for the current step
                temp_e_t = y[:, ii-1+k] - G @ temp_x_t_tmin1[:, k]
                fy_t_tmin1[m, k] = np.exp(-(qlog2pi + logdet(temp_Sigma) +
                                            temp_e_t.T @ inverse(temp_Sigma, approach=approach) @ temp_e_t) / 2)

        # Calculate filtered model probabilities
        fy_t[:, ii-1] = fy_t_tmin1.sum(axis=1)  # sum conditional density after filtering <future_steps> with all models
        Mprob_prior = A @ Mprob[:, ii-1]  # prior model probability
        bayes_num = fy_t[:, ii-1] * Mprob_prior
        Mprob[:, ii] = bayes_num / bayes_num.sum()  # filtered model probability

        # Single joint filter by merging innovations from K models using filtered model probability
        x_update = np.zeros(p, dtype=np.float64)
        P_update = np.zeros((p, p), dtype=np.float64)
        for m in range(K):
            G = ssm_array[m].G  # current model observation matrix
            x_update += Mprob[m, ii] * K_t[:, :, m, ii] @ (y[:, ii-1] - G @ x_t_tmin1[:, ii])
            P_update += Mprob[m, ii] * (P_t_tmin1[:, :, ii] - K_t[:, :, m, ii] @ G @ P_t_tmin1[:, :, ii])
        x_t_t[:, ii] = x_t_tmin1[:, ii] + x_update
        P_t_t[:, :, ii] = P_update

    return Mprob, fy_t


def _parallel(ssm_array, K, T, A, method, fix_prior=False, mimic1991=False):
    """
    "Parallel model" approach treats the Gaussian process
    hidden states as known from the Kalman filtering and
    smoothing. The rest is then just the classical
    alpha-beta forward backward algorithm on a discrete
    state Markov Chain (hidden Markov model) that
    captures the switching state variable.

    Reference:
        He, M., Das, P., Hotan, G., & Purdon, P. L. (2023). Switching
        state-space modeling of neural signal dynamics. PLoS
        Computational Biology, 19(8), e1011395.

    The key assumption here is that after running
    Kalman filtering and smoothing of the parallel
    models, we use the predicted or interpolated density
    as approximations to the observation probability in
    the hidden Markov model and use forward backward
    algorithm to complete *approximate* inference.

    Two approximations are at play:
        1) Approximate conditional independence across y
        2) Approximate density with static history models

    Abbreviations:
        - 'a parp' = filtered MC with predicted Gaussian states (conditional independence is not necessary)
        - 'a pari' = filtered MC with interpolated Gaussian states (still non-causal)
        - 'ab parp' = smoothed MC with predicted Gaussian states
        - 'ab pari' = smoothed MC with interpolated Gaussian states (best option)
        - 'a pars' and 'ab pars': testing methods, do not use
    """
    """ SSM """
    # Run parallel kalman filtering and smoothing on an array of ssm objects,
    # then compute the conditional density of y at each time step
    #
    # Note that all the choices of hidden state estimates can be viewed as
    # approximations of fy_j(t|t-1) in the S&S 1991 approach if alpha-filtering
    # is used for HMM

    if 'parp' in method:  # predicted conditional density
        _, _, _, _, _, _, _, x_t_tmin1_all, P_t_tmin1_all, _ = Ssm.par_kalman(ssm_array, method='kalman')
        fy_t = _compute_fy_t(ssm_array, x_t_tmin1_all, P_t_tmin1_all)
    elif 'pars' in method:  # smoothed conditional density (do not use)
        x_t_n_all, P_t_n_all, *_ = Ssm.par_kalman(ssm_array, method='kalman')
        fy_t = _compute_fy_t(ssm_array, x_t_n_all, P_t_n_all)
    elif 'pari' in method:  # interpolated density using De Jong filtering and smoothing
        _, _, _, _, _, _, _, _, _, fy_t = Ssm.par_kalman(ssm_array, method='dejong', skip_interp=False)
        fy_t = np.vstack(fy_t)
    else:
        raise Exception('Unrecognized switching inference method.')

    """ HMM """
    # Force mimic1991 to be False if doing smoothing on HMM
    if 'ab' in method and mimic1991:
        raise Exception('Smoothing HMM cannot mimic the 1991 method.')

    if mimic1991:
        # If we try to use fy_t as an approximation of fy_j(t|t-1)
        # in the 1991 method, we need a slightly different
        # initialization of the alpha state at t=1, therefore we
        # cannot use te dedicated implementation of forward-backward
        # algorithm as the other traditional methods. We implement
        # the associated alpha filtering below.
        HMM_smooth = None

        if fix_prior:
            assert len(fix_prior) == K, 'Fix prior needs to provide the prior for each model.'  # type: ignore
            # A fixed predictor / prior lacks temporal continuity, not MC anymore
            predictor = fix_prior  # this is provided as an option to mimic the 1991 method

        norm_a = np.zeros((K, T), dtype=np.float64)  # (index 1 corresponds to t=0)
        norm_a[:, 0] = np.ones(K, dtype=np.float64) / K  # equivalent to pi_0_0 in the 1991 method
        for ii in range(1, T):  # t=1 -> t=T
            if not fix_prior:
                predictor = A @ norm_a[:, ii-1]  # equivalent to Mprob_prior in the 1991 method
            # noinspection PyUnboundLocalVariable
            a = fy_t[:, ii-1] * predictor
            norm_a[:, ii] = a / a.sum()

        # This is equivalent to the 1991 approach using custom hidden state
        # estimates to approximate fy_j(t|t-1) and doing the same filtering
        # estimates for discrete switching states
        Mprob = norm_a[:, 1:]  # Truncate the first time point to output vectors t=1 -> t=T
    else:
        # Invoke dedicated forward-backward algorithm with more conventional initialization
        Mprob = None
        HMM_smooth = method.split()[0]

    return Mprob, fy_t, HMM_smooth


def _compute_fy_t(ssm_array, x_t_all, P_t_all, y=None):
    """ Compute the conditional Gaussian density of P(y_t|x_t) """
    # Array dimensions
    y = ssm_array[0].y if y is None else y
    K = len(ssm_array)  # number of models
    q, T = y.shape  # number of channels and number of time points
    approach = 'svd' if q >= 5 else 'gaussian'
    qlog2pi = y.shape[0] * np.log(2 * np.pi)

    fy_t = np.zeros((K, T), dtype=np.float64)  # (index 1 corresponds to t=1)

    for m in range(K):
        x_t = x_t_all[m][:, 1:]  # (index 1 corresponds to be t=0, etc.)
        P_t = P_t_all[m][:, :, 1:]
        for ii in range(T):  # t=1 -> t=T
            Sigma = ssm_array[m].G @ P_t[:, :, ii] @ ssm_array[m].G.T + ssm_array[m].R
            e_t = y[:, ii] - ssm_array[m].G @ x_t[:, ii]
            fy_t[m, ii] = np.exp(-(qlog2pi + logdet(Sigma) + e_t.T @ inverse(Sigma, approach=approach) @ e_t) / 2)
    return fy_t
