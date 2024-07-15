"""
Author: Ran Liu <rliu20@stanford.edu>

Unit tests for functions in somata/pac/pac_model.py
"""

import numpy as np
from somata.pac.pac_model import (
    fit_pac_regression,
    kmod,
    autocovariances,
    block_toeplitz,
    ar_parameters,
    mvar_ssm,
    optimize_arp,
)


def generate_pac_data():
    """ Generate test data for pac regression (not stochastic)"""
    n = int(1e3)
    Fs = 100
    phase = np.arange(0, n / Fs * 2 * np.pi, 2 * np.pi / Fs)
    amplitude = np.sin(phase)
    amplitude[amplitude < 0] = 0

    return phase, amplitude


def generate_ar3_data(rng=np.random.default_rng(0), n=int(1e3)):
    """ Generate test data from a univariate AR(3) process """
    p = 3  # autoregressive order of 3
    x0 = np.array([1., 2., 3.])
    F = np.array([[1, -0.5, 0.3], [1., 0., 0.], [0., 1., 0.]])  # hard-coded for AR(3)

    x = np.zeros((p, n), dtype=np.float64)  # augmented hidden states
    current = x0

    for i in range(n):
        current = F @ current
        current[0] += rng.standard_normal()  # Q = 1.0
        x[:, i] = current

    y = x[0, :] + rng.standard_normal(n)  # R = 1.0; G = [1, 0, 0]

    return x, y


def generate_mvar_data(q, rng=np.random.default_rng(0), n=int(1e3)):
    """
    Generate test data from a multivariate AR(3) process

    :param q: number of observed variables (dimensionality of the mvAR process)
    :param rng: random number generator
    :param n: length of the mvAR time series
    """
    data = [generate_ar3_data(rng) for i in range(q)]  # univariate AR(3) data for each variable
    x = np.row_stack([x[0, :] for x, _ in data])
    y = np.row_stack([y for _, y in data])  # make a mvAR(3) from three univariate AR(3) data

    return x, y


def test_pac_regression():
    phase, amplitude = generate_pac_data()

    # test whether the constrained regression is fitted correctly
    fit = fit_pac_regression(phase, amplitude, verbose=False)
    posterior_mean = fit.mean(0)  # posterior mean as an approximate to posterior mode

    fitted_kmod = kmod(posterior_mean[0], posterior_mean[1], posterior_mean[2])

    assert fitted_kmod >= 0, "Estimated Kmod should be non-negative."
    assert fitted_kmod < 1, "Estimated Kmod should be less than 1."


def test_autocovariances():
    rng = np.random.default_rng(0)
    q = 2  # mvAR with 2 variables
    p = 2  # up to lag 2

    _, y = generate_mvar_data(q=q, rng=rng)
    acov = autocovariances(y, p, bias=True)

    # Assert that we have p + 1 covariance matrices
    assert len(acov) == p + 1, 'Incorrect number of autocovariance matrices.'

    n = y.shape[1]
    for i in range(len(acov)):
        # Assert that each covariance matrix is of the right shape
        assert np.array_equal(acov[i].shape, (q, q))

        for j in range(q):
            for k in range(q):
                # Assert that each entry is the correct value
                assert np.isclose(
                    acov[i][j, k], np.cov(y[j, :(n - i)], y[k, i:n], bias=True)[0, 1]
                ), 'Incorrect autocovariance value.'


def test_block_toeplitz():
    rng = np.random.default_rng(0)
    q = 5  # mvAR with 5 variables
    p = 4  # up to lag 4

    _, y = generate_mvar_data(q=q, rng=rng)
    acov = autocovariances(y, p, bias=True)

    C = block_toeplitz(acov)

    # Assert that the block-toeplitz matrix is of the right shape
    assert np.array_equal(C.shape, ((p + 1) * q, (p + 1) * q)), 'Incorrect shape of C.'

    # Assert that the block-toeplitz matrix is symmetric
    assert np.allclose(C, C.T), 'C is not symmetric.'  # use np.allclose() due to numerical precision errors in np.cov()

    # Assert that the block-toeplitz matrix is positive definite
    assert np.all(np.linalg.eigvals(C) > 0), 'C is not positive definite.'

    for i in range(p + 1):
        for j in range(i + 1):
            # Assert diagonal and lower triangular matrices
            assert np.array_equal(
                C[(i * q):((i + 1) * q), (j * q):((j + 1) * q)], acov[i-j]
            ), 'Incorrect block-toeplitz matrix in diagonal or lower triangular blocks.'
        for j in range(i + 1, p + 1):
            # Assert upper triangular matrices
            assert np.array_equal(
                C[(i * q):((i + 1) * q), (j * q):((j + 1) * q)], acov[j-i].T,
            ), 'Incorrect block-toeplitz matrix in upper triangular blocks.'


def test_ar_parameters():
    rng = np.random.default_rng(0)
    q = 3  # mvAR with 3 variables
    p = 3  # up to lag 3, same as the generative AR(3) process

    _, y = generate_mvar_data(q=q, rng=rng)
    acov = autocovariances(y, p, bias=True)

    # get "approximate ground truth" based on the first univariate AR(3) variable
    A_hat, Q_hat = ar_parameters(autocovariances(y[0, :][None, :], p, bias=True), R=1.)
    A_hat = np.squeeze(np.concatenate(A_hat))

    A, Q = ar_parameters(acov, R=1.)

    assert len(A) == p, 'Incorrect number of AR parameter matrices.'
    for i in range(p):
        assert np.array_equal(A[i].shape, (q, q)), 'Incorrect shape of AR parameter matrix.'

    a = np.array([[A_k[i, i] for A_k in A] for i in range(q)])  # diagonal elements of A matrices

    assert len(a) == q, 'Incorrect number of mvAR variables.'
    for j in range(q):
        assert np.argmax(a[j]) == np.argmax(A_hat), 'Incorrect leading AR coefficient.'
        for i in range(p):
            assert np.sign(a[j][i]) == np.sign(A_hat[i]), 'Incorrect sign of AR coefficients.'

    assert np.allclose(Q, Q.T), 'Q is not symmetric.'  # use np.allclose() due to numerical precision errors
    assert ((np.abs(np.diag(Q) - Q_hat) / Q_hat) < 0.25).all(), 'Incorrect diagonal elements of Q.'


def test_mvar_ssm():
    rng = np.random.default_rng(0)
    q = 3  # mvAR with 3 variables

    x, y = generate_mvar_data(q=q, rng=rng)
    obs_error = np.mean((x - y) ** 2)

    for p in range(1, 6):
        acov = autocovariances(y, p, bias=True)
        R = 1.
        A, Q = ar_parameters(acov, R=R)

        model = mvar_ssm(y, A, Q, R)
        z = model.kalman_filt_smooth(return_dict=True)

        filtered_x = z["x_t_t"][:q, 1:]
        smoothed_x = z["x_t_n"][:q, 1:]

        filtered_error = np.mean((x - filtered_x) ** 2)
        smoothed_error = np.mean((x - smoothed_x) ** 2)

        assert filtered_error < obs_error, "Filtered error should be less than observation error."
        assert smoothed_error < filtered_error, "Smoothed error should be less than filtered error."


def test_optimize_arp():
    rng = np.random.default_rng(0)
    q = 3  # mvAR with 3 variables
    _, y = generate_mvar_data(q=q, rng=rng)
    bic = []

    for p in range(1, 6):
        A, Q, R = optimize_arp(y, p=p)
        assert R > 0 and not np.isclose(R, 0), 'Observation noise variance R should be positive.'
        assert (np.diag(Q) > 0).all(), 'Diagonal elements of state noise covariance Q should be positive.'
        assert not np.isclose(Q, np.zeros_like(Q)).any(), 'State noise covariance Q should not be zero.'

        model = mvar_ssm(y, A, Q, R)
        z = model.kalman_filt_smooth(return_dict=True)
        bic.append(-2 * z['logL'].sum() + (q * q * (p + 1) + 1) * np.log(y.shape[1]))

    assert np.argmin(bic) == 0, 'Optimal mvAR order should be 1.'


if __name__ == "__main__":
    test_pac_regression()
    test_autocovariances()
    test_block_toeplitz()
    test_ar_parameters()
    test_mvar_ssm()
    test_optimize_arp()
