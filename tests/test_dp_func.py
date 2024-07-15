"""
Author: Mingjian He <mh1@stanford.edu>

Testing functions for exact inference signal processing methods in somata/exact_inferences/dp_func.py
"""

import numpy as np
from codetiming import Timer
from test_load_data import _load_data  # type: ignore


def test_forward_backward(show_plot=False):
    from somata.exact_inference import forward_backward

    input_data = _load_data("forward_backward_test_data.mat")

    p_ht_v1_t, p_ht_v1_T, p_ht_tmin1_v1_T, logL = forward_backward(A=input_data['A'], py_x=input_data['qt_m'],
                                                                   p1=input_data['p1'], compute_edge=True)

    # Verify that the results are close within tolerance
    assert np.allclose(p_ht_v1_t.flatten(), input_data['filt'].flatten()), 'Filtering results are off.'
    assert np.allclose(p_ht_v1_T.flatten(), input_data['smooth'].flatten()), 'Smoothing results are off.'
    assert np.allclose(p_ht_tmin1_v1_T.flatten(), input_data['edges'].flatten()), 'Edge marginal results are off.'
    assert np.allclose(logL.flatten(), input_data['logL'].flatten()), 'Log likelihood results are off.'

    if show_plot:
        import matplotlib.pyplot as plt

        # Compare the differences in filtered, smoothed, edge marginals, and log likelihood results
        fig, axs = plt.subplots(2, 2, tight_layout=True)

        # filtered
        var_diff = p_ht_v1_t.flatten() - input_data['filt'].flatten()
        axs[0, 0].hist(var_diff)
        axs[0, 0].set_title('Filtered', fontweight="bold")

        # smoothed
        var_diff = p_ht_v1_T.flatten() - input_data['smooth'].flatten()
        axs[0, 1].hist(var_diff)
        axs[0, 1].set_title('Smoothed', fontweight="bold")

        # edge marginals
        var_diff = p_ht_tmin1_v1_T.flatten() - input_data['edges'].flatten()
        axs[1, 0].hist(var_diff)
        axs[1, 0].set_title('Edge Marginals', fontweight="bold")

        # log likelihood
        var_diff = logL.flatten() - input_data['logL'].flatten()
        axs[1, 1].hist(var_diff)
        axs[1, 1].set_title('logL', fontweight="bold")


def test_viterbi():
    from somata.exact_inference import viterbi

    input_data = _load_data("viterbi_test_data.mat")

    viterbi_path, bin_path = viterbi(A=input_data['A'], py_x=input_data['py_x'], p1=input_data['p1'])

    # Verify that the results are identical
    assert (viterbi_path + 1 == input_data['viterbi_path']).all(), 'Viterbi path results are off.'
    assert (bin_path == input_data['bin_path']).all(), 'Binary path results are off.'


def test_kalman(show_plot=False):
    from somata.exact_inference import kalman

    input_data = _load_data("kalman_inputs.mat")

    # Run the python kalman function on the given inputs
    with Timer():
        x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, x_t_tmin1, P_t_tmin1, fy_t_interp = kalman(**input_data)

    # Compare with MATLAB results
    output_data = _load_data("kalman_outputs.mat")

    # Verify that the results are close within tolerance
    for elem in output_data.keys():
        py_var = locals()[elem]
        ml_var = output_data[elem]
        if elem == 'fy_t_interp':
            assert np.isnan(py_var), "NaN fy_t_interp failed in Python."
            assert np.isnan(ml_var), "Nan fy_t_interp failed in MATLAB."
        else:
            assert np.allclose(py_var.flatten(), ml_var.flatten()), 'Results of kalman differ from MATLAB outputs.'

    if show_plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(5, 2, tight_layout=True)
        fig.suptitle('kalman', y=0.995, fontweight="bold", fontsize=16)
        ax_num = 0

        for elem in output_data.keys():
            py_var = locals()[elem]
            ml_var = output_data[elem]

            axs_index = np.unravel_index(ax_num, axs.shape, 'F')

            if elem == 'fy_t_interp':
                assert np.isnan(py_var), "NaN fy_t_interp failed in Python."
                assert np.isnan(ml_var), "Nan fy_t_interp failed in MATLAB."
                axs[axs_index].axvline(x=0, color='b', linestyle='--', linewidth=4)
                axs[axs_index].set_xlim([-1, 1])
            else:
                assert py_var.shape == ml_var.shape, "Dimensions don't match between MATLAB and Python outputs"
                if (py_var == ml_var).all():
                    axs[axs_index].axvline(x=0, color='b', linestyle='--', linewidth=4)
                    axs[axs_index].set_xlim([-1, 1])
                else:
                    var_diff = py_var.flatten() - ml_var.flatten()
                    axs[axs_index].hist(var_diff)

            axs[axs_index].set_title(elem, fontweight="bold")
            ax_num += 1


def test_djkalman(show_plot=False):
    from somata.exact_inference import djkalman

    input_data = _load_data("kalman_inputs.mat")

    # Run the python djkalman function on the given inputs
    with Timer():
        x_t_n, P_t_n, P_t_tmin1_n, logL, x_t_t, P_t_t, K_t, \
            x_t_tmin1, P_t_tmin1, fy_t_interp = djkalman(**input_data, skip_interp=False)

    # Compare with MATLAB results
    output_data = _load_data("djkalman_outputs.mat")

    # Verify that the results are close within tolerance
    for elem in output_data.keys():
        assert np.allclose(locals()[elem].flatten(), output_data[elem].flatten()), \
            'Results of djkalman differ from MATLAB outputs.'

    if show_plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(5, 2, tight_layout=True)
        fig.suptitle('djkalman', y=0.995, fontweight="bold", fontsize=16)
        ax_num = 0

        for elem in output_data.keys():
            py_var = locals()[elem]
            ml_var = output_data[elem]

            axs_index = np.unravel_index(ax_num, axs.shape, 'F')

            assert py_var.shape == ml_var.shape, "Dimensions don't match between MATLAB and Python outputs."
            if (py_var == ml_var).all():
                axs[axs_index].axvline(x=0, color='b', linestyle='--', linewidth=4)
                axs[axs_index].set_xlim([-1, 1])
            else:
                var_diff = py_var.flatten() - ml_var.flatten()
                axs[axs_index].hist(var_diff)

            axs[axs_index].set_title(elem, fontweight="bold")
            ax_num += 1


def test_inverse(dim=128):
    from somata.exact_inference import inverse

    np.random.seed(1)

    # Test with a small symmetric positive definite matrix
    A = np.random.rand(dim, dim)
    A = A @ A.T + np.eye(dim)
    A_np_inv = np.linalg.inv(A)

    assert np.allclose(A_np_inv.flatten(), inverse(A, approach='gaussian').flatten()), 'Gaussian inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse(A, approach='cholesky').flatten()), 'Cholesky inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse(A, approach='qr').flatten()), 'QR inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse(A, approach='svd').flatten()), 'SVD inversion is off.'

    # Test with a diagonal matrix
    A = np.eye(dim) * np.random.rand(dim)
    A_np_inv = np.linalg.inv(A)

    assert np.allclose(A_np_inv.flatten(), inverse(A, approach='gaussian').flatten()), 'Gaussian inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse(A, approach='cholesky').flatten()), 'Cholesky inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse(A, approach='qr').flatten()), 'QR inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse(A, approach='svd').flatten()), 'SVD inversion is off.'

    # Test with a singular diagonal matrix
    A = np.eye(dim) * np.random.rand(dim)
    A_man_inv = 1.0 / np.diagonal(A)
    A[-1, -1] = 0.0
    A_man_inv[-1] = 0.0
    A_man_inv = np.eye(dim) * A_man_inv

    assert np.allclose(A_man_inv.flatten(), inverse(A, approach='svd').flatten()), 'SVD inversion is off.'
    assert A.dtype == inverse(A, approach='svd').dtype, 'SVD inversion changed the dtype.'


def test_inverse_torch(dim=128, atol=1e-3):
    from somata.exact_inference import inverse_torch
    import torch

    np.random.seed(1)

    # Test with a symmetric positive definite matrix
    A = np.random.rand(dim, dim)
    A = A @ A.T + np.eye(dim)
    A_np_inv = np.linalg.inv(A)
    A = torch.as_tensor(data=A, dtype=torch.float32)

    assert np.allclose(A_np_inv.flatten(), inverse_torch(A, approach='gaussian').cpu().numpy().flatten(),
                       atol=atol), 'Gaussian inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse_torch(A, approach='cholesky').cpu().numpy().flatten(),
                       atol=atol), 'Cholesky inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse_torch(A, approach='qr').cpu().numpy().flatten(),
                       atol=atol), 'QR inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse_torch(A, approach='svd').cpu().numpy().flatten(),
                       atol=atol), 'SVD inversion is off.'

    # Test with a diagonal matrix
    A = np.eye(dim) * np.random.rand(dim)
    A_np_inv = np.linalg.inv(A)
    A = torch.as_tensor(data=A, dtype=torch.float32)

    assert np.allclose(A_np_inv.flatten(), inverse_torch(A, approach='gaussian').cpu().numpy().flatten(),
                       atol=atol), 'Gaussian inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse_torch(A, approach='cholesky').cpu().numpy().flatten(),
                       atol=atol), 'Cholesky inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse_torch(A, approach='qr').cpu().numpy().flatten(),
                       atol=atol), 'QR inversion is off.'
    assert np.allclose(A_np_inv.flatten(), inverse_torch(A, approach='svd').cpu().numpy().flatten(),
                       atol=atol), 'SVD inversion is off.'

    # Test with a singular diagonal matrix
    A = np.eye(dim) * np.random.rand(dim)
    A_man_inv = 1.0 / np.diagonal(A)
    A[-1, -1] = 0.0
    A_man_inv[-1] = 0.0
    A_man_inv = np.eye(dim) * A_man_inv
    A = torch.as_tensor(data=A, dtype=torch.float32)

    assert np.allclose(A_man_inv.flatten(), inverse_torch(A, approach='svd').cpu().numpy().flatten(),
                       atol=atol), 'SVD inversion is off.'
    assert A.dtype == inverse_torch(A, approach='svd').dtype, 'SVD inversion changed the dtype.'


def test_logdet():
    from somata.exact_inference import logdet

    np.random.seed(1)

    for ii in range(8):
        dim = 2 ** ii

        # Test with a symmetric positive semi-definite matrix
        A = np.random.rand(dim, dim)
        A = A @ A.T
        assert np.allclose(np.log(np.linalg.det(A)), logdet(A)
                           ), 'Log determinant results are off for positive semi-definite matrices.'

        # Test with a diagonal matrix
        A = np.eye(dim) * np.random.rand(dim)
        assert np.allclose(np.log(np.linalg.det(A)), logdet(A)
                           ), 'Log determinant results are off for diagonal matrices.'


def test_logdet_torch(rtol=1e-3, atol=1e-3):
    from somata.exact_inference import logdet_torch
    import torch

    np.random.seed(1)

    for ii in range(8):
        dim = 2 ** ii

        # Test with a symmetric positive semi-definite matrix
        A = np.random.rand(dim, dim)
        A = A @ A.T
        numpy_logdet = np.log(np.linalg.det(A))
        A = torch.as_tensor(data=A, dtype=torch.float32)
        assert np.allclose(numpy_logdet, logdet_torch(A).cpu().numpy(),
                           rtol=rtol, atol=atol), 'Log determinant results are off for positive semi-definite matrices.'

        # Test with a diagonal matrix
        A = np.eye(dim) * np.random.rand(dim)
        numpy_logdet = np.log(np.linalg.det(A))
        A = torch.as_tensor(data=A, dtype=torch.float32)
        assert np.allclose(numpy_logdet, logdet_torch(A).cpu().numpy(),
                           rtol=rtol, atol=atol), 'Log determinant results are off for diagonal matrices.'


if __name__ == "__main__":
    test_forward_backward()
    test_viterbi()
    test_kalman()
    test_djkalman()
    test_inverse()
    test_inverse_torch()
    test_logdet()
    test_logdet_torch()
    print('DP function tests finished without exception.')
