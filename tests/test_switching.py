"""
Author: Mingjian He <mh1@stanford.edu>

Testing functions for switching methods in somata/switching
"""

import numpy as np
from test_load_data import _load_data  # type: ignore


def test_switching():
    from somata.switching import switching
    from somata import OscillatorModel as Osc

    input_data = _load_data("switching_input.mat")

    o1 = Osc(**input_data, G=([1, 0, 1, 0], [1, 0, 0, 0]), Fs=100)
    print(o1)

    method_list = ('static', 'gpb1', 'gpb2', 'imm', '1991',
                   'a parp', 'ab parp', 'a pari', 'ab pari', 'a pars', 'ab pars')
    for method in method_list:
        Mprob, fy_t = switching(o1, method=method)
        mat_results = _load_data(method + '_results.mat')

        # Verify that the results are close within tolerance
        assert np.allclose(Mprob.flatten(), mat_results['Mprob'].flatten()), method + 'Mprob results are off.'
        assert np.allclose(fy_t.flatten(), mat_results['fy_t'].flatten()), method + 'fy_t results are off.'

    # Test 1991 with 5 future steps of pseudo-smoothing
    Mprob, fy_t = switching(o1, method='1991', future_steps=5)
    mat_results = _load_data('1991_future5_results.mat')
    assert np.allclose(Mprob.flatten(), mat_results['Mprob'].flatten()), '1991 ps=5 Mprob results are off.'
    assert np.allclose(fy_t.flatten(), mat_results['fy_t'].flatten()), '1991 ps=5 fy_t results are off.'


def test_vb_learn_original():
    from somata.switching import VBSwitchModel as Vbs
    from somata import OscillatorModel as Osc

    input_data = _load_data("switching_input.mat")

    o1 = Osc(**input_data, G=([1, 0, 1, 0], [1, 0, 0, 0]), Fs=100)
    print(o1)

    # SOMATA has better estimations of mu0 and S0, so remove them from EM updates when comparing with MATLAB results
    vb_model = Vbs(o1)
    print(vb_model)
    vb_results = vb_model.learn(keep_param=('mu0', 'S0'), return_dict=True, original=True)
    mat_results = _load_data("vb_learn_original_results.mat")

    # Verify that the results are close within tolerance
    assert np.allclose(vb_results['h_t_m'].flatten(), mat_results['Mprob'].flatten()), \
        'Results of h_t_m are different from MATLAB outputs.'
    assert np.allclose(vb_results['q_t_m'].flatten(), mat_results['fy_t'].flatten()), \
        'Results of q_t_m are different from MATLAB outputs.'
    assert np.allclose(vb_results['A'].flatten(), mat_results['A'].flatten()), \
        'Results of A are different from MATLAB outputs.'
    assert np.allclose(vb_results['VB_iter'], mat_results['VB_iter'].flatten()), \
        'Different numbers of VB iterations were executed from MATLAB outputs.'
    assert np.allclose([x[-1] for x in vb_results['logL_bound']], mat_results['logL_bound'].flatten()), \
        'Results of logL_bound are different from MATLAB outputs.'


def test_vb_learn():
    from somata.switching import VBSwitchModel as Vbs
    from somata import OscillatorModel as Osc

    input_data = _load_data("switching_input.mat")

    o1 = Osc(**input_data, G=([1, 0, 1, 0], [1, 0, 0, 0]), Fs=100)
    print(o1)

    # SOMATA has better estimations of mu0 and S0, so remove them from EM updates when comparing with MATLAB results
    vb_model = Vbs(o1)
    print(vb_model)
    vb_results = vb_model.learn(keep_param=('mu0', 'S0'), shared_R=True, shared_comp=[0, 0],
                                normalize_q_t=True, return_dict=True)
    mat_results = _load_data("vb_learn_results.mat")

    # Verify that the results are close within tolerance
    assert np.allclose(vb_results['h_t_m'].flatten(), mat_results['Mprob'].flatten()), \
        'Results of h_t_m are different from MATLAB outputs.'
    assert np.allclose(vb_results['h_t_m_soft'].flatten(), mat_results['Mprob_soft'].flatten()), \
        'Results of h_t_m_soft are different from MATLAB outputs.'
    assert np.allclose(vb_results['h_t_m_hard'].flatten(), mat_results['Mprob_hard'].flatten()), \
        'Results of h_t_m_hard are different from MATLAB outputs.'
    assert np.allclose(vb_results['q_t_m'].flatten(), mat_results['fy_t'].flatten()), \
        'Results of q_t_m are different from MATLAB outputs.'
    assert np.allclose(vb_results['A'].flatten(), mat_results['A'].flatten()), \
        'Results of A are different from MATLAB outputs.'
    assert np.allclose(vb_results['VB_iter'], mat_results['VB_iter'].flatten()), \
        'Different numbers of VB iterations were executed from MATLAB outputs.'
    assert np.allclose([x[-1] for x in vb_results['logL_bound']], mat_results['logL_bound'].flatten()), \
        'Results of logL_bound are different from MATLAB outputs.'


if __name__ == "__main__":
    test_switching()
    test_vb_learn_original()
    test_vb_learn()
    print('Switching inference tests finished without exception.')
