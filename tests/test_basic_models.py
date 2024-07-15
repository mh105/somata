"""
Author: Mingjian He <mh1@stanford.edu>

Testing functions for basic state-space models in somata/basic_models
"""

from codetiming import Timer
from test_load_data import _load_data  # type: ignore

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # suppress a duplicate OpenMP library error with pytorch on Mac osx-64


def test_ssm():
    from somata.basic_models import StateSpaceModel as Ssm

    input_data = _load_data("kalman_inputs.mat")
    input_data.pop('R_weights')
    s1 = Ssm(components='Osc', **input_data)
    print(s1)

    with Timer():
        _ = s1.kalman_filt_smooth()
        _ = s1.dejong_filt_smooth()
        _ = Ssm.par_kalman(s1, method='kalman')
        _ = Ssm.par_kalman(s1, method='dejong')
        s1.m_estimate(**s1.dejong_filt_smooth(EM=True))

    _ = Ssm()


def test_gen():
    from somata.basic_models import GeneralSSModel as Gen

    input_data = _load_data("kalman_inputs.mat")
    input_data.pop('R_weights')
    g1 = Gen(**input_data)
    print(g1)

    with Timer():
        _ = g1.kalman_filt_smooth()
        _ = g1.dejong_filt_smooth()
        _ = Gen.par_kalman(g1, method='kalman')
        _ = Gen.par_kalman(g1, method='dejong')
        g1.m_estimate(**g1.dejong_filt_smooth(EM=True))

    _ = Gen()


def test_osc():
    from somata.basic_models import OscillatorModel as Osc

    input_data = _load_data("kalman_inputs.mat")
    input_data.pop('R_weights')
    o1 = Osc(**input_data, Fs=100)
    print(o1)

    with Timer():
        _ = o1.kalman_filt_smooth()
        _ = o1.dejong_filt_smooth()
        _ = Osc.par_kalman(o1, method='kalman')
        _ = Osc.par_kalman(o1, method='dejong')
        o1.m_estimate(**o1.dejong_filt_smooth(EM=True))

    _ = Osc()
    o2 = Osc(freq=[1, 10, 15], Fs=100, add_dc=True)
    o1.append(o2)
    _ = Osc(freq=5, Fs=80)
    _ = Osc(w=0.1)
    _ = Osc(add_dc=True)


def test_arn():
    from somata.basic_models import AutoRegModel as Arn

    input_data = _load_data("kalman_inputs.mat")
    input_data.pop('R_weights')
    a1 = Arn(coeff=0.95, R=input_data['R'], y=input_data['y'])
    print(a1)

    with Timer():
        _ = a1.kalman_filt_smooth()
        _ = a1.dejong_filt_smooth()
        _ = Arn.par_kalman(a1, method='kalman')
        _ = Arn.par_kalman(a1, method='dejong')
        a1.m_estimate(**a1.dejong_filt_smooth(EM=True))

    _ = Arn()
    a2 = Arn(coeff=[0.99, 0.5, 0.1])
    a1.append(a2)
    _ = Arn(F=a1.F)


if __name__ == "__main__":
    test_ssm()
    test_gen()
    test_osc()
    test_arn()
    print('SOMATA class tests finished without exception.')
