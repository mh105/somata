# Author: Mingjian He <mh1@stanford.edu>
""" Demo of SOMATA (basic_models) """

# noinspection PyProtectedMember
from tests.test_load_data import _load_data
from somata.basic_models import StateSpaceModel as Ssm
from somata.basic_models import GeneralSSModel as Gen
from somata.basic_models import OscillatorModel as Osc
from somata.basic_models import AutoRegModel as Arn
import numpy as np
from codetiming import Timer

# Load an example data
input_data = _load_data("kalman_inputs.mat")
_ = input_data.pop('R_weights')  # remove extra variable
#
#
# ####################################################
#
#
# WHAT ARE SOMATA BASIC MODELS?
# They are a set of useful objects for you to play with.
s1 = Ssm()
s1
print(s1)

# Create a Ssm object with parameters
input_data
s1 = Ssm(**input_data)
print(s1)

# Kalman filtering and smoothing
results = s1.kalman_filt_smooth()

kalman_results = s1.dejong_filt_smooth(return_dict=True)
_ = [print(x) for x in kalman_results.keys()]

short_kalman_results = s1.dejong_filt_smooth(EM=True)
_ = [print(x) for x in short_kalman_results.keys()]

# What if we want to try some different parameters
# in parallel? (For switching, maybe?)
s1 = Ssm(**input_data)
print(s1)
s1.R

s2 = s1.copy()
s2.R[0, 0] *= 2
s2.R

s3 = s1 + s2
print(s3)
s3.R

s1.Q
s4 = s1.copy()
s4.Q[2:, 2:] *= 2
s4.Q
s5 = s1 + s4
print(s5)

print(s3)

s6 = s3 * s5
print(s6)
s6.R

with Timer():
    par_kalman_results = Ssm.par_kalman(
        s6, method='dejong', return_dict=True)
_ = [print(x) for x in par_kalman_results.keys()]
len(par_kalman_results['x_t_n_all'])
type(par_kalman_results['x_t_n_all'])
par_kalman_results['x_t_n_all'][0].shape

#
#
#
#
#
#
#
#
# E step is easy, show me the M step
# (Old buddy ssp_decomp, maybe?)
g1 = Gen(**input_data)
g1
print(g1)
_ = g1.dejong_filt_smooth()

g1.R
g1.m_estimate(**g1.dejong_filt_smooth(EM=True))
g1.R

# Do 10 EM iterations
_ = [g1.m_estimate(**g1.dejong_filt_smooth(EM=True))
     for x in range(10)]

#
#
#
#
#
# How about oscillator models?
s1
s1 = Ssm(components='Osc', **input_data)
s1
print(s1)
with Timer():
    _ = s1.kalman_filt_smooth()
    _ = s1.dejong_filt_smooth()

# Do EM following oscillator update equations
s1.m_estimate(**s1.dejong_filt_smooth(EM=True))

#
#
#
#
#
# Oscillators are special enough to have a class
# and additional perks
o1 = Osc(**input_data)
o1
print(o1)

o1 = Osc(**input_data, Fs=100)
print(o1)

# In case you are wondering, of course yes
o1.m_estimate(**o1.kalman_filt_smooth(EM=True))
o1.m_estimate(**o1.dejong_filt_smooth(EM=True))
print(o1)

# We could do MAP estimation as well
priors = o1.initialize_priors()
priors
o1.m_estimate(**o1.dejong_filt_smooth(EM=True), priors=priors)
_ = [o1.m_estimate(**o1.dejong_filt_smooth(EM=True),
                   priors=priors)
     for x in range(50)]
print(o1)
#
#
#
#
#
# bona fide append()
o2 = Osc(freq=5, Fs=100)
print(o2)
o1.append(o2)
print(o1)

# Yes it still works
o1.m_estimate(**o1.dejong_filt_smooth(EM=True))
print(o1)

# maybe we want to add a DC oscillator?
o3 = Osc(freq=20, Fs=100, add_dc=True)
print(o3)
o4 = Osc(freq=[1, 10, 15], Fs=100, add_dc=True)
print(o4)

o3.append(o4)
print(o3)

# maybe we just want a DC oscillator?
o5 = Osc(add_dc=True)
print(o5)

# Of course EM still works
o5.R = np.array([[0.5]])
o5.m_estimate(y=input_data['y'],
              **o5.dejong_filt_smooth(y=input_data['y'], EM=True),
              keep_param=('R',))
print(o5)

#
#
#
#
#
#
# AR models are fully supported as well
a1 = Arn(coeff=0.95, R=input_data['R'], y=input_data['y'])
print(a1)

with Timer():
    _ = a1.kalman_filt_smooth()
    _ = a1.dejong_filt_smooth()
    _ = Arn.par_kalman(a1, method='kalman')
    _ = Arn.par_kalman(a1, method='dejong')
    a1.m_estimate(**a1.dejong_filt_smooth(EM=True))

a2 = Arn(coeff=[0.99, 0.5, 0.1])
print(a2)
a1.append(a2)
print(a1)

# Lastly, test functions are important
exec(open('tests/test_basic_models.py').read())
