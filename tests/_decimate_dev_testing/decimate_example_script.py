import itertools
import numpy as np
import matplotlib.pyplot as plt
from somata.basic_models import StateSpaceModel as Ssm
from somata.basic_models import OscillatorModel as Osc
from somata.source_loc.source_loc_utils import simulate_oscillation
from somata.pac.decimate import DecimatedModel as Dec


"""
##########################################################
Example script to do decimated oscillator learning
"""

# Oscillator parameters for simulating source activity
Fs = 100  # (Hz) sampling frequency
T = 10  # (s) total duration of simulated activity
a = 0.98  # (unitless) damping factor, only relevant if using Matsuda oscillator
f = 1  # (Hz) center frequency of oscillation in Hertz
Q = 1  # (Am^2) state noise covariance for the active oscillator only
mu0 = [0, 0]  # (Am) initial state mean for the active oscillator only
S0 = Q  # (Am^2) initial state variance for the active oscillator only
R = 1  # (V^2) observation noise variance, assuming diagonal covariance matrix with the same noise for each channel

neeg = 1
ntime = T * Fs + 1

# Simulate the hidden state oscillator activity
sim_x = simulate_oscillation(f, a, Q, mu0, S0, Fs, T)

# Add observation noise
sim_y = sim_x + np.random.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime)[:, 0]

# Visualize the simulated slow wave observation
plt.figure()
plt.plot(sim_y, label='y')
plt.plot(sim_x, linewidth=0.7, label='x')
plt.legend()

""" Now we do oscillator learning at the original Fs """
o0 = Osc(a=a, freq=f, sigma2=Q, y=sim_y, Fs=Fs, R=R)
print(o0)

o1 = o0.copy()
for x in range(100):
    _ = o1.m_estimate(**o1.dejong_filt_smooth(EM=True))

print(o1)

""" Now we can decimate by a factor K """
K = 10
y_list = [sim_y[k::K] for k in range(K)]

plt.figure()
plt.plot(y_list[0], '-o')

# The tricky part is how to initialize the state noise variance
o2 = Osc(a=a**K, freq=f, sigma2=Q*K**2, Fs=Fs/K, R=R)

# This is the actual variance Q
Q_dec = [np.linalg.matrix_power(o0.F, k[0]) @ o0.Q @ np.linalg.matrix_power(o0.F, k[1]) for k
         in itertools.product(range(K), range(K))]
np.sum(Q_dec, 0)

# Let's create an array of Ssm objects
ssm_array = np.empty(K, dtype=Osc)  # mutable array
for k in range(K):
    ssm_array[k] = Osc(a=a**K, freq=f, sigma2=Q*K**2, y=y_list[k], Fs=Fs/K, R=R)

d1 = Dec(ssm_array=ssm_array)
em_log = d1.learn()

print(d1.ssm_array[0])

""" Let's compare the log likelihoods """
original_e_results = o1.dejong_filt_smooth(EM=True)
print('Original logL = ' + str(original_e_results['logL'].sum()))

x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_all, *_ = Ssm.par_kalman(d1.ssm_array, method='dejong',
                                                                     skip_check_observed=True)
print('Decimated logL = ' + str(float(np.sum([logL.sum() for logL in logL_all]))))
