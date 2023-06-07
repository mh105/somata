import numpy as np
import matplotlib.pyplot as plt
from somata.basic_models import StateSpaceModel as Ssm
from somata.basic_models import OscillatorModel as Osc
from somata.source_loc.source_loc_utils import simulate_oscillation


"""
##########################################################
Example script to visualize the log-likelihood surface
"""

# Oscillator parameters for simulating source activity
Fs = 100  # (Hz) sampling frequency
T = 5  # (s) total duration of simulated activity
# a = 0.98  # (unitless) damping factor, only relevant if using Matsuda oscillator
alpha = 2.02  # corresponding to a=0.98 at 100 Hz sampling rate
a = np.exp(-alpha / Fs)
f = 1  # (Hz) center frequency of oscillation in Hertz
Q = 1  # (Am^2) state noise covariance for the active oscillator only
mu0 = [0, 0]  # (Am) initial state mean for the active oscillator only
Q0 = Q  # (Am^2) initial state variance for the active oscillator only
R = 1  # (V^2) observation noise variance, assuming diagonal covariance matrix with the same noise for each channel

neeg = 1
ntime = int(T * Fs) + 1

# Simulate the hidden state oscillator activity
sim_x = simulate_oscillation(f, a, Q, mu0, Q0, Fs, T)

# Add observation noise
sim_y = sim_x + np.random.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime)[:, 0]

# Visualize the simulated slow wave observation
plt.figure()
# plt.plot(sim_y, label='y')
plt.plot(sim_x, linewidth=0.7, label='x')
plt.legend()

# Decimate by a factor K
K = 10
y_list = [sim_y[k::K] for k in range(K)]

# This is the actual variance Q for the decimated sequences
Q_dec = np.sum([a**(2*k) for k in range(K)])


"""
Look at one parameter at a time
"""
# freq
f_list = np.linspace(start=0.01, stop=2, num=100)
ll_list = np.zeros(len(f_list))
ll_dec_list = np.zeros(len(f_list))
ll_dec_only_1_list = np.zeros(len(f_list))

for f_idx in range(len(f_list)):
    o1 = Osc(a=a, freq=f_list[f_idx], sigma2=Q, y=sim_y, Fs=Fs, R=R)
    em_results = o1.dejong_filt_smooth(EM=True)
    ll_list[f_idx] = em_results['logL'].sum()

    # Let's create an array of Ssm objects
    ssm_array = np.empty(K, dtype=Osc)  # mutable array
    for k in range(K):
        ssm_array[k] = Osc(a=a**K, freq=f_list[f_idx], sigma2=Q_dec, y=y_list[k], Fs=Fs / K, R=R)

    # Run an e-step
    x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_all, *_ = Ssm.par_kalman(ssm_array, method='dejong',
                                                                         skip_check_observed=True)
    logL = float(np.sum([logL.sum() for logL in logL_all]))  # total logL summed across sequences
    ll_dec_list[f_idx] = logL

    # a single decimated sequence
    o2 = Osc(a=a**K, freq=f_list[f_idx], sigma2=Q_dec, y=y_list[0], Fs=Fs / K, R=R)
    em_results = o2.dejong_filt_smooth(EM=True)
    ll_dec_only_1_list[f_idx] = em_results['logL'].sum()

plt.figure()
plt.plot(f_list, ll_list)
plt.axvline(x=f, color='red')
plt.title('(Original) log-likelihood: freq')

plt.figure()
plt.plot(f_list, ll_dec_list)
plt.axvline(x=f, color='red')
plt.title('(Decimated) log-likelihood: freq')

plt.figure()
plt.plot(f_list, ll_dec_list / K, label='average')
plt.plot(f_list, ll_dec_only_1_list, label='1 sequence')
plt.legend()
plt.axvline(x=f, color='red')
plt.title('(Decimated 1sequence) log-likelihood: freq')


# a
a_list = np.logspace(start=np.log10(a-0.01), stop=0, num=100)
ll_list = np.zeros(len(a_list))
ll_dec_list = np.zeros(len(a_list))
ll_dec_only_1_list = np.zeros(len(a_list))

for a_idx in range(len(a_list)):
    o1 = Osc(a=a_list[a_idx], freq=f, sigma2=Q, y=sim_y, Fs=Fs, R=R)
    em_results = o1.dejong_filt_smooth(EM=True)
    ll_list[a_idx] = em_results['logL'].sum()

    # Let's create an array of Ssm objects
    ssm_array = np.empty(K, dtype=Osc)  # mutable array
    for k in range(K):
        ssm_array[k] = Osc(a=a_list[a_idx] ** K, freq=f, sigma2=Q_dec, y=y_list[k], Fs=Fs / K, R=R)

    # Run an e-step
    x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_all, *_ = Ssm.par_kalman(ssm_array, method='dejong',
                                                                         skip_check_observed=True)
    logL = float(np.sum([logL.sum() for logL in logL_all]))  # total logL summed across sequences
    ll_dec_list[a_idx] = logL

    # a single decimated sequence
    o2 = Osc(a=a_list[a_idx] ** K, freq=f, sigma2=Q_dec, y=y_list[0], Fs=Fs / K, R=R)
    em_results = o2.dejong_filt_smooth(EM=True)
    ll_dec_only_1_list[a_idx] = em_results['logL'].sum()

plt.figure()
plt.plot(a_list, ll_list)
plt.axvline(x=a, color='red')
plt.title('(Original) log-likelihood: a')

plt.figure()
plt.plot(a_list, ll_dec_list)
plt.axvline(x=a, color='red')
plt.title('(Decimated) log-likelihood: a')

plt.figure()
plt.plot(a_list, ll_dec_list / K, label='average')
plt.plot(a_list, ll_dec_only_1_list, label='1 sequence')
plt.legend()
plt.axvline(x=a, color='red')
plt.title('(Decimated 1sequence) log-likelihood: a')


# Q
q_list_original = np.linspace(start=Q/2, stop=Q*2, num=100)
q_list = np.linspace(start=2*Q_dec-Q*K, stop=2*K-Q_dec, num=100)
ll_list = np.zeros(len(q_list))
ll_dec_list = np.zeros(len(q_list))
ll_dec_only_1_list = np.zeros(len(q_list))

for q_idx in range(len(q_list)):
    o1 = Osc(a=a, freq=f, sigma2=q_list_original[q_idx], y=sim_y, Fs=Fs, R=R)
    em_results = o1.dejong_filt_smooth(EM=True)
    ll_list[q_idx] = em_results['logL'].sum()

    # Let's create an array of Ssm objects
    ssm_array = np.empty(K, dtype=Osc)  # mutable array
    for k in range(K):
        ssm_array[k] = Osc(a=a ** K, freq=f, sigma2=q_list[q_idx], y=y_list[k], Fs=Fs / K, R=R)

    # Run an e-step
    x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_all, *_ = Ssm.par_kalman(ssm_array, method='dejong',
                                                                         skip_check_observed=True)
    logL = float(np.sum([logL.sum() for logL in logL_all]))  # total logL summed across sequences
    ll_dec_list[q_idx] = logL

    # a single decimated sequence
    o2 = Osc(a=a ** K, freq=f, sigma2=q_list[q_idx], y=y_list[0], Fs=Fs / K, R=R)
    em_results = o2.dejong_filt_smooth(EM=True)
    ll_dec_only_1_list[q_idx] = em_results['logL'].sum()

plt.figure()
plt.plot(q_list, ll_list)
plt.axvline(x=Q, color='red')
plt.title('(Original) log-likelihood: Q')

plt.figure()
plt.plot(q_list, ll_dec_list)
plt.axvline(x=Q_dec, color='red')
plt.title('(Decimated) log-likelihood: Q')

plt.figure()
plt.plot(q_list, ll_dec_list / K, label='average')
plt.plot(q_list, ll_dec_only_1_list, label='1 sequence')
plt.legend()
plt.axvline(x=Q_dec, color='red')
plt.title('(Decimated 1sequence) log-likelihood: Q')


"""
Vary two parameters on a grid
"""
a_list = np.logspace(start=np.log10(a-0.01), stop=0, num=100)
f_list = np.linspace(start=0.01, stop=2, num=100)

ll_mat = np.zeros((len(a_list), len(f_list)))
ll_dec_mat = np.zeros((len(a_list), len(f_list)))
ll_dec_only_1_mat = np.zeros((len(a_list), len(f_list)))

for a_idx in range(len(a_list)):
    print(a_list[a_idx])
    for f_idx in range(len(f_list)):
        o1 = Osc(a=a_list[a_idx], freq=f_list[f_idx], sigma2=Q, y=sim_y, Fs=Fs, R=R)
        em_results = o1.dejong_filt_smooth(EM=True)
        ll_mat[a_idx, f_idx] = em_results['logL'].sum()

        # Let's create an array of Ssm objects
        ssm_array = np.empty(K, dtype=Osc)  # mutable array
        for k in range(K):
            ssm_array[k] = Osc(a=a_list[a_idx] ** K, freq=f_list[f_idx], sigma2=Q_dec, y=y_list[k], Fs=Fs / K, R=R)

        # Run an e-step
        x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_all, *_ = Ssm.par_kalman(ssm_array, method='dejong',
                                                                             skip_check_observed=True)
        logL = float(np.sum([logL.sum() for logL in logL_all]))  # total logL summed across sequences
        ll_dec_mat[a_idx, f_idx] = logL

        # a single decimated sequence
        o2 = Osc(a=a_list[a_idx] ** K, freq=f_list[f_idx], sigma2=Q_dec, y=y_list[0], Fs=Fs / K, R=R)
        em_results = o2.dejong_filt_smooth(EM=True)
        ll_dec_only_1_mat[a_idx, f_idx] = em_results['logL'].sum()

_, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y = np.meshgrid(f_list, a_list)
surf = ax.plot_surface(X, Y, ll_mat)
plt.title('(Original) log-likelihood')
plt.xlabel('freq')
plt.ylabel('a')

_, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y = np.meshgrid(f_list, a_list)
_ = ax.plot_surface(X, Y, ll_dec_mat)
plt.title('(Decimated) log-likelihood')
plt.xlabel('freq')
plt.ylabel('a')

_, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y = np.meshgrid(f_list, a_list)
_ = ax.plot_surface(X, Y, ll_dec_only_1_mat)
plt.title('(Decimated 1sequence) log-likelihood')
plt.xlabel('freq')
plt.ylabel('a')

# generate a figure with normalized log-likelihood to compare among methods
_, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y = np.meshgrid(f_list, a_list)
ax.plot_surface(X, Y, ll_mat / len(sim_y))
surf1 = ax.plot_surface(X, Y, ll_dec_mat / len(sim_y))
surf2 = ax.plot_surface(X, Y, ll_dec_only_1_mat / len(y_list[0]))
plt.xlabel('freq')
plt.ylabel('a')
