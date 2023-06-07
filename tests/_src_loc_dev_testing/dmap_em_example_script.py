import torch
import matplotlib.pyplot as plt
import mne
import numpy as np
from codetiming import Timer
from somata import OscillatorModel as Osc
from somata.source_loc import SourceLocModel as Src
from somata.source_loc.source_loc_utils import simulate_oscillation


"""
##########################################################
Example script to run one dynamic source localization
"""
# Check whether CUDA is available
print('CUDA available: ' + str(torch.cuda.is_available()))

# Check CUDA floating point computation precision
print('CUDA matmul allow_tf32: ' + str(torch.backends.cuda.matmul.allow_tf32))
print('CUDA cudnn allow_tf32: ' + str(torch.backends.cudnn.allow_tf32))

# Load forward model G
fwd = mne.read_forward_solution('tests/_src_loc_dev_testing/eeganes02-neeg64-fwd.fif')
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)
G = fwd['sol']['data']

# Oscillator parameters for simulating source activity
simulation_mode = 'sinusoid'  # (oscillator or sinusoid) used for simulating source activity
Fs = 100  # (Hz) sampling frequency
T = 10  # (s) total duration of simulated activity
a = 0.98  # (unitless) damping factor, only relevant if using Matsuda oscillator
f = 10  # (Hz) center frequency of oscillation in Hertz
Q = 0  # (Am^2) state noise covariance for the active oscillator only
mu0 = [0, 0]  # (Am) initial state mean for the active oscillator only
Q0 = Q  # (Am^2) initial state variance for the active oscillator only
R = 1  # (V^2) observation noise variance, assuming diagonal covariance matrix with the same noise for each channel

# Assume a noiseless background of source activity
neeg, nsources = G.shape  # (64,1162)
ntime = T * Fs + 1
x_blank = np.zeros((G.shape[1], ntime))

# Create an average referencing matrix
h = np.ones((neeg, 1), dtype=np.float64)
H = np.eye(neeg, dtype=np.float64) - (h @ h.T) / (h.T @ h)

# Define maximal number of EM steps
max_iter = 10

# Simulate source activity in one source at a time
vidx = 0
with Timer():
    print('vertex' + str(vidx))

    # Simulate the source activity in a single source point
    simulated_src = simulate_oscillation(f, a, Q, mu0, Q0, Fs, T, oscillation_type=simulation_mode)

    # Place simulated_src in the correct row of x that correspond to the activated source/vertex index
    x = np.copy(x_blank)
    x[vidx, :] += simulated_src

    # Multiply by fwd model to get EEG scalp activity and add observation noise
    y = H @ (G @ x + np.random.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime).T)

    # Dynamic source localization
    components = Osc(a=0.999, freq=f, Fs=Fs)
    src1 = Src(components=components, fwd=fwd)
    x_t_n, P_t_n = src1.learn(y=y, max_iter=max_iter)

    plt.figure(); plt.plot(x[vidx, :]); plt.title('True source')
    plt.figure(); plt.plot(x_t_n[vidx*2, :]); plt.title('estimated activity')
    plt.figure(); plt.plot(x_t_n[:, 4]); plt.title('all source activity')
