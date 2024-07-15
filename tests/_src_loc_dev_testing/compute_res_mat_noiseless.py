import torch
import matplotlib; matplotlib.use('TkAgg')  # noqa: E702
import matplotlib.pyplot as plt
import mne
import pickle
import numpy as np
from codetiming import Timer
from somata import OscillatorModel as Osc
from somata.source_loc import SourceLocModel as Src
from somata.source_loc.source_loc_utils import simulate_oscillation


"""
##########################################################
The plan for computing the empirical resolution matrix is as following:
    [Step 1] Define the parameters for a single oscillator, default here is a 10 Hz alpha oscillation
    [Step 2] Generate a source activity using the above parameters for this oscillator, put it in to one source
    [Step 3] Project the source activity onto scalp EEG by passing through the forward model gain matrix
    [Step 4] Perform dynamic source localization on scalp EEG and estimate source location using noise covariance (??)
    [Step 5] Repeat steps 1-4 by iterating through sources and construct the resolution matrix
"""
# Check whether CUDA is available
print('CUDA available: ' + str(torch.cuda.is_available()))

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
S0 = Q  # (Am^2) initial state variance for the active oscillator only
R = 0  # (V^2) observation noise variance, assuming diagonal covariance matrix with the same noise for each channel

# Assume a noiseless background of source activity for resolution matrix calculation
neeg, nsources = G.shape  # (64,1162)
ntime = T * Fs + 1
x_blank = np.zeros((G.shape[1], ntime))

res_mat = np.zeros((nsources, nsources), dtype=np.float64)
em_iters = np.zeros(nsources, dtype=np.float64)
max_iter = 10

# Simulate source activity in one source at a time
for vidx in range(nsources):  # Note vidx is a single source/vertex
    with Timer():
        print('vertex' + str(vidx))

        # Simulate the source activity in a single source point
        simulated_src = simulate_oscillation(f, a, Q, mu0, S0, Fs, T, oscillation_type=simulation_mode)

        # Place simulated_src in the correct row of x that correspond to the activated source/vertex index
        x = np.copy(x_blank)
        x[vidx, :] += simulated_src

        # Multiply by fwd model to get EEG scalp activity and add observation noise
        y = G @ x + np.random.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime).T

        # Dynamic source localization
        components = Osc(a=0.999, freq=f, Fs=Fs)
        src1 = Src(components=components, fwd=fwd, d1=0, d2=0, m1=1, m2=0)
        x_t_n, P_t_n = src1.learn(y=y, R=0.01, max_iter=max_iter)

        # Store the hidden state estimates in resolution matrix
        res_mat[:, vidx] = x_t_n[:, 100:-100].max(axis=1)[0:-1:2]  # cutoff beginning and end

        # Record how many EM iterations successfully completed
        em_iters[vidx] = src1.em_log['em_iter']

        """
        plt.figure(); plt.plot(x[vidx, :]); plt.title('True source')
        plt.figure(); plt.plot(x_t_n[vidx*2, :]); plt.title('estimated activity')
        plt.figure(); plt.plot(x_t_n[:, 4]); plt.title('all source activity')
        """

# Save the first pass results
with open('res_mat_raw.pickle', 'wb') as openfile:
    pickle.dump((res_mat, em_iters), openfile)

#
#
#
#
#
# Postprocess the resolution matrix and re-run vertices that reached numerical precision limit
outlier_threshold = 1
nan_vidx = np.where([np.isnan(res_mat[0, x]) for x in range(res_mat.shape[1])])[0]
out_vidx = np.where([np.max(res_mat[:, x]) > outlier_threshold for x in range(res_mat.shape[1])])[0]
rerun_vidx = np.hstack([nan_vidx, out_vidx])

while len(rerun_vidx) > 0:
    print('Number of rerun vertices: ' + str(len(rerun_vidx)))

    for vidx in rerun_vidx:
        with Timer():
            print('vertex' + str(vidx))

            # Simulate the source activity in a single source point
            simulated_src = simulate_oscillation(f, a, Q, mu0, S0, Fs, T, oscillation_type=simulation_mode)

            # Place simulated_src in the correct row of x that correspond to the activated source/vertex index
            x = np.copy(x_blank)
            x[vidx, :] += simulated_src

            # Multiply by fwd model to get EEG scalp activity and add observation noise
            y = G @ x + np.random.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime).T

            # Dynamic source localization
            components = Osc(a=0.999, freq=f, Fs=Fs)
            src1 = Src(components=components, fwd=fwd, d1=0, d2=0, m1=1, m2=0)
            x_t_n, P_t_n = src1.learn(y=y, R=0.01, max_iter=em_iters[vidx] - 1)

            # Update the resolution matrix column and EM iterations for the rerun vertex
            res_mat[:, vidx] = x_t_n[:, 100:-100].max(axis=1)[0:-1:2]
            em_iters[vidx] = src1.em_log['em_iter']

    nan_vidx = np.where([np.isnan(res_mat[0, x]) for x in range(res_mat.shape[1])])[0]
    out_vidx = np.where([np.max(res_mat[:, x]) > outlier_threshold for x in range(res_mat.shape[1])])[0]
    rerun_vidx = np.hstack([nan_vidx, out_vidx])

# Save the post-processed results
with open('res_mat_processed.pickle', 'wb') as openfile:
    pickle.dump((res_mat, em_iters), openfile)

###
# Visualize the resolution matrix
with open('res_mat_processed.pickle', 'rb') as openfile:
    res_mat, em_iters = pickle.load(openfile)

plt.figure()
plt.imshow(res_mat, vmin=0, vmax=1, interpolation='none')
plt.xlabel('# Source Vertex', fontsize=16)
plt.ylabel('# Source Vertex', fontsize=16)
plt.title('SOMATA Resolution Matrix', fontsize=20)
plt.colorbar()
plt.savefig('10Hz_noiseless_resolution_matrix.svg')
plt.close()

#
#
#
#
#
# Find the vertices that are not well recovered
bad_thresh = 0.7
print(np.where(np.diag(res_mat) < bad_thresh)[0])
print(len(np.where(np.diag(res_mat) < bad_thresh)[0]))

# Try to re-process the missed sources with non-diagonal Q and F
outlier_threshold = 1
rerun_vidx = np.where(np.diag(res_mat) < bad_thresh)[0]
em_iters[rerun_vidx] = 31  # reset their EM iterations to a higher value

while len(rerun_vidx) > 0:
    print('Number of rerun vertices: ' + str(len(rerun_vidx)))

    for vidx in rerun_vidx:
        with Timer():
            print('vertex' + str(vidx))

            # Simulate the source activity in a single source point
            simulated_src = simulate_oscillation(f, a, Q, mu0, S0, Fs, T, oscillation_type=simulation_mode)

            # Place simulated_src in the correct row of x that correspond to the activated source/vertex index
            x = np.copy(x_blank)
            x[vidx, :] += simulated_src

            # Multiply by fwd model to get EEG scalp activity and add observation noise
            y = G @ x + np.random.multivariate_normal(np.zeros(neeg), R * np.eye(neeg, neeg), ntime).T

            # Dynamic source localization
            components = Osc(a=0.999, freq=f, Fs=Fs)
            src1 = Src(components=components, fwd=fwd, d1=0.5, d2=0.25, m1=0.95, m2=0.05)
            x_t_n, P_t_n = src1.learn(y=y, R=0.01, max_iter=em_iters[vidx] - 1)

            # Update the resolution matrix column and EM iterations for the rerun vertex
            res_mat[:, vidx] = x_t_n[:, 100:-100].max(axis=1)[0:-1:2]
            em_iters[vidx] = src1.em_log['em_iter']

    nan_vidx = np.where([np.isnan(res_mat[0, x]) for x in range(res_mat.shape[1])])[0]
    out_vidx = np.where([np.max(res_mat[:, x]) > outlier_threshold for x in range(res_mat.shape[1])])[0]
    rerun_vidx = np.hstack([nan_vidx, out_vidx])

with open('res_mat_processed_followed_by_ndqf095.pickle', 'wb') as openfile:
    pickle.dump((res_mat, em_iters), openfile)

#
#
#
#
#
# Visualize the resolution matrix again
with open('res_mat_processed_followed_by_ndqf095.pickle', 'rb') as openfile:
    res_mat, em_iters = pickle.load(openfile)

plt.figure()
plt.imshow(res_mat, vmin=0, vmax=0.2, interpolation='none')
plt.xlabel('# Source Vertex', fontsize=16)
plt.ylabel('# Source Vertex', fontsize=16)
plt.title('SOMATA Resolution Matrix', fontsize=20)
plt.colorbar()
plt.savefig('10Hz_noiseless_ndqf095_resolution_matrix.svg')
plt.close()

# Compare with MNE resolution matrix
MNE_R = 1 * np.eye(src1.nchannel, dtype=np.float64)
MNE_G = src1.lfm
SNR = 3
MNE_Q = np.trace(MNE_R) / np.trace(MNE_G @ MNE_G.T) * SNR * np.eye(src1.nsource, dtype=np.float64)
M = MNE_Q @ MNE_G.T @ np.linalg.inv(MNE_G @ MNE_Q @ MNE_G.T + MNE_R)
MNE_res_mat = np.abs(M @ G)

plt.figure()
plt.imshow(MNE_res_mat, vmin=0, vmax=0.08, interpolation='none')
plt.xlabel('# Source Vertex', fontsize=16)
plt.ylabel('# Source Vertex', fontsize=16)
plt.title('MNE Resolution Matrix', fontsize=20)
plt.colorbar()
