# Author: Mingjian He <mh1@stanford.edu>
from somata.oscillator_search import DecomposedOscillatorModel as DecOsc
from somata import OscillatorModel as Osc
import numpy as np

# Simulate 10-s data with two oscillators, one slow and one alpha frequency
np.random.seed(1)  # to ensure reproducible results
o1 = Osc(a=[0.996, 0.95], freq=[0.1, 10], sigma2=[0.4, 0.2], R=1.2, Fs=100)
x, y = o1.simulate(duration=10)

# Initialize a DecomposedOscillatorModel object
do1 = DecOsc(y, o1.Fs, noise_start=None, osc_range=7)
# noise_start determines the frequency above which is used to estimate the observation noise; default: (Nyquist - 20 Hz)
# osc_range is maximum number of total oscillators; default: 7

# Plot multitaper spectrogram, mean spectrum, and raw data time trace
_ = do1.plot_mtm()
_ = do1.plot_trace()

# Run through the set of decomposed oscillators to find the best model
do1.iterate(plot_fit=True)  # this is the dOsc algorithm
# plot_fit=True plots fitted theoretical spectra during each iteration

# Inspect the selected model
print(do1.get_knee_osc())

# Plot log-likelihood and the selected model (may not be the highest)
_ = do1.plot_log_likelihoods()

# Plot estimated hidden states x_t in the frequency domain
_ = do1.plot_fit_spectra(sim_osc=o1, sim_x=x[:, 1:])
# sim_osc is the true OscillatorModel used for the data generation (optional)
# sim_x is the true hidden states x_t underlying the data generation (optional)

# Plot estimated hidden states x_t in the time domain
_ = do1.plot_fit_traces(sim_x=x[:, 1:])
# sim_x is the true hidden states x_t underlying the data generation (optional)

# Plot diagnostics of residuals and run statistical tests on autocorrelations and normality
do1.diagnose_residual()
