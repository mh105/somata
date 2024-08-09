# Author: Mingjian He <mh1@stanford.edu>
from somata.oscillator_search import IterativeOscillatorModel as IterOsc
from somata.oscillator_search.helper_functions import innovations_wrapper
from somata import OscillatorModel as Osc
import numpy as np

# Simulate 10-s data with two oscillators, one slow and one alpha frequency
np.random.seed(1)  # to ensure reproducible results
o1 = Osc(a=[0.996, 0.95], freq=[0.1, 10], sigma2=[0.4, 0.2], R=1.2, Fs=100)
x, y = o1.simulate(duration=10)

# Initialize an IterativeOscillatorModel object
io1 = IterOsc(y, o1.Fs, noise_start=None, osc_range=7)
# noise_start determines the frequency above which is used to estimate the observation noise; default: (Nyquist - 20 Hz)
# osc_range is maximum number of total oscillators; default: 7

# Plot multitaper spectrogram, mean spectrum, and raw data time trace
_ = io1.plot_mtm()
_ = io1.plot_trace()

# Run through iterations to find the best model
io1.iterate(freq_res=1, plot_fit=True, verbose=True)  # this is the iOsc+ algorithm
# freq_res is the minimal resolution in Hz from existing frequencies when adding a new oscillator
# plot_fit=True plots innovation spectrum and AR fitting during each iteration
# verbose=True prints parameters throughout the method

# Inspect the selected model
print(io1.get_knee_osc())

# Plot log-likelihood and the selected model (may not be the highest)
_ = io1.plot_log_likelihoods()

# Plot fitting at a specific iteration and the innovation spectrum
_ = innovations_wrapper(io1, 0)  # the same as the first plot produced by io1.iterate()

# Plot estimated hidden states x_t in the frequency domain
_ = io1.plot_fit_spectra(sim_osc=o1, sim_x=x[:, 1:])
# sim_osc is the true OscillatorModel used for the data generation (optional)
# sim_x is the true hidden states x_t underlying the data generation (optional)

# Plot estimated hidden states x_t in the time domain
_ = io1.plot_fit_traces(sim_x=x[:, 1:])
# sim_x is the true hidden states x_t underlying the data generation (optional)

# Plot diagnostics of residuals and run statistical tests on autocorrelations and normality
io1.diagnose_residual()
