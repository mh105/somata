from somata.oscillator_search import IterativeOscillatorModel as IterOsc
from somata.oscillator_search.helper_functions import (
    random, simulate_matsuda, sim_to_osc_object, innovations_wrapper)

random.seed(1)  # to ensure reproducible results
fs = 100  # sampling frequency (Hz)
osc1 = {'a': 0.996, 'q': 0.4, 'f': 0.1}  # set simulating parameters for slow oscillation
osc2 = {'a': 0.95, 'q': 0.2, 'f': 10}  # set simulating parameters for alpha oscillation
y, param_list, ob_noise = simulate_matsuda([osc1, osc2], R=1.2, Fs=fs, T=10)
sim_osc0, sim_x0 = sim_to_osc_object(y, param_list)  # save simulations as Osc object to pass into plotting functions

# Initialize Iterative Oscillator object
io1 = IterOsc(y, fs, noise_start=None, osc_range=7)
# noise_start determines the frequency above which is used to estimate the observation noise; default: Nyquist - 20 Hz
# osc_range is maximum number of total oscillators, set to default

# Run through iterations to fit model
io1.iterate(freq_res=1, plot_fit=True, verbose=True)
# freq_res is the minimal resolution in Hz from existing frequencies when adding a new oscillator
# plot_fit=True plots innovation spectrum and AR fitting during iterations
# verbose=True prints parameters throughout the method

# Plot frequency domain (from parameters and from estimated x_t_n)
for version in ['theoretical', 'actual']:
    _ = io1.get_knee_osc().visualize_freq(version, y=y, sim_osc=sim_osc0, sim_x=sim_x0)

# Plot time domain estimated x_t
_ = io1.get_knee_osc().visualize_time(y=y, sim_x=sim_x0)

# Plot likelihood and selected model (may not be the highest likelihood)
_ = io1.plot_log_likelihoods()

# Plot fitting at a specific iteration and the innovation spectrum
_ = innovations_wrapper(io1, 0)  # this should look like the first plot produced by io1.iterate()

# Other helpful plotting methods
_ = io1.plot_mtm()  # plot multitaper spectrogram and mean spectrum
_ = io1.plot_trace()  # plot raw time trace
_ = io1.plot_fit_spectra()  # plot fitted spectra (equivalent to calling .visualize_freq(version) manually)
