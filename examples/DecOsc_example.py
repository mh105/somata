from somata.oscillator_search import DecomposedOscillatorModel as DecOsc
from somata.oscillator_search.helper_functions import random, simulate_matsuda, sim_to_osc_object

random.seed(1)  # to ensure reproducible results
fs = 100  # sampling frequency (Hz)
osc1 = {'a': 0.996, 'q': 0.4, 'f': 0.1}  # set simulating parameters for slow oscillation
osc2 = {'a': 0.95, 'q': 0.2, 'f': 10}  # set simulating parameters for alpha oscillation
y, param_list, ob_noise = simulate_matsuda([osc1, osc2], R=1.2, Fs=fs, T=10)
sim_osc0, sim_x0 = sim_to_osc_object(y, param_list)  # save simulations as Osc object to pass into plotting functions

# Initialize Decomposed Oscillator object
do1 = DecOsc(y, fs, noise_start=None, osc_range=7)
# noise_start determines the frequency above which is used to estimate the observation noise; default: Nyquist - 20 Hz
# osc_range is maximum number of total oscillators, set to default

# Run through decomposed oscillators to fit model
do1.iterate(plot_fit=True)
# plot_fit=True plots fitted spectrum at each decomposition of oscillators

# Plot frequency domain (from parameters and from estimated x_t_n)
for version in ['theoretical', 'actual']:
    _ = do1.get_knee_osc().visualize_freq(version, y=y, sim_osc=sim_osc0, sim_x=sim_x0)

# Plot time domain estimated x_t
_ = do1.get_knee_osc().visualize_time(y=y, sim_x=sim_x0)

# Plot likelihood and selected model (may not be the highest likelihood)
_ = do1.plot_log_likelihoods()

# Other helpful plotting methods
_ = do1.plot_mtm()  # plot multitaper spectrogram and mean spectrum
_ = do1.plot_trace()  # plot raw time trace
_ = do1.plot_fit_spectra()  # plot fitted spectra (equivalent to calling .visualize_freq(version) manually)
