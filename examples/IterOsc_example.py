from somata.iterative_oscillator import IterativeOscillatorModel as IterOsc
from somata.iterative_oscillator.helper_functions import *

random.seed(1)  # to ensure reproducible results
fs = 100  # sampling frequency (Hz)
noise_start = 30
osc1 = {'a': 0.996, 'q': 0.4, 'f': 0.1}  # set simulating parameters for slow oscillation
osc2 = {'a': 0.95, 'q': 0.2, 'f': 10}  # set simulating parameters for alpha oscillation
y, param_list, ob_noise = simulate_matsuda([osc1, osc2], R=1.2, Fs=fs, T=10)
sim_osc0, sim_x0 = sim_to_osc_object(y, param_list)  # save simulations as Osc object to pass into plotting functions

# Initialize Iterative Oscillator object
io_orig = IterOsc(y, fs, noise_start, ar_order=13, burg_flag=False, verbose=True)  # ar_order and burg_flag are set as
# defaults. ar_order is the order of the AR model fit to the innovations. Set burg_flag to True to use the Burg
# algorithm to fit the AR model to innovations instead of the default Yule-Walker algorithm.

# Run through iterations to fit model
io_orig.iterate(osc_range=7, plot_innov=True, verbose=True)
# osc_range is maximum number of total oscillators, set to default.
# plot_innov=True plots innovation spectrum and AR fitting during iterations
# verbose=True prints parameters throughout the method

# Plot frequency domain (from parameters and from estimated x_t)
for version in ['theoretical', 'actual']:
    _ = io_orig.scaled_osc[io_orig.knee_index].visualize_freq(version, y=io_orig.y_original, sim_osc=sim_osc0,
                                                              sim_x=sim_x0)

# Plot time domain estimated x_t
_ = io_orig.scaled_osc[io_orig.knee_index].visualize_time(y=io_orig.y_original, sim_x=sim_x0)

# Plot likelihood and selected model (may not be the highest likelihood)
plt.figure()
plt.plot(range(1, len(io_orig.scaled_osc) + 1), io_orig.ll, '-*', label='__nolegend__')
plt.scatter(io_orig.knee_index + 1, io_orig.ll[io_orig.knee_index], color='red', zorder=2,
            label='model selected with knee')
plt.xlabel('Index of Model (# of Oscillations)')
plt.grid('on')
plt.title('Log Likelihood')

# this should look like the first plot produced by io_orig.iterate() but with additional poles marked in orange
_ = innovations_wrapper(io_orig, 0, plot_all_poles=True)
