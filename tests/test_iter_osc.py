from somata.iterative_oscillator import IterativeOscillatorModel as IterOsc
from somata.iterative_oscillator.helper_functions import *
from test_load_data import _load_data
import pickle


def test_iterative_osc(plot_on=False):
    random.seed(1)
    fs = 100
    noise_start = 30
    osc1 = {'a': 0.996, 'q': 0.4, 'f': 0.1}
    osc2 = {'a': 0.95, 'q': 0.2, 'f': 10}
    y, param_list, ob_noise = simulate_matsuda([osc1, osc2], R=1.2, Fs=fs, T=10)
    sim_osc0, sim_x0 = sim_to_osc_object(y, param_list)

    io_test = IterOsc(y, fs, noise_start)
    io_test.iterate(plot_innov=plot_on)

    if plot_on:
        # plot frequency domain
        for version in ['theoretical', 'actual']:
            _ = io_test.fitted_osc[io_test.knee_index].visualize_freq(version, y=y, sim_osc=sim_osc0, sim_x=sim_x0)

        # plot time domain
        _ = io_test.fitted_osc[io_test.knee_index].visualize_time(y=y, sim_x=sim_x0)

        # plot likelihood
        plt.figure()
        plt.plot(range(len(io_test.fitted_osc)), io_test.ll, '-*', label='__nolegend__')
        plt.scatter(io_test.knee_index, io_test.ll[io_test.knee_index], color='red', zorder=2, label='knee')
        plt.xlabel('Index of Model (# of Oscillations)')
        plt.grid('on')
        plt.title('Log Likelihood')

        _ = innovations_wrapper(io_test, 0, plot_all_poles=True)

    with open(_load_data('iter_osc_obj.pkl', return_path=True), 'rb') as inp:
        io_true = pickle.load(inp)

    assert io_true.knee_index == io_test.knee_index, "Does not choose same oscillator"
    assert len(io_true.fitted_osc) == len(io_test.fitted_osc), "Does not have the same number of maximum oscillators"
    assert np.allclose(io_true.ll, io_test.ll), 'Log-likelihoods are not the same.'

    osc_true = io_true.fitted_osc[io_true.knee_index]
    osc_orig = io_test.fitted_osc[io_test.knee_index]

    assert np.allclose(osc_true.freq, osc_orig.freq), "Frequencies in IterOsc are not the same."
    assert np.allclose(osc_true.a, osc_orig.a), "Radii in IterOsc are not the same."
    assert np.allclose(osc_true.sigma2, osc_orig.sigma2), "sigma^2 in IterOsc are not the same."
    assert np.allclose(osc_true.R, osc_orig.R), "R (obs noise) in IterOsc are not the same."


if __name__ == "__main__":
    test_iterative_osc()
    print('Iterative oscillator test finished without exception.')
