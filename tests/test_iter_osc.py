"""
Author: Mingjian He <mh1@stanford.edu>

Testing functions for the iOsc algorithm in somata/oscillator_search/iter_osc.py
"""

from somata.oscillator_search import IterativeOscillatorModel as IterOsc
from somata.oscillator_search.helper_functions import (
    random, np, simulate_matsuda, sim_to_osc_object, innovations_wrapper)
from test_load_data import _load_data  # type: ignore
import pickle


def test_iterative_osc(plot_on=False):
    random.seed(1)
    fs = 100
    osc1 = {'a': 0.996, 'q': 0.4, 'f': 0.1}
    osc2 = {'a': 0.95, 'q': 0.2, 'f': 10}
    y, param_list, ob_noise = simulate_matsuda([osc1, osc2], R=1.2, Fs=fs, T=10)
    sim_osc0, sim_x0 = sim_to_osc_object(y, param_list)

    io_test = IterOsc(y, fs)
    io_test.iterate(plot_fit=plot_on)

    # Verify diagnostic statistical tests
    acf = io_test.diagnose_residual_acf()
    assert 1.9 < acf['Durbin-Watson d'].item() < 2.1, "Failed Durbin-Watson test."
    assert acf['Ljung-Box'].item() > 0.05, "Failed Ljung-Box test."
    assert acf['Lagrange Multiplier'].item() > 0.05, "Failed Lagrange Multiplier test."
    assert acf['Brock-Dechert-Scheinkman'].item() > 0.05, "Failed Brock-Dechert-Scheinkman test."

    norm = io_test.diagnose_residual_norm()
    assert norm['Shapiro-Wilk'].item() > 0.05, "Failed Shapiro-Wilk test."
    assert norm['D\'Agostino and Pearson'].item() > 0.05, "Failed D'Agostino and Pearson test."
    assert norm['Anderson-Darling H'].item() is False, "Failed Anderson-Darling test."
    assert norm['Cramer-von Mises'].item() > 0.05, "Failed Cramer-von Mises test."

    if plot_on:
        # Plot frequency domain (from parameters and from estimated x_t_n)
        for version in ['theoretical', 'actual']:
            _ = io_test.get_knee_osc().visualize_freq(version, y=y, sim_osc=sim_osc0, sim_x=sim_x0)

        # Plot time domain estimated x_t
        _ = io_test.get_knee_osc().visualize_time(y=y, sim_x=sim_x0)

        # Plot likelihood and selected model (may not be the highest likelihood)
        _ = io_test.plot_log_likelihoods()

        # Plot fitting at a specific iteration and the innovation spectrum
        _ = innovations_wrapper(io_test, 0)

        # Plot multitaper spectrogram and mean spectrum
        _ = io_test.plot_mtm()

        # Plot raw time trace
        _ = io_test.plot_trace()

        # Plot fitted spectra (equivalent to calling .visualize_freq(version) manually)
        _ = io_test.plot_fit_spectra()

        # Plot the residual spectrum shifted by white noise processes
        _ = io_test.plot_residual()

        # Plot residual linear line fit
        _ = io_test.plot_residual_fit()

        # Plot autocorrelation function of the residuals
        _ = io_test.plot_acf()

        # Plot partial autocorrelation function of the residuals
        _ = io_test.plot_pacf()

    with open(_load_data('test_iter_osc_obj.pkl', return_path=True), 'rb') as inp:
        io_true = pickle.load(inp)

    assert io_true.knee_index == io_test.knee_index, "Does not choose the same oscillator"
    assert len(io_true.fitted_osc) == len(io_test.fitted_osc), "Does not have the same number of maximum oscillators"
    assert np.allclose(io_true.ll, io_test.ll), 'Log-likelihoods are not the same.'

    osc_true = io_true.get_knee_osc()
    osc_orig = io_test.get_knee_osc()

    assert np.allclose(osc_true.freq, osc_orig.freq), "Frequencies in IterOsc are not the same."
    assert np.allclose(osc_true.a, osc_orig.a), "Radii in IterOsc are not the same."
    assert np.allclose(osc_true.sigma2, osc_orig.sigma2), "sigma^2 in IterOsc are not the same."
    assert np.allclose(osc_true.R, osc_orig.R), "R (obs noise) in IterOsc are not the same."


if __name__ == "__main__":
    test_iterative_osc()
    print('Iterative oscillator test finished without exception.')
