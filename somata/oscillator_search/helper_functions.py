"""
Author: Amanda Beck <ambeck@mit.edu>
        Mingjian He <mh1@stanford.edu>
"""

import numpy as np
from numpy import random
import math
from scipy.stats import chi2
from spectrum import aryule, arma2psd, arburg
import matplotlib.pyplot as plt
import warnings
from kneed import KneeLocator
from ..multitaper import fast_psd_multitaper
from ..multitaper.multitaper_spectrogram_python import multitaper_spectrogram as mtm
from ..basic_models import OscillatorModel as Osc
from ..basic_models import AutoRegModel as Arn


def innovations_wrapper(iter_osc, iter_num, plot_all_poles=True, ax_limit=None, figsize=(8, 6), horizontal=False):
    """ Plot the innovation spectrum of the fitted model at a specific iteration """
    y = iter_osc.added_osc[0].y
    osc = iter_osc.fitted_osc[iter_num]
    innovations, _ = get_innovations(osc, y=y)

    add_freq = iter_osc.added_osc[iter_num + 1].freq[0] if iter_num + 1 < len(iter_osc.added_osc) else None
    add_radius = iter_osc.added_osc[iter_num + 1].a[0] if iter_num + 1 < len(iter_osc.added_osc) else None

    a, p = fit_ar(np.squeeze(innovations), ar_order=iter_osc.osc_range * 2 - 1)
    fig, axlim = plot_innovations(osc, y, innovations, a, p, add_freq, add_radius, plot_all_poles=plot_all_poles,
                                  ax_limit=ax_limit, figsize=figsize, horizontal=horizontal, counter=iter_num)

    return fig, axlim


def get_innovations(osc, y=None):
    """ Calculate innovations / one-step prediction error """
    y = osc.y if y is None else y
    kalman_out = osc.dejong_filt_smooth(y=y, return_dict=True)
    x_pred = kalman_out['x_t_tmin1'][:, 1:]
    y_pred = osc.G @ x_pred
    return (y - y_pred).squeeze(), kalman_out['logL'].sum()


def fit_ar(y, ar_order, burg_flag=False):
    """ Fit AR model to a time series """
    if burg_flag:  # use the Burg method
        a, p, _ = arburg(y, ar_order)
    else:  # default is Yule Walker algorithm
        a, p, _ = aryule(y, ar_order)
    return a, p  # first output is the negative of AR coefficients in y_t = \sum a*y_{t-1} + e_t


def get_ar_psd(fs, a, p, ar_hz=None, df=0.01):
    """ Get the PSD spectrum of a fitted AR model """
    if ar_hz is None:
        nfft = int(2 ** np.ceil(np.log2(fs / df)))  # nfft to achieve a [< df] sampling resolution of frequencies
        ar_hz = np.arange(0, fs / 2, fs / nfft)  # one-sided frequency 0 to Nyquist
        assert ar_hz.size == nfft // 2, 'Incorrect nfft in the AR theoretical spectrum.'
    ar_psd = arma2psd(A=a, rho=p, NFFT=2 * ar_hz.size, sides='centerdc')[ar_hz.size:] * 2  # make spectrum one-sided
    return ar_psd, ar_hz


def initialize_allosc(fs, y, R, osc_range=7, drop_y=False, fill_up=True):
    """ Initialize all new oscillators using spectral decomposition """
    a, p = fit_ar(np.squeeze(y), ar_order=osc_range * 2 - 1)
    r = np.roots(np.concatenate([[1], a]))

    # AR estimated root frequencies and radii
    root_freqs = np.abs(np.arctan2(np.imag(r), np.real(r)) * fs / 2 / np.pi)  # force to positive frequencies
    root_freqs = np.mod(root_freqs, fs / 2)  # map Nyquist to dc frequency
    root_radii = np.abs(r)  # compute the magnitude of complex roots

    # spectral decomposition to estimate root specific variances
    eig, eigvec = np.linalg.eig(Arn(coeff=-a, sigma2=p).F)
    assert np.all(r == eig), 'eigenvalues do not equal to complex roots.'
    c = eigvec[0, :] * np.linalg.inv(eigvec)[:, 0]  # sum of c is 1
    root_sigma2 = np.real(p * c * c.conj())  # the variance in each mode when observed alone

    # sort by descending root variance
    unique_freqs = np.unique(root_freqs)
    unique_sigma2 = np.array([np.sum(root_sigma2[root_freqs == x]) for x in unique_freqs])
    add_indices = np.argsort(unique_sigma2)[::-1]

    # lower bound sigma2 to be above 0 after sorting
    unique_sigma2[unique_sigma2 <= 0] = np.finfo(float).eps

    # create a list of oscillators to be added sequentially
    added_osc = []
    for ii in range(len(add_indices)):
        add_freq = unique_freqs[add_indices[ii]]
        add_radius = max(root_radii[root_freqs == add_freq])
        add_sigma2 = unique_sigma2[add_indices[ii]]
        if ii == 0 and not drop_y:
            added_osc.append(Osc(y=y, a=add_radius, freq=max([add_freq, 0.1]), sigma2=add_sigma2, Fs=fs, R=R))
        else:
            added_osc.append(Osc(a=add_radius, freq=max([add_freq, 0.1]), sigma2=add_sigma2, Fs=fs))

    # fill up to osc_range oscillators to be added
    if fill_up:
        while len(added_osc) < osc_range:
            o1 = added_osc[0].copy()
            _ = [o1.append(x) for x in added_osc[1:]]
            residual = o1.y - o1.G @ o1.dejong_filt_smooth(return_dict=True)['x_t_tmin1'][:, 1:]
            more_osc = initialize_allosc(fs, residual, R, osc_range=osc_range, drop_y=True, fill_up=False)
            for ii in range(min([osc_range - len(added_osc), len(more_osc)])):
                added_osc.append(more_osc[ii])
        assert len(added_osc) == osc_range, 'Incorrect number of oscillators during decomposition.'

    return added_osc


def initialize_newosc(fs, innovations, existing_freqs=None, freq_res=1, ar_order=13, burg_flag=False, sigma2='eig'):
    """ Initialize the new oscillator to be added """
    a, p = fit_ar(np.squeeze(innovations), ar_order=ar_order, burg_flag=burg_flag)
    r = np.roots(np.concatenate([[1], a]))
    root_freqs = np.abs(np.arctan2(np.imag(r), np.real(r)) * fs / 2 / np.pi)  # force to positive frequencies
    root_radii = np.abs(r)  # compute the magnitude of complex roots

    # within [freq_res] Hz of existing frequencies is considered as duplicated
    root_freqs_sel = [True if existing_freqs is None else min(abs(x - existing_freqs)) >= freq_res for x in root_freqs]
    root_freqs_sel = np.invert(root_freqs_sel) if not any(root_freqs_sel) else root_freqs_sel  # all root_freqs overlap

    # look for the root with the largest AR theoretical PSD scaled by radius
    ar_psd, ar_hz = get_ar_psd(fs, a, p)
    root_psd = np.array([ar_psd[np.argmin(abs(freq - ar_hz))] * radius for freq, radius in zip(root_freqs, root_radii)])
    sel_idx = np.argmax(root_psd[root_freqs_sel])
    add_freq = max([root_freqs[root_freqs_sel][sel_idx], 0.1])  # lower bound added frequency to 0.1 Hz
    add_radius = root_radii[root_freqs_sel][sel_idx]

    # compute the corresponding sigma2 for the new oscillator
    # via eigendecomposition and transform of the AR process white noise variance
    eig, eigvec = np.linalg.eig(Arn(coeff=-a, sigma2=p).F)  # eig is equivalent to complex roots in r
    assert np.all(r == eig), 'eigenvalues do not equal to complex roots.'
    c = eigvec[0, :] * np.linalg.inv(eigvec)[:, 0]  # sum of c is 1
    root_sigma2 = np.real(p * c * c.conj())  # the variance in each mode when observed alone
    add_sigma2_eig = np.sum(root_sigma2[root_freqs == root_freqs[root_freqs_sel][sel_idx]])  # sum across pairs of roots

    # via empirical matching to the fitted AR PSD
    add_psd = root_psd[root_freqs_sel][sel_idx]  # this psd is already scaled by radius
    w = Osc.hz_to_rad(add_freq, fs)
    z = np.exp(1j * Osc.hz_to_rad(root_freqs[root_freqs_sel][sel_idx], fs))
    A = (1 - 2 * add_radius ** 2 * np.cos(w) ** 2 + add_radius ** 4 * np.cos(2 * w)) / (
            add_radius * (add_radius ** 2 - 1) * np.cos(w))
    b = 0.5 * (A - 2 * add_radius * np.cos(w) + np.sqrt((A - 2 * add_radius * np.cos(w)) ** 2 - 4))
    V = 0.5 * add_psd * np.abs(1 - 2 * add_radius * np.cos(w) * z + add_radius ** 2 * z ** 2
                               ) ** 2 / np.abs(1 + b * z) ** 2
    add_sigma2_psd = -V * b / (add_radius * np.cos(w))

    if sigma2 == 'avg':
        add_sigma2 = np.mean([add_sigma2_eig, add_sigma2_psd])
    elif sigma2 == 'eig':
        add_sigma2 = add_sigma2_eig
    else:
        add_sigma2 = add_sigma2_psd

    return add_freq, add_radius, add_sigma2, root_freqs, root_radii, a, p


def aic_calc(osc, ll):
    """ Calculate AIC """
    if osc.dc_idx is None:
        k = osc.ncomp * 3 + 1  # each oscillator has 3 parameters + obs noise variance
    else:
        k = osc.ncomp * 3  # each osc has 3 parameters, each dc has 2, + obs noise variance
    return 2 * k - 2 * ll


def aic_weights(aic_list):
    """
    Calculate AIC weights.
    Weights represent each model's likelihood relative to the set of models.
    They can be used as a measure of model probability
    """
    deltas = (aic_list - np.min(aic_list)) / 2
    return np.exp(-deltas) / np.sum(np.exp(-deltas))


def ebic_calc(osc, ll, osc_range, gamma=0.5):
    """
    Calculate extended Bayesian Information Criterion (EBIC).
    When gamma=0, EBIC is equivalent to BIC

    Reference:
        Chen, J., & Chen, Z. (2008). Extended Bayesian information criteria
        for model selection with large model spaces. Biometrika, 95(3), 759-771.
    """
    if osc.dc_idx is None:
        k = osc.ncomp * 3 + 1  # each oscillator has 3 parameters + obs noise variance
    else:
        k = osc.ncomp * 3  # each osc has 3 parameters, each dc has 2, + obs noise variance

    return k * np.log(osc.ntime) - 2 * ll + 2 * gamma * np.log(math.comb(osc_range, osc.ncomp))


def get_knee(ll_vec):
    """ Use 'kneed' python package to find a knee on the log likelihoods """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='No knee/elbow found', category=UserWarning)
        kneedle = KneeLocator(range(len(ll_vec)), ll_vec, S=1.0,
                              curve="concave", direction="increasing", online=False)  # stop at the first knee
    knee_index = kneedle.knee if kneedle.knee is not None else len(ll_vec) - 1  # if no knee found, output last index
    return knee_index  # Returns int, index of model at knee


def get_last_increase(sequence):
    """ Find the index before the first decrease in a sequence """
    sequence_diff = np.diff(sequence)
    for idx in range(len(sequence_diff)):
        if sequence_diff[idx] <= 0:
            return idx
    return len(sequence) - 1  # return index of the last value in sequence


def simulate_matsuda(param_list, R=1, Fs=100, T=10):
    """
    Simulate time series from matsuda oscillators (Matsuda and Komaki 2017)
        param_list = array of dictionaries, one for each oscillation with the following entries:
            a = scalar damping factor
            q = scalar state noise covariance
            f = scalar frequency
        R = scalar observation noise
        Fs = scalar sampling frequency
        T = scalar time length in seconds
    """
    # total number of samples in the time series
    N = T * Fs

    # verify all required keys are available
    for ii in range(len(param_list)):
        for key, name in zip(['a', 'q', 'f'], ['Damping factor', 'State noise cov', 'Frequency']):
            assert key in param_list[ii], '%s is missing from oscillation %d' % (name, ii)

    # generate the time series with an extra t=0 point that will get removed during return
    y = np.zeros((1, N + 1))
    for params in param_list:
        w = params['f'] * 2 * np.pi / Fs
        noise = random.normal(0, params['q'] ** (1 / 2), size=(2, N))
        F = Osc.get_rot_mat(params['a'], w)
        params['x_osc'] = np.zeros((2, N + 1))
        for ii in range(N):
            params['x_osc'][:, ii + 1] = noise[:, ii] + F @ params['x_osc'][:, ii]
        y += params['x_osc'][0, :]
        params['R'] = R
        params['Fs'] = Fs

    # add observation noise
    noise_meas = random.normal(0, R ** (1 / 2), size=(1, N + 1))
    y += noise_meas

    return y.squeeze()[1:], param_list, noise_meas


def sim_to_osc_object(y, param_list):
    """ Reorganize a list of simulated oscillators into an OscillatorModel object """
    assert np.all([param_list[0]['R'] == ii['R'] for ii in param_list]), \
        'Not all observation noises are the same. Check R'
    assert np.all([param_list[0]['Fs'] == ii['Fs'] for ii in param_list]), \
        'Not all sampling frequencies are the same. Check Fs'

    osc_init = {'y': y, 'a': [ii['a'] for ii in param_list], 'freq': [ii['f'] for ii in param_list],
                'sigma2': [ii['q'] for ii in param_list],
                'Fs': param_list[0]['Fs'], 'R': param_list[0]['R'], 'add_dc': False}
    osc_out = Osc(**osc_init)
    x_osc = np.vstack([param_list[ii]['x_osc'] for ii in range(len(param_list))])
    return osc_out, x_osc[:, 1:]


def plot_param(iterosc, title, sim_osc=None):
    """ Plotting parameters: AIC, R, frequency """
    osc_num = len(iterosc.AIC)
    osc_choice = np.argmin(iterosc.AIC)

    f, ax = plt.subplots(5, 1, sharex='all', figsize=(4, 14))
    ax[0].set_title(title)
    _ = [ii.grid(axis='y', which='both') for ii in ax]
    _ = [ii.axvspan(osc_choice - 0.25, osc_choice + 0.25, alpha=0.3, color='yellow') for ii in ax]

    for (ii, label) in zip(range(5), ['AIC weights', 'Obs noise (R)',
                                      'Frequency', 'Damping Factor', 'State noise (Q)']):
        ax[ii].set_ylabel(label)

    AIC_weights = aic_weights(iterosc.AIC)
    ax[0].scatter(range(osc_num), AIC_weights)
    ax[1].scatter(range(osc_num), [ii.R[0] for ii in iterosc.fitted_osc])
    for ii in range(osc_num):
        value_on(ax[0], ii, AIC_weights[ii])
        value_on(ax[1], ii, iterosc.fitted_osc[ii].R[0])

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for ii in range(len(iterosc.fitted_osc)):
        osc = iterosc.fitted_osc[ii]
        for jj in range(ii + 1):
            ax[2].scatter(ii, np.abs(osc.w[jj]) * osc.Fs / (2 * np.pi), color=colors[jj])
            value_on(ax[2], ii, np.abs(osc.w[jj]) * osc.Fs / (2 * np.pi))
            ax[3].scatter(ii, osc.a[jj], color=colors[jj])
            value_on(ax[3], ii, osc.a[jj])
            ax[4].scatter(ii, osc.sigma2[jj], color=colors[jj])
            value_on(ax[4], ii, osc.sigma2[jj])

    ax[-1].set_xticks(np.arange(osc_num), [str(ii) for ii in np.arange(osc_num) + 1])
    ax[-1].set_xlabel('Number of Oscillations')
    ax[-1].margins(x=0.25)

    if sim_osc is not None:
        ax[1].axhline(sim_osc.R[0], linestyle=':')
        for ii in range(sim_osc.ncomp):
            ax[2].axhline(sim_osc.w[ii] * sim_osc.Fs / (2 * np.pi), linestyle=':', color=colors[ii])
            ax[3].axhline(sim_osc.a[ii], linestyle=':', color=colors[ii])
            ax[4].axhline(sim_osc.sigma2[ii], linestyle=':', color=colors[ii])

    for ii in ax:
        ii.margins(y=0.25)
        ii.autoscale_view(tight=False)


def value_on(ax, current_osc, value):
    """ Add text of value at (x,y)=(current_osc,value) """
    ax.text(current_osc + 0.1, value, '%0.2f' % value)


def visualize_em(em_params, y, bw=None):
    """
    Plot tracked parameters from EM iterations.

    iterosc.all_params is a list of lists. The inner lists are one list per oscillator,
    with the Osc objects over the EM iterations. The inner list is em_params
    """
    fig, ax = plt.subplots(1, 1)
    Fs = em_params[0].Fs
    psd_mt, f_hz = fast_psd_multitaper(y, Fs, 0, Fs / 2, bw)
    ax.plot(f_hz, 10 * np.log10(psd_mt), color='black', label='observed data')
    cm = plt.get_cmap('coolwarm')
    for ii in range(len(em_params)):
        # noinspection PyProtectedMember
        h_i, f_th = em_params[ii]._theoretical_spectrum()
        for jj in range(len(h_i)):
            cl = int(ii / len(em_params) * 256)  # y further 0~Convert to 255
            ax.plot(f_th, 10 * np.log10(h_i[jj]), c=cm(cl))
    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB)')
    return fig


def plot_mtm(osc, frequency_range=None, time_bandwidth=2, window_params=(1, 0.05), df=0.1,
             detrend_opt='constant', ax=(None, None)):
    """ Plot multitaper spectrogram and mean spectrum """
    min_nfft = int(2 ** np.ceil(np.log2(osc.added_osc[0].Fs / df)))
    if ax[0] is not False:
        mt_spectrogram, stimes, sfreqs, (fig_spectrogram, ax0) = mtm(osc.added_osc[0].y, osc.added_osc[0].Fs,
                                                                     frequency_range=frequency_range,
                                                                     time_bandwidth=time_bandwidth,
                                                                     window_params=list(window_params),
                                                                     min_nfft=min_nfft, detrend_opt=detrend_opt,
                                                                     return_fig=True, ax=ax[0])
        ax0.set_xlabel('Time (Second)')
        ax0.set_title('Multitaper Spectrogram')
    else:
        mt_spectrogram, stimes, sfreqs = mtm(osc.added_osc[0].y, osc.added_osc[0].Fs,
                                             frequency_range=frequency_range,
                                             time_bandwidth=time_bandwidth,
                                             window_params=list(window_params),
                                             min_nfft=min_nfft, detrend_opt=detrend_opt,
                                             return_fig=False)
        fig_spectrogram = None

    if ax[1] is not False:
        if ax[1] is None:
            fig_spectrum, ax1 = plt.subplots(1, 1)
        else:
            ax1 = ax[1]
            fig_spectrum = ax1.get_figure()
        ax1.plot(sfreqs, 10 * np.log10(np.mean(mt_spectrogram, axis=1)))
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('PSD (dB)')
        ax1.set_title('Mean Spectrum')
    else:
        fig_spectrum = None

    return fig_spectrogram, fig_spectrum


def plot_innovations(osc, y, innovations, a, p, add_freq=None, add_radius=None, reiterate=False, ax=None,
                     plot_all_poles=True, ax_limit=None, figsize=(8, 6), horizontal=False, bw=1, counter=0):
    """ Plot fitted OscillatorModel and the innovations / one-step prediction error for that model """
    psd_mt, fy_hz = fast_psd_multitaper(innovations, osc.Fs, 0, osc.Fs / 2, bw)
    psd_ar, _ = get_ar_psd(osc.Fs, a, p, ar_hz=fy_hz)
    psd_y, _ = fast_psd_multitaper(y.squeeze(), osc.Fs, 0, osc.Fs / 2, bw)

    if ax is None:
        if horizontal:
            fig, [ax0, ax1] = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, [ax0, ax1] = plt.subplots(2, 1, sharex='all', figsize=figsize)
    else:
        assert len(ax) == 2, 'Need two axes for the innovations plot.'
        fig = ax[0].get_figure()
        ax0, ax1 = ax

    ax0.plot(fy_hz, 10 * np.log10(psd_y), color='black', label='observed data')

    h_th, f_th = osc.oscillator_spectra('theoretical')
    for ii in range(len(h_th)):
        ax0.plot(f_th, 10 * np.log10(h_th[ii]), label='osc %d fit, %0.2f Hz' % (ii + 1, osc.freq[ii]))
    ax0.legend(loc='upper right')
    ax0.set_ylabel('PSD (dB)')
    if reiterate:
        ax0.set_title('Re-iteration %d' % counter)
    else:
        ax0.set_title('Iteration %d' % counter)

    ax1.plot(fy_hz, 10 * np.log10(psd_mt), label='OSPE', color='tab:blue')
    ax1.plot(fy_hz, 10 * np.log10(psd_ar), label='AR fit', linestyle='dashed', color='tab:blue')
    if ax_limit is not None:
        ax1.set_ylim(ax_limit)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (dB)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax1.set_title('One-Step Prediction Error (OSPE) - Iteration %d' % counter)
    if add_freq is not None:
        ax1.axvline(add_freq, color='black', linestyle='dashed',
                    label='frequency of new oscillator: %0.2f Hz' % add_freq)
    # ax1.axhline(20 * np.log10(b), color='black', label='observation noise estimate')

    ax2 = ax1.twinx()
    if plot_all_poles:
        r = np.roots(np.concatenate([[1], a]))
        root_freqs = np.abs(np.arctan2(np.imag(r), np.real(r)) * osc.Fs / 2 / np.pi)  # force to positive frequencies
        ax2.scatter(root_freqs, np.abs(r), color='tab:orange')
    if add_freq is not None and add_radius is not None:
        ax2.scatter(add_freq, add_radius, facecolor='tab:green', linewidth=2, label='pole for initialization')

    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Root (magnitude)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.spines['left'].set_color('tab:blue')
    ax2.spines['right'].set_color('tab:green')
    ax2.spines['left'].set(lw=3)
    ax2.spines['right'].set(lw=3)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower center')

    fig.set_tight_layout(True)  # otherwise, the right y-label is slightly clipped

    return fig, ax1.get_ylim()


def plot_fit_line(fig, slope, intercept, ax=None):
    """ Add a linear fit line to an existing innovation plot figure """
    ax = ax or fig.axes[0]
    x_limits = ax.get_xlim()
    l1 = ax.plot(x_limits, [x * slope + intercept for x in x_limits], label='linear fit', color='black')
    ax.set_xlim(x_limits)

    lines = [x for x in ax.get_legend().legend_handles] + l1
    labels = [x.get_label() for x in lines]
    ax.legend(lines, labels, loc='lower center')

    return fig


def plot_acf(residual, Fs, ax=None):
    """ Plot the autocorrelation function of the residuals """
    from statsmodels.graphics.tsaplots import plot_acf

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    else:
        assert len(ax) == 2, 'Need two axes for the acf plot.'
        fig = ax[0].get_figure()

    plot_acf(residual, lags=Fs, ax=ax[0])
    ax[0].set_xlabel('Lags (Samples)')
    ax[0].set_title(ax[0].get_title() + ' (Fs = {0:.1f} Hz)'.format(Fs))

    plot_acf(residual, lags=Fs, ax=ax[1])
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_title(ax[1].get_title() + ' (Fs = {0:.1f} Hz)'.format(Fs))
    freqs = np.array([Fs, Fs/2, 30, 20, 12, 10, 8, 4, 2, 1])
    ax[1].set_xticks(Fs/freqs, ['{0:.0f}'.format(x) for x in freqs])
    ax[1].set_xticks([], minor=True)

    return fig


def plot_pacf(residual, Fs, ax=None):
    """ Plot the partial autocorrelation function of the residuals """
    from statsmodels.graphics.tsaplots import plot_pacf

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    else:
        assert len(ax) == 2, 'Need two axes for the pacf plot.'
        fig = ax[0].get_figure()

    # if the residual is too short, the pacf will be unreliable
    lags = max(Fs/2, 10) if residual.size <= Fs * 2 else Fs  # at least 10 lags

    plot_pacf(residual, lags=lags, ax=ax[0])
    ax[0].set_xlabel('Lags (Samples)')
    ax[0].set_title(ax[0].get_title() + ' (Fs = {0:.1f} Hz)'.format(Fs))

    plot_pacf(residual, lags=lags, ax=ax[1])
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_title(ax[1].get_title() + ' (Fs = {0:.1f} Hz)'.format(Fs))
    freqs = np.array([Fs, Fs/2, 30, 20, 12, 10, 8, 4, 2, 1])
    ax[1].set_xticks(Fs/freqs, ['{0:.0f}'.format(x) for x in freqs])
    ax[1].set_xticks([], minor=True)

    return fig


def plot_residual(osc, y, residual, fit_version='predicted', ax=None, bw=1):
    """
    Plot the best fitted OscillatorModel and the residuals from that model

    This function is similar to plot_innovations(), but it shows empirical PSD
    of fitted oscilaltors, and it shows the confidence interval of the residuals
    """
    psd_y, _ = fast_psd_multitaper(y.squeeze(), osc.Fs, 0, osc.Fs / 2, bw)
    psd_mt, fy_hz = fast_psd_multitaper(residual, osc.Fs, 0, osc.Fs / 2, bw)

    # Recompute the number of tapers used to form Chi-sqaured confidence intervals
    # this calculation is taken directly from the fast_psd_multitaper() function
    window_length = min(residual.size / osc.Fs, 10)  # max to 10s windows
    time_bandwidth = np.max([1., bw / 2 * window_length])
    num_tapers = math.floor(2 * time_bandwidth) - 1
    alpha = 0.95
    lower_bound = 10 * np.log10(2 * num_tapers / chi2.ppf(0.5 + alpha / 2, 2 * num_tapers))
    upper_bound = 10 * np.log10(2 * num_tapers / chi2.ppf(0.5 - alpha / 2, 2 * num_tapers))

    if ax is None:
        fig, [ax0, ax1] = plt.subplots(2, 1, sharex='all')
    else:
        assert len(ax) == 2, 'Need two axes for the model/residual plot.'
        fig = ax[0].get_figure()
        ax0, ax1 = ax

    ax0.plot(fy_hz, 10 * np.log10(psd_y), color='black', label='observed data')

    if fit_version == 'theoretical':
        h_th, f_th = osc.oscillator_spectra('theoretical')
    else:
        kalman_out = osc.kalman_filt_smooth(y, return_dict=True)
        if fit_version == 'predicted':
            x = kalman_out['x_t_tmin1'][:, 1:]
        elif fit_version == 'filtered':
            x = kalman_out['x_t_t'][:, 1:]
        elif fit_version == 'smoothed':
            x = kalman_out['x_t_n'][:, 1:]
        else:
            raise ValueError('Invalid fit_version. Choose from "predicted", "filtered", or "smoothed"')
        h_th, f_th = osc.oscillator_spectra('actual', x, bw)

    for ii in range(len(h_th)):
        ax0.plot(f_th, 10 * np.log10(h_th[ii]), label='osc %d fit, %0.2f Hz' % (ii + 1, osc.freq[ii]))
    ax0.legend(loc='upper right')
    ax0.set_ylabel('PSD (dB)')
    ax0.set_title(f'Selected Model Fit ({fit_version})')

    noise_level = np.squeeze((np.sum(osc.sigma2) + osc.R)) * 2  # adjust for the one-sided spectrum
    ax1.plot(fy_hz, 10 * np.log10(psd_mt / noise_level), label='residual / noises', color='tab:blue')

    ax1.axhline(0, color='gray', linestyle='dashed')
    ax1.axhline(lower_bound, color='gray')
    ax1.axhline(upper_bound, color='gray')
    xlimits = ax1.get_xlim()
    ax1.fill_between([ax1.get_xlim()[0], ax1.get_xlim()[1]], lower_bound, upper_bound,
                     facecolor='gray', alpha=0.25, label='95% CI')
    ax1.set_xlim(xlimits)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (dB)')
    ax1.set_title('Residual over estimated white noises')
    ax1.legend(loc='lower center')

    return fig
