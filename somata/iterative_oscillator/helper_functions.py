"""
Author: Amanda Beck <ambeck@mit.edu>
        Mingjian He <mh105@mit.edu>
"""

import numpy as np
from numpy import random
from spectrum import aryule, arma2psd, arburg
import matplotlib.pyplot as plt
from kneed import KneeLocator
from ..multitaper import fast_psd_multitaper
from ..basic_models import OscillatorModel as Osc


def innovations_wrapper(iter_osc, iter_num, plot_all_poles=True, ax_limit=None, fig_size=(8, 6), horizontal=False):
    """ Plots the innovations for fitted models, used for Beck et al. 2022 """
    y = iter_osc.added_osc[0].y
    osc = iter_osc.fitted_osc[iter_num]
    innovations, ll = find_innovations(osc, y=y)

    add_freq = iter_osc.added_osc[iter_num + 1].freq[0] if iter_num + 1 < len(iter_osc.added_osc) else None
    add_radius = iter_osc.added_osc[iter_num + 1].a[0] if iter_num + 1 < len(iter_osc.added_osc) else None

    a, b, _ = fit_ar(np.squeeze(innovations), iter_osc.osc_range * 2 - 1, iter_osc.burg_flag)
    axlim, fig = innovations_plot(osc, y, innovations, a, b, add_freq, add_radius, plot_all_poles=plot_all_poles,
                                  ax_limit=ax_limit, fig_size=fig_size, horizontal=horizontal, counter=iter_num)

    return axlim, fig


def find_innovations(osc, y=None):
    """ Calculate innovations / one step prediction error """
    y = osc.y if y is None else y
    kalman_out = osc.dejong_filt_smooth(y=y, return_dict=True)
    x_pred = kalman_out['x_t_tmin1'][:, 1:]
    y_pred = osc.G @ x_pred
    return (y - y_pred).squeeze(), kalman_out['logL'].sum()


def fit_ar(y, ar_order, burg_flag=False):
    """ Fit AR model to a time series """
    if burg_flag:  # use Burg algorithm
        a, p, k = arburg(y, ar_order)
    else:  # default is Yule Walker algorithm
        a, p, k = aryule(y, ar_order)
    return a, p, k


def get_ar_psd(fs, a, p, ar_hz=None, df=0.01):
    """ Get the PSD spectrum of a fitted AR model """
    if ar_hz is None:
        nfft = int(2 ** np.ceil(np.log2(fs / df)))  # nfft to achieve a [< df] sampling resolution of frequencies
        ar_hz = np.arange(0, fs / 2, fs / nfft)  # one-sided frequency 0 to Nyquist
        assert ar_hz.size == nfft // 2, 'Incorrect nfft in the AR theoretical spectrum.'
    ar_psd = arma2psd(A=a, rho=p, NFFT=2 * ar_hz.size, sides='centerdc')[ar_hz.size:] * 2  # make spectrum one-sided
    return ar_psd, ar_hz


def initialize_newosc(fs, innovations, existing_freqs=None, freq_res=1, ar_order=13, burg_flag=False):
    """ Initialize the new oscillator to be added """
    a, p, k = fit_ar(np.squeeze(innovations), ar_order, burg_flag)
    r = np.roots(np.concatenate(([1], a)))
    sampling_constant = fs / 2 / np.pi
    root_freqs = np.abs(np.arctan2(np.imag(r), np.real(r)) * sampling_constant)  # force to positive frequencies
    root_radii = np.abs(r)  # compute the magnitude of complex roots

    # within [freq_res] Hz of existing frequencies is considered as duplicated
    root_freqs_idx = [min(abs(x - existing_freqs)) >= freq_res
                      if existing_freqs is not None else True for x in root_freqs]

    # look for the root with the largest AR theoretical PSD scaled by radius
    ar_psd, ar_hz = get_ar_psd(fs, a, p)
    root_psd = np.array(
        [ar_psd[np.argmin(abs(freq - ar_hz))] * radius for freq, radius in zip(root_freqs, root_radii)])
    sel_idx = np.argmax(root_psd[root_freqs_idx])
    add_freq = max([root_freqs[root_freqs_idx][sel_idx], 0.1])  # lower bound added frequency to 0.1 Hz
    add_radius = root_radii[root_freqs_idx][sel_idx]

    return add_freq, add_radius, root_freqs, root_radii, a, p


def aic_calc(osc, ll):
    """ Calculate AIC """
    if osc.dc_idx is None:
        param_num = osc.ncomp * 3 + 1  # each oscillator has 3 parameters + obs noise variance
    else:
        param_num = osc.ncomp * 3  # each osc has 3 parameters, each dc has 2, + obs noise variance
    return 2 * param_num - 2 * ll


def aic_weights(aic_list):
    """
    Calculate AIC weights.
    Weights represent each model's likelihood relative to the set of models.
    They can be used as a measure of model probability
    """
    deltas = (aic_list - np.min(aic_list)) / 2
    return np.exp(-deltas) / np.sum(np.exp(-deltas))


def get_knee(ll_vec):
    """ Use 'kneed' python package to find knee/elbow """
    kneedle = KneeLocator(range(len(ll_vec)), ll_vec, S=1.0, curve="concave",
                          direction="increasing")
    return kneedle.knee  # Returns int, index of model at knee


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
    plt.figure()
    Fs = em_params[0].Fs
    psd_mt, f_hz = fast_psd_multitaper(y, Fs, 0, Fs / 2, bw)
    plt.plot(f_hz, 10 * np.log10(psd_mt), color='black', label='observed data')
    cm = plt.get_cmap('coolwarm')
    for ii in range(len(em_params)):
        # noinspection PyProtectedMember
        h_i, f_th = em_params[ii]._theoretical_spectrum()
        for jj in range(len(h_i)):
            cl = int(ii / len(em_params) * 256)  # y further 0~Convert to 255
            plt.plot(f_th, 10 * np.log10(h_i[jj]), c=cm(cl))
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB)')


def innovations_plot(osc, y, innovations, a, b, add_freq=None, add_radius=None, plot_all_poles=False, ax_limit=None,
                     fig_size=(8, 6), horizontal=False, bw=1, counter=0):
    """ Plot fitted OscillatorModel and the innovations / one-step prediction error for that model """
    psd_mt, fy_hz = fast_psd_multitaper(innovations, osc.Fs, 0, osc.Fs / 2, bw)
    psd_ar, _ = get_ar_psd(osc.Fs, a, b, ar_hz=fy_hz)
    psd_y, _ = fast_psd_multitaper(y.squeeze(), osc.Fs, 0, osc.Fs / 2, bw)

    if horizontal:
        fig, [ax0, ax1] = plt.subplots(1, 2, figsize=fig_size)
    else:
        fig, [ax0, ax1] = plt.subplots(2, 1, sharex='all', figsize=fig_size)
    ax0.plot(fy_hz, 10 * np.log10(psd_y), color='black', label='observed data')

    kalman_out = osc.dejong_filt_smooth(y, return_dict=True)

    # noinspection PyProtectedMember
    h_th, f_th = osc._oscillator_spectra('theoretical', kalman_out['x_t_n'], bw)
    for ii in range(len(h_th)):
        ax0.plot(f_th, 10 * np.log10(h_th[ii]), label='osc %d fit, %0.2f Hz' % (ii + 1, osc.freq[ii]))
    ax0.legend(loc='upper right')
    ax0.set_ylabel('PSD (dB)')
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
        r = np.roots(np.concatenate(([1], a)))
        sampling_constant = osc.Fs / 2 / np.pi
        root_freqs = np.abs(np.arctan2(np.imag(r), np.real(r)) * sampling_constant)  # force to positive frequencies
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
    ax2.legend(lines + lines2, labels + labels2, loc='lower center')

    fig.tight_layout()  # otherwise, the right y-label is slightly clipped
    plt.show()

    return ax1.get_ylim(), fig


def plot_fit_line(fig, slope, intercept):
    """ Add a linear fit line to an existing innovation plot figure """
    ax = fig.axes[1]
    x_limits = ax.get_xlim()
    ax.plot(x_limits, [x * slope + intercept for x in x_limits], label='linear fit', color='black')
    ax.set_xlim(x_limits)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = fig.axes[2].get_legend_handles_labels()
    fig.axes[2].legend(lines + lines2, labels + labels2, loc='lower center')

    fig.show()
    return fig
