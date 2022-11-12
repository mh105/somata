""" Author: Amanda Beck <ambeck@mit.edu> """

import numpy as np
from numpy import random
from numpy.linalg import eig, inv
from scipy.linalg import toeplitz
from spectrum import aryule, arma_estimate, arma2psd, arburg
import matplotlib.pyplot as plt
from kneed import KneeLocator
from ..multitaper import fast_psd_multitaper
from ..basic_models import OscillatorModel as Osc


def innovations_wrapper(iter_osc, osc_num, fig_size=(8, 6), plot_all_poles=False, horizontal=False, ax_limit=None):
    """ Plots the innovations for models in the original scale, used for Beck et al. 2022 """
    osc = iter_osc.fitted_osc[osc_num]  # we're using the scaled version like in the iterations
    innovations, ll = find_innovations(osc, y=iter_osc.added_osc[0].y)

    add_freq, add_radius, all_freq, all_radii, a, b = initialize_newosc(osc.Fs, innovations, existing_freqs=None,
                                                                        ar_order=iter_osc.ar_order,
                                                                        burg_flag=iter_osc.burg_flag)

    axlim, fig = innovations_plot(osc, iter_osc.added_osc[0].y, innovations, a, b, add_freq,
                                  plot_all_poles=plot_all_poles,
                                  ax_limit=ax_limit, fig_size=fig_size, horizontal=horizontal)

    return axlim, fig


def initial_param(y, fs, noise_start, ar_order=13, burg_flag=False):
    """ Scale y data and prepare initialization of OscillatorModel object """
    R_est = initialize_r(y, fs, noise_start)
    power = np.ceil(200 / fs)
    y_scaled = y / np.sqrt(R_est)
    # TODO: test remove scaling all together and set R here to be R_est directly
    add_freq, _, _, _, _, b = initialize_newosc(fs, y_scaled, ar_order=ar_order, burg_flag=burg_flag)
    osc_init = {'y': y_scaled, 'a': 0.98 ** power, 'freq': add_freq, 'sigma2': b,
                'Fs': fs, 'R': b}
    return osc_init, R_est


def initialize_r(y, fs, noise_start, noise_end=None, variance=False):
    """ Find mean power above a certain frequency (noise_start) """
    fft_data = np.fft.fftshift(np.fft.fft(np.squeeze(y)))
    psd_data = np.abs(fft_data) ** 2 / y.size
    freq = np.linspace(-fs / 2, fs / 2, psd_data.size)

    min_idx = np.argmin(np.abs(freq - noise_start))
    max_idx = np.argmin(np.abs(freq - noise_end)) if noise_end is not None else None
    noise_spectrum = psd_data[min_idx:max_idx]
    R_estimated = np.mean(noise_spectrum)

    if variance:  # also return the variance across the specified noise frequencies
        return R_estimated, np.var(noise_spectrum)
    else:
        return R_estimated


def find_innovations(osc, y=None):
    """ Calculate innovations / one step prediction error """
    y = osc.y if y is None else y
    kalman_out = osc.dejong_filt_smooth(y=y, return_dict=True)
    x_pred = kalman_out['x_t_tmin1'][:, 1:]
    y_pred = osc.G @ x_pred
    return (y - y_pred).squeeze(), kalman_out['logL'].sum()


def initialize_newosc(fs, innovations, existing_freqs=None, ar_order=13, burg_flag=False):
    """ Fit AR model to innovations """
    if burg_flag:  # use Burg algorithm
        a, p, k = arburg(np.squeeze(innovations), ar_order)
    else:  # default is Yule Walker algorithm
        a, p, k = aryule(np.squeeze(innovations), ar_order)
    r = np.roots(np.concatenate(([1], a)))
    sampling_constant = fs / 2 / np.pi
    root_freqs = np.abs(np.arctan2(np.imag(r), np.real(r)) * sampling_constant)  # force to positive frequencies
    root_radii = np.abs(r)
    root_radii_copy = root_radii.copy()
    if existing_freqs is not None:
        root_radii_copy[np.isin(root_freqs, existing_freqs)] = -1  # ignore frequencies with existing oscillators
    largest_root_idx = np.argmax(root_radii_copy)
    add_freq = root_freqs[largest_root_idx]
    add_radius = root_radii[largest_root_idx]
    return add_freq, add_radius, root_freqs, root_radii, a, p


def initialize_arma(fs, innovations, ar_order=7):
    """ Fit ARMA model to innovations """
    a, b, rho = arma_estimate(innovations, ar_order, 1, ar_order + 5)
    r = np.roots(np.concatenate(([1], a)))
    sampling_constant = fs / 2 / np.pi
    root_freqs = np.abs(np.arctan2(np.imag(r), np.real(r)) * sampling_constant)
    root_radii = np.abs(r)
    largest_root = np.argmax(root_radii)
    return root_freqs[largest_root], root_radii[largest_root], root_freqs, root_radii, a, b, rho


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
    return kneedle.knee  # Returns int,  index of model at knee


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
    plt.plot(f_hz.squeeze(), 10 * np.log10(psd_mt.squeeze()), color='black', label='observed data')
    cm = plt.get_cmap('coolwarm')
    for ii in range(len(em_params)):
        # noinspection PyProtectedMember
        h_i = em_params[ii]._theoretical_spectrum()
        for jj in range(len(h_i)):
            cl = int(ii / len(em_params) * 256)  # y further 0~Convert to 255
            plt.plot(f_hz, 10 * np.log10(h_i[jj]), c=cm(cl))
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')


def innovations_plot(osc, y, innovations, a, b, add_freq, plot_all_poles=False, ax_limit=None, fig_size=(8, 6),
                     horizontal=False, bw=2):
    """ Plot fitted OscillatorModel and the innovations / one-step prediction error for that model """
    psd_mt, fy_hz = fast_psd_multitaper(innovations, osc.Fs, 0, osc.Fs / 2, bw)
    ar_psd = arma2psd(A=a, rho=b, NFFT=2 * fy_hz.size, T=osc.Fs, sides='centerdc')
    psd_y, freq_y = fast_psd_multitaper(y.squeeze(), osc.Fs, 0, osc.Fs / 2, bw)

    if horizontal:
        fig, [ax0, ax1] = plt.subplots(1, 2, figsize=fig_size)
    else:
        fig, [ax0, ax1] = plt.subplots(2, 1, sharex='all', figsize=fig_size)
    ax0.plot(freq_y, 10 * np.log10(psd_y), color='black', label='observed data')

    kalman_out = osc.dejong_filt_smooth(y, return_dict=True)

    # noinspection PyProtectedMember
    f_th, h_th = osc._oscillator_spectra('theoretical', kalman_out['x_t_n'], bw)
    h_sum = np.zeros((1, h_th[0].size))
    for ii in range(len(h_th)):
        ax0.plot(f_th, 10 * np.log10(2*h_th[ii]),
                 label='osc %d fit, %0.2f Hz' % (ii + 1, osc.freq[ii]))  # make spectrum one-sided
        h_sum += np.sqrt(h_th[ii])
    h_sum += np.sqrt(osc.R)
    ax0.legend(loc='upper right')
    ax0.set_ylabel('Power (dB)')
    ax0.set_title('Iteration %d' % len(h_th))

    ax1.plot(fy_hz, 10 * np.log10(psd_mt), label='OSPE', color='tab:blue')
    ax1.plot(fy_hz, 10 * np.log10(ar_psd[fy_hz.size:] * osc.Fs * 2), linestyle='dashed', color='tab:blue',
             label='AR fit')
    if ax_limit is not None:
        ax1.set_ylim(ax_limit)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (dB)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax1.set_title('One-Step Prediction Error (OSPE) - Iteration %d' % len(h_th))
    ax1.axvline(add_freq, color='black', linestyle='dashed', label='frequency of new oscillation, %0.2f Hz' % add_freq)
    # ax1.axhline(20 * np.log10(b), color='black', label='observation noise estimate')
    ax1.legend(loc='upper right')

    r = np.roots(np.concatenate(([1], a)))
    sampling_constant = osc.Fs / 2 / np.pi
    root_freqs = np.abs(np.arctan2(np.imag(r), np.real(r)) * sampling_constant)
    for rf in range(len(root_freqs)):
        if root_freqs[rf] == osc.Fs / 2:
            root_freqs[rf] = 0

    ax2 = ax1.twinx()
    if plot_all_poles:
        ax2.scatter(root_freqs, np.abs(r), color='tab:orange')
    ax2.scatter(add_freq, np.max(abs(r)), facecolor='tab:green', linewidth=2, label='pole for initialization')

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
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    fig.tight_layout()  # otherwise, the right y-label is slightly clipped
    plt.show()

    return ax1.get_ylim(), fig


def compare_arma(fs, innovations, a, b, add_freq):
    """ Plot AR and ARMA models to compare """
    add_freq1, add_radius1, root_freqs1, root_radii1, a1, b1, rho1 = initialize_arma(fs, innovations)
    psd_mt, fy_hz = fast_psd_multitaper(innovations, fs, 0, fs / 2, 1)
    ar_psd = arma2psd(A=a, rho=b, NFFT=2 * fy_hz.size, T=fs, sides='centerdc')
    arma_psd = arma2psd(A=a1, B=b1, rho=rho1, NFFT=2 * fy_hz.size, T=fs, sides='centerdc')

    fig, ax = plt.subplots(2, 1, sharex='all', figsize=(8, 8))
    ax[0].plot(fy_hz, 10 * np.log10(psd_mt), label='observed data')
    ax[0].plot(fy_hz, 10 * np.log10(arma_psd[fy_hz.size:] * fs * 2), linestyle='dashed',
               label='arma')  # sqrt(2) factor may be off
    ax[0].plot(fy_hz, 10 * np.log10(ar_psd[fy_hz.size:] * fs * 2), label='ar')
    ax[0].set_ylabel('Power (dB)')
    ax[0].set_title('Innovations')
    ax[0].axvline(add_freq, color='black', label='AR fit')
    ax[0].axvline(add_freq1, color='black', linestyle='dashed', label='ARMA fit')
    ax[0].axhline(10 * np.log10(b), color='black', label='_nolegend_')
    ax[0].axhline(10 * np.log10(rho1), color='black', linestyle='dashed', label='_nolegend_')
    ax[0].legend()

    r = np.roots(np.concatenate(([1], a)))
    sampling_constant = fs / 2 / np.pi
    root_freqs = np.abs(np.arctan2(np.imag(r), np.real(r)) * sampling_constant)
    z = np.roots(np.concatenate(([1], b1)))
    for rf in range(len(root_freqs)):
        if root_freqs[rf] == fs / 2:
            root_freqs[rf] = 0
        if root_freqs1[rf] == fs / 2:
            root_freqs1[rf] = 0

    ax[1].scatter(root_freqs, np.abs(r), color='tab:green', label='ar poles')
    ax[1].scatter(root_freqs1, root_radii1, color='tab:orange', label='arma poles')
    ax[1].scatter(0, np.abs(z), marker='^', color='tab:orange', label='arma zeroes')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Root (magnitude)')

    print('ARMA: add freq: %0.2f Hz, add radius: %0.3f, add q: %0.3f' % (add_freq1, add_radius1, rho1))


def find_a(y, p, Ri=None):
    """ Fit AR parameters based on R (Matsuda and Komaki 2017) """
    Ck = np.asarray([cov_k(y, ii) for ii in range(p + 1)])
    C = toeplitz(Ck)

    if Ri is None:
        w, v = eig(C)
        Ri = np.min(np.abs(w))
    C_new = C - Ri * np.eye(p + 1)

    a_vec = np.squeeze(inv(C_new[:-1, :-1]) @ Ck[1:, None])
    Q = Ck[0] - np.sum(a_vec * Ck[1:])

    return a_vec, Q, Ri, Ck


def cov_k(y, k):
    """ Calculate covariance at lag k """
    N = y.size
    y = y.squeeze()
    if k > 0:
        return np.sum(y[:(-k)] * y[k:]) / N
    else:
        return np.sum(y ** 2) / N
