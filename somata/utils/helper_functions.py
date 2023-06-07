"""
Author: Mingjian He <mh105@mit.edu>

general helper functions used throughout SOMATA
"""


import numpy as np
from itertools import groupby
import scipy.signal as signal


def consecutive(data):
    """
    This function takes a list of values and returns a list of tuples.
    Each tuple contains the value, the start index, and the end index
    of a consecutive sequence of that value.
    Parameters
    ----------
    data : list or ndarray
        A list of values.
    Returns
    -------
    list
        A list of tuples. Each tuple contains the value, the start index,
        and the end index of a consecutive sequence of that value.
    Examples
    --------
    >> consecutive([1, 1, 1, 2, 2, 3, 3, 3, 3])
    [(1, 0, 2), (2, 3, 4), (3, 5, 8)]
    """
    data_group = [(v, sum(1 for _ in g)) for v, g in groupby(data)]
    vals, cons = zip(*data_group)

    start_inds = np.cumsum((0,) + cons[:-1])
    end_inds = start_inds + cons - 1

    return list(zip(vals, start_inds, end_inds))


def estimate_r(y, Fs, freq_cutoff, plot_freqz=False):
    """ Estimate the observation noise covariance R in a given time series """
    Fs_nq = Fs / 2  # Nyquist frequency
    wp = np.min([Fs_nq - 10, freq_cutoff])  # passband edge frequency in Hz
    ws = wp - 2  # stopband edge frequency in Hz
    N, Wn = signal.buttord(wp=wp, ws=ws, gpass=1, gstop=50, fs=Fs)  # select filter order
    sos = signal.butter(N, Wn, btype="high", output='sos', fs=Fs)  # highpass filter

    # Visualize the filter frequency response
    if plot_freqz:
        import matplotlib.pyplot as plt
        w, h = signal.sosfreqz(sos, fs=Fs)
        plt.figure()
        plt.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-3)))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(Fs_nq, color='green')
        plt.show()

    # Apply the high pass filter and estimate observation noise covariance
    y_filt = signal.sosfiltfilt(sos, y)
    R = np.cov(y_filt) * Fs_nq / (Fs_nq - wp)  # scale up based on wp

    return R
