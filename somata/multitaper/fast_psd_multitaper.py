import numpy as np
from .multitaper_spectrogram_python import multitaper_spectrogram


def fast_psd_multitaper(x, sfreq, freq_min, freq_max, bandwidth):
    """ Fast multitaper PSD estimate with windowing """
    window_length = min(x.size / sfreq, 10)  # max to 10s windows
    time_bandwidth = np.max([1., bandwidth / 2 * window_length])
    spect, _, freq = multitaper_spectrogram(data=x, fs=sfreq, frequency_range=[freq_min, freq_max],
                                            time_bandwidth=time_bandwidth,
                                            window_params=[window_length, 1], detrend_opt='linear',
                                            multiprocess=True, plot_on=False, verbose=False)
    psd = np.mean(spect, 1) * sfreq  # scale back the normalization by sampling frequency
    return psd, freq
