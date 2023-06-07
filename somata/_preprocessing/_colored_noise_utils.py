# Author: Proloy Das <pdas6@mgh.harvard.edu>

from functools import partial
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import zpk2sos, sosfilt


def sxx(theta, w):
    temp = theta[2] / (1 - theta[2]) ** 2
    return theta[0] + theta[1] / (1 + 4 * temp * np.sin(w/2)**2)


def psd_multitaper(x, sfreq, fmin, fmax, bandwidth, adaptive=True,
                   low_bias=True):
    """Compute power spectral density (PSD) using a multi-taper method.
    Parameters
    ----------
    x : array, shape=(..., n_times)
        The data to compute PSD from.
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD
        (slow, use n_jobs >> 1 to speed up computation).
    low_bias : bool
        Only use tapers with more than 90%% spectral concentration within
        bandwidth.

    The PSD is only normalized by the length of the signal
    (as opposed to by the sampling rate and length of the
    signal as done in nitime).
    """
    from mne.time_frequency import psd_array_multitaper
    psds, freqs = psd_array_multitaper(x, sfreq,
                                       fmin=fmin, fmax=fmax,
                                       bandwidth=bandwidth,
                                       adaptive=adaptive,
                                       low_bias=low_bias,
                                       normalization='length',
                                       n_jobs=4)
    return psds, freqs
                

def _fit_params(ws, Pxx, fit_db=False, Aw=None):
    if Aw is None:
        Aw = Pxx.mean()
    print('Initial Observation Noise Variance: %0.3f \nMinimum Observation Noise Variance: %0.3f' % (Aw, Aw/2))
    if fit_db:
        return least_squares(lambda theta: 10*np.log10(sxx(theta, ws)) - 10*np.log10(Pxx), [Aw, 1, 0.5],
                             bounds=(np.array([Aw / 2, 0., -1.]), np.array([np.inf, np.inf, 1.])), loss='cauchy')
    else:
        return least_squares(lambda theta: sxx(theta, ws) - Pxx, [Aw, 1, 0.5],
                             bounds=(np.array([Aw / 2, 0., -1.]), np.array([np.inf, np.inf, 1.])), loss='cauchy')


def _spectral_factorization(Aw, Ac, q):
    tt = Ac / Aw
    D = - 2 * tt + (tt + 1) * (1+q**2) / q
    D /= 2
    gamma = D - np.sqrt(D ** 2 - 1)  
    K = Aw * q / gamma
    k = K ** (1/2)
    whitener_sos = zpk2sos([q], [gamma], 1/k)  
    colorer_sos = zpk2sos([gamma], [q],  k)  
    return whitener_sos, colorer_sos, (D, gamma, K)


def _setup_filters(whitener_sos, colorer_sos):
    """
    Use of sosfiltfilt is not desired since they will flatten the spectrum
    twice. Also, phase is not big of a concern here.  

    signature of the returned functions:
    fun(x, padtype='odd', padlen=None)
    for information about padtype, padlen see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html#scipy.signal.sosfiltfilt
    """
    return [partial(sosfilt, sos, axis=-1) for sos in (whitener_sos, colorer_sos)]


def setup_filters(x, sfreq, fmin, fmax):
    """Returns whitener and colorer for AR(1) + white noise model over [fmin, fmax]
    Parameters
    ----------
    x : array, shape=(..., n_times)
        The data to compute PSD from.
    sfreq : float
        The sampling frequency.
    fmin : float
        The lower frequency of interest.
    fmax : float
        The upper frequency of interest.
    
    Returns
    -------
    whitener : function
        The whitener filter with signature fun(x, padtype='odd', padlen=None)
    colorer : function
        The colorer filter with signature fun(x, padtype='odd', padlen=None)

    Notes
    -----
    A simple noise model (AR(1) + white noise) is used to quantify the nature of 
    temporal correlations in neural data (i.e. EEG, MEG, LFP) recordings. The AR
    component represents the excess low-frequency noise observed in these time
    series data, while the white noise component represents scanner noise. The
    returned filters are the _whitener_ and _colorer_ respectively, and should be
    used before and after time-domain or frequency domain analysis respectively:   
             ----------      ---------------------       ---------
    data -> | whitener | -> | analysis / modeling | ->  | colorer | -> output
             ----------      ---------------------       ---------
    For more info, see:
    [1] Purdon, P. L., & Weisskoff, R. M. (1998). Effect of temporal autocorrelation
        due to physiological noise and stimulus paradigm on voxel-level false-
        positive rates in fMRI. Human Brain Mapping, 6(4), 239–249.
        https://doi.org/10.1002/(SICI)1097-0193(1998)6:4<239::AID-HBM4>3.0.CO;2-4
    
    Parameters
    ----------
    x: ndarray
        data with last dimension as time
    
    Returns
    -------
    whitener: function
        partial sosfiltfilt object with sos, and axis=-1 fixed  
    colorer: function   
        partial sosfiltfilt object with sos, and axis=-1 fixed
    
    Example: 
    ```
    whitener, colorer = setup_filters(x) TODO: this example call is missing required arguments
    ```
    signature of the returned functions:
    fun(x, padtype='odd', padlen=None)
    for information about padtype, padlen see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html#scipy.signal.sosfiltfilt
    """
    Pxx, freqs = psd_multitaper(x, sfreq, fmin, fmax, bandwidth=2)
    ws = (freqs / sfreq) * 2 * np.pi
    res = _fit_params(ws, Pxx[0])  # Pxx[0] -> np.squeeze(Pxx)
    *sos, (_, gamma, _) = _spectral_factorization(*res.x)
    if np.abs(gamma) > 1.0:
        raise ValueError(f'gamma (={gamma}) cannot have magnitude greater than 1.') 
    return _setup_filters(*sos)


def test_fit_params():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed=0)
    Aw = 1.8; Ac = 3.6; q = 0.65; noise = 1.
    ws = np.arange(0, np.pi, 2*np.pi/100)
    Pxx = sxx([Aw, Ac, q], ws) + noise * rng.normal(size=ws.shape[0])
    res1 = _fit_params(ws, Pxx)
    fig, ax = plt.subplots()
    ax.plot(ws, Pxx, '-o', label='Aw = %.2f, Ac = %.2f, q = %.2f' % (Aw, Ac, q))
    ax.plot(ws, sxx(res1.x, ws), label='Aw = %.2f, Ac = %.2f, q = %.2f' % tuple(res1.x))
    ax.legend()
    sos1, sos2, params = _spectral_factorization(*res1.x)
    assert np.allclose(params, np.array([1.122089889072777, 0.6130852533774259, 2.597274197197961]))
    fig.show()


def test_filters():
    try:
        import mne
    except ImportError:
        return
    import os
    import matplotlib.pyplot as plt

    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                        'sample_audvis_raw.fif')
    # the preload flag loads the data into memory now
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
    raw.crop(tmax=120.).pick_channels(['EEG 020'])
    raw.filter(None, h_freq=50.).resample(100.)
    x = raw.get_data() * 1e6
    sfreq = raw.info['sfreq']

    fmin = 40.0
    fmax = sfreq/2 - 1
    pxx, freqs = psd_multitaper(x, sfreq, fmin, fmax, bandwidth=2.)
    ws = (freqs / sfreq) * 2 * np.pi
    res1 = _fit_params(ws, pxx[0])
    fig, ax = plt.subplots()
    ax.plot(ws, pxx.T, '-o', label='mtm-spectrum')
    ax.plot(ws, sxx(res1.x, ws), label='fit Aw = %.2f, Ac = %.2f, q = %.2f' % tuple(res1.x))
    ax.legend()
    fig.show()
    sos1, sos2, params = _spectral_factorization(*res1.x)
    whitener, colorer = _setup_filters(sos1, sos2)
    whitened_x = whitener(x)
    res = np.vstack((x, whitened_x))
    pxx, freqs = psd_multitaper(res, sfreq, 0., sfreq/2, bandwidth=5.)
    fig, ax = plt.subplots()
    ws = (freqs / sfreq) * 2 * np.pi
    ax.plot(ws, 10*np.log10(pxx[0]), '.-', label='mtm-spectrum')
    ax.plot(ws, 10*np.log10(pxx[1]), '.-', label='whitened-mtm-spectrum')
    ax.plot(ws, 10*np.log10(sxx(res1.x, ws)), label='fit Aw = %.2f, Ac = %.2f, q = %.2f' % tuple(res1.x))
    ax.legend()
    fig.show()

    whitener_, colorer_ = setup_filters(x, sfreq, fmin, fmax)
    whitened_x_ = whitener_(x)
    assert np.allclose(whitened_x, whitened_x_)
