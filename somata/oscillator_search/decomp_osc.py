"""
Author: Mingjian He <mh1@stanford.edu>
"""

from .helper_functions import (
    np, plt,
    initialize_allosc, plot_mtm, plot_innovations,
    plot_residual, plot_acf, plot_pacf,
    get_innovations, get_knee, get_ar_psd,
    fit_ar, plot_fit_line, ebic_calc
    )
import pandas as pd
from ..utils import estimate_r


class DecomposedOscillatorModel(object):
    """
    DecomposedOscillatorModel is an object class containing fitted OscillatorModel objects
    from the decomposed oscillator search algorithm
    """
    def __init__(self, y, fs, noise_start=None, osc_range=7, iterate=False, **kwargs):
        """
        Inputs:
            :param y: observed data
            :param fs: sampling frequency (Hz)
            :param noise_start: frequency (Hz) above which white noise is assumed to estimate R
            :param osc_range: maximum number of oscillators
            :param iterate: shortcut to apply the method iterate() on the initialized DecomposedOscillatorModel
            :param **kwargs: key-word pair arguments for iterate()
        """
        if noise_start is None:
            noise_start = fs / 2 - 20
            if noise_start <= 0:
                noise_start = fs / 2 - 5
        assert noise_start > 0, 'Please redefine noise_start. Entered or calculated frequency is less than zero.'
        R = estimate_r(y, fs, freq_cutoff=noise_start)

        # populate instance attributes
        self.noise_start = noise_start
        self.osc_range = osc_range
        self.keep_param = ()

        self.added_osc = initialize_allosc(fs, y, R, osc_range=osc_range)
        self.fitted_osc = []
        self.ll = []
        self.ebic = []
        self.knee_index = None

        self.iterate(**kwargs) if iterate else None  # perform learning iterations

    def reset(self):
        """ Re-instantiate the instance object with the same data """
        self.__init__(self.added_osc[0].y, self.added_osc[0].Fs, noise_start=self.noise_start, osc_range=self.osc_range)

    def iterate(self, freq_res=None, keep_param=(), preiterate=False,
                R_hp=0.1, model_select='knee', plot_fit=False, verbose=None):
        """
        Decomposed Oscillator Search (dOsc) Algorithm

        Inputs:
        :param self: DecomposedOscillatorModel class instance
        :param freq_res: not used but kept for inheritance by the OscillatorModel class
        :param keep_param: a tuple of strings for parameters to keep and not update
        :param preiterate: prefit the oscillators to be added and sort in descending order of peak PSD
        :param R_hp: hyperparameter in Inverse Gamma Prior determining weight on prior mode compared to MLE
        :param model_select: 'knee', 'max', or 'ebic' to select with the knee, maximum likelihood, or minimum BIC
        :param plot_fit: plot the fitted oscillator spectra at each iteration
        :param verbose: not used but kept for inheritance by the OscillatorModel class
        """
        # Prefit the AR eigenmodes as oscillators and sort them in descending order of peak PSD
        self.added_osc = self.preiterate(keep_param=keep_param, R_hp=R_hp) if preiterate else self.added_osc

        while len(self.fitted_osc) < self.osc_range:
            o1 = self.added_osc[0].copy()
            _ = [o1.append(x) for x in self.added_osc[1:len(self.fitted_osc) + 1]]

            # Run EM iterations
            priors = o1.initialize_priors(kappa=0, R_hyperparameter=R_hp, R_sigma2=o1.R[0, 0], Q_sigma2='MLE')
            for _ in range(50):
                _ = o1.m_estimate(**o1.dejong_filt_smooth(EM=True), priors=priors, keep_param=keep_param)

            if plot_fit:
                fig = o1.visualize_freq('theoretical', bw=1)
                fig.get_axes()[0].set_title('Iteration %d' % len(self.fitted_osc))
                fig.set_tight_layout(True)

            self.fitted_osc.append(o1.copy(drop_y=True))
            self.ll.append(o1.dejong_filt_smooth(return_dict=True)['logL'].sum())
            self.ebic.append(ebic_calc(osc=o1, ll=self.ll[-1], osc_range=self.osc_range, gamma=0))

        # Find the knee_index for the selected model of fitted oscillators
        if model_select == 'knee':
            self.knee_index = get_knee(self.ll)
        elif model_select == 'max':
            self.knee_index = np.argmax(self.ll)
        elif model_select == 'ebic':
            self.knee_index = np.argmin(self.ebic)
        else:
            raise ValueError('model_select must be either "knee", "max", or "ebic"')

    def preiterate(self, keep_param=(), R_hp=0.1):
        """ Prefit the oscillators to be added and sort in descending order of peak PSD """
        # first fit with all oscillators to be added at once
        o1 = self.added_osc[0].copy()
        _ = [o1.append(x) for x in self.added_osc[1:]]  # use all available oscillators

        priors = o1.initialize_priors(kappa=0, R_hyperparameter=R_hp, R_sigma2=o1.R[0, 0], Q_sigma2='MLE')
        for _ in range(50):
            _ = o1.m_estimate(**o1.dejong_filt_smooth(EM=True), priors=priors, keep_param=keep_param)
        o1.fill_components()  # fill component parameters after learning

        # get the theoretical spectral peaks for each fitted oscillator
        peak_psd = np.squeeze(np.array([c._theoretical_spectrum(f_hz=c.freq[0])[0] for c in o1.components]))

        # create a new list of oscillators to be added in descending order of peak PSD
        add_indices = np.argsort(peak_psd)[::-1]
        added_osc = [o1.components[add_indices[0]].copy()]  # keep only one copy of y
        for ii in range(1, len(add_indices)):
            added_osc.append(o1.components[add_indices[ii]].copy(drop_y=True))

        return added_osc

    def get_knee_osc(self):
        """ Return the OscillatorModel at the selected knee index """
        return self.fitted_osc[self.knee_index]

    def get_residual(self):
        """ Compute the one-step prediction error from the best model """
        y = self.added_osc[0].y
        osc = self.get_knee_osc()
        residual, _ = get_innovations(osc, y=y)
        return residual

    def get_residual_fit(self, residual=None):
        """
        Fit a straight line to the residuals from the best model using
        a theoretical AR model spectrum
        """
        # Obtain the theoretical spectrum of a fitted AR model
        residual = self.get_residual() if residual is None else residual
        a, p = fit_ar(np.squeeze(residual), self.osc_range * 2 - 1)
        ar_psd, ar_hz = get_ar_psd(self.get_knee_osc().Fs, a, p)

        # Fit a straight line in semi-log space, i.e., log(PSD) against linear frequency (Hz)
        from scipy import stats
        slope, intercept, *_ = stats.linregress(ar_hz, 10 * np.log10(ar_psd))

        return slope, intercept

    def get_fit_spectra(self, bw=1):
        """ Compute the theoretical and empirical spectra of oscillators from the best model """
        selected_osc = self.get_knee_osc()
        spectra_theo = selected_osc.oscillator_spectra('theoretical')
        kalman_out = selected_osc.dejong_filt_smooth(y=self.added_osc[0].y, return_dict=True)
        spectra_empi = selected_osc.oscillator_spectra('actual', kalman_out['x_t_n'][:, 1:], bw)
        return spectra_theo, spectra_empi

    def diagnose_residual_acf(self):
        """
        Diagnose autocorrelations that might be present in the residuals
        using a series of Portmanteau tests (meaning they have unspecified
        alternative hypothesis but a well-defined null hypothesis)
        """
        from statsmodels.stats.stattools import durbin_watson
        from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_lm
        from statsmodels.tsa.stattools import bds

        residual = self.get_residual()
        osc = self.get_knee_osc()

        # Durbin-Watson test for AR(1) process
        dw_dstat = durbin_watson(residual)
        # Ljung-Box test for autocorrelations up to 1s lags
        lb_df = acorr_ljungbox(residual, lags=[osc.Fs], model_df=3*osc.ncomp)
        # Lagrange Multiplier test for autocorrelations up to second order
        lm_results = acorr_lm(residual, nlags=2, ddof=3*osc.ncomp)
        # Brock-Dechert-Scheinkman test for iid. of residuals at second order embedding
        bds_results = bds(residual, max_dim=3)

        # Create a dataframe to store the results
        df = pd.DataFrame({'Durbin-Watson d': dw_dstat,
                           'Ljung-Box': lb_df['lb_pvalue'].values[0],
                           'Lagrange Multiplier': lm_results[1],
                           'Brock-Dechert-Scheinkman': bds_results[1][1]},
                          index=['p-value'])

        return df

    def diagnose_residual_norm(self):
        """ Diagnose whether residuals are white noise using normality tests """
        from scipy.stats import zscore, shapiro, normaltest, anderson, cramervonmises

        residual = self.get_residual()

        # Shapiro-Wilk test for normality
        sw_results = shapiro(residual)
        # D'Agostino and Pearson test for normality
        dp_results = normaltest(residual, nan_policy='omit')
        # Anderson-Darling test for normality using critical value at 5% significance level
        ad_results = anderson(residual, dist='norm')
        ad_reject_null = ad_results.statistic >= ad_results.critical_values[2]
        # Cramer-von Mises test for normality after z-scoring the residuals
        cvm_results = cramervonmises(zscore(residual), cdf='norm')

        # Create a dataframe to store the results
        df = pd.DataFrame({'Shapiro-Wilk': sw_results.pvalue,
                           'D\'Agostino and Pearson': dp_results.pvalue,
                           'Anderson-Darling H': ad_reject_null,
                           'Cramer-von Mises': cvm_results.pvalue},
                          index=['p-value'])

        return df

    def diagnose_residual(self):
        """ Generate diagnostic plots and statistics on the residuals """
        from IPython.display import display

        _ = self.plot_residual()
        _ = self.plot_residual_fit()
        _ = self.plot_acf()
        _ = self.plot_pacf()

        print('>>> Portmanteau tests for autocorrelations:')
        display(self.diagnose_residual_acf())

        print('>>> Normality tests for white residuals:')
        display(self.diagnose_residual_norm())

        return

    def plot_residual(self, **kwargs):
        """ Plot the best fit model and its residuals against white noises """
        fig = plot_residual(self.get_knee_osc(), self.added_osc[0].y, self.get_residual(), **kwargs)
        return fig

    def plot_residual_fit(self, ax=None):
        """ Plot the one-step prediction error fitted with an AR model and a straight line """
        residual = self.get_residual()
        slope, intercept = self.get_residual_fit(residual=residual)
        a, p = fit_ar(np.squeeze(residual), self.osc_range * 2 - 1)

        fig, _ = plot_innovations(self.get_knee_osc(), self.added_osc[0].y,
                                  residual, a, p, ax=ax, figsize=(6.4, 4.8), counter=self.knee_index)
        ax1 = fig.axes[1] if ax is None else ax[1]
        fig = plot_fit_line(fig, slope, intercept, ax=ax1)
        return fig

    def plot_acf(self, ax=None):
        """ Plot the autocorrelation function of the residuals """
        fig = plot_acf(self.get_residual(), Fs=self.get_knee_osc().Fs, ax=ax)
        return fig

    def plot_pacf(self, ax=None):
        """ Plot the partial autocorrelation function of the residuals """
        fig = plot_pacf(self.get_residual(), Fs=self.get_knee_osc().Fs, ax=ax)
        return fig

    def plot_fit_spectra(self, bw=1, ax=(None, None)):
        """ Plot the fitted theoretical and empirical spectra """
        selected_osc = self.get_knee_osc()
        if ax[0] is not False:
            fig_theo = selected_osc.visualize_freq('theoretical', bw=bw, y=self.added_osc[0].y, ax=ax[0])
        else:
            fig_theo = None
        if ax[1] is not False:
            fig_empi = selected_osc.visualize_freq('actual', bw=bw, y=self.added_osc[0].y, ax=ax[1])
        else:
            fig_empi = None
        return fig_theo, fig_empi

    def plot_log_likelihoods(self, ax=None):
        """ Plot the trajectory of log likelihoods from iterations """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        ax.plot(range(len(self.fitted_osc)), self.ll, '-*', label='__nolegend__')
        ax.scatter(self.knee_index, self.ll[self.knee_index], color='red', zorder=2, label='knee')
        ax.set_xlabel('Index of Model')
        ax.grid('on')
        ax.set_title('Log Likelihood')
        return fig

    def plot_mtm(self, **kwargs):
        """ Plot multitaper spectrogram and mean spectrum """
        return plot_mtm(self, **kwargs)

    def plot_trace(self, ax=None):
        """ Plot the time trace """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()
        ax.plot(self.added_osc[0].y[0, :])
        ax.set_xlabel('Samples')
        ax.set_ylabel('Voltage')
        ax.set_title('Time Trace')
        return fig

    def __repr__(self):
        """ Unambiguous and concise representation when calling DecomposedOscillatorModel() """
        return 'DecOsc(' + str(len(self)) + ')<' + hex(id(self))[-4:] + '>'

    def __str__(self):
        """ Helpful information when calling print(DecomposedOscillatorModel()) """
        print_str = "number of oscillators = %d\n " % len(self)
        if self.fitted_osc:
            print_str += str(self.fitted_osc[-1])
        else:
            print_str += 'There is no fitted oscillator. Initial oscillator:\n '
            print_str += str(self.added_osc[0])
        return print_str

    def __len__(self):
        return self.added_osc[0].ncomp + len(self.fitted_osc) - 1
