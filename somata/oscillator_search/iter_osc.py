"""
Author: Amanda Beck <ambeck@mit.edu>
        Mingjian He <mh1@stanford.edu>
"""

from .helper_functions import (
    np, initialize_newosc, plot_innovations,
    get_innovations, get_knee, aic_calc, fit_ar
    )
from ..basic_models import OscillatorModel as Osc
from .decomp_osc import DecomposedOscillatorModel
from ..utils import estimate_r


class IterativeOscillatorModel(DecomposedOscillatorModel):
    """
    IterativeOscillatorModel is an object class containing fitted OscillatorModel objects
    from the iterative oscillator search algorithm
    """
    def __init__(self, y, fs, noise_start=None, osc_range=7, iterate=False, **kwargs):
        """
        Inputs:
            :param y: observed data
            :param fs: sampling frequency (Hz)
            :param noise_start: frequency (Hz) above which white noise is assumed to estimate R
            :param osc_range: maximum number of oscillators
            :param iterate: shortcut to apply the method iterate() on the initialized IterativeOscillatorModel
            :param **kwargs: key-word pair arguments for iterate()
        """
        if noise_start is None:
            noise_start = fs / 2 - 20
            if noise_start <= 0:
                noise_start = fs / 2 - 5
        assert noise_start > 0, 'Please redefine noise_start. Entered or calculated frequency is less than zero.'

        # initialize the first oscillator
        add_freq, add_radius, add_sigma2, *_ = initialize_newosc(fs, y)
        R = estimate_r(y, fs, freq_cutoff=noise_start)
        init_params = {'y': y, 'a': add_radius, 'freq': add_freq, 'sigma2': add_sigma2, 'Fs': fs, 'R': R}

        # populate instance attributes
        self.noise_start = noise_start
        self.osc_range = osc_range
        self.freq_res = None
        self.keep_param = ()

        self.added_osc = [Osc(**init_params)]
        self.fitted_osc = []
        self.priors = []
        self.all_params = []  # a list of lists containing Osc objects with only parameters (y dropped)
        self.ll = []
        self.ll_xre = []
        self.AIC = []
        self.knee_index = None
        self.reiterated = False

        self.iterate(**kwargs) if iterate else None  # perform learning iterations

    def iterate(self, freq_res=1, keep_param=(), reiterate=False, sigma2_method='eig',
                freq_hp=0, R_hp=0.1, Q_hp=None, R_sigma2='constant', Q_sigma2='MLE',
                no_priors=False, track_params=False, plot_fit=False, verbose=False):
        """
        Iterative Oscillator Search (iOsc) Algorithm

        Reference:
            Beck, A. M., He, M., Gutierrez, R. G., & Purdon, P. L. (2022). An iterative
            search algorithm to identify oscillatory dynamics in neurophysiological
            time series. bioRxiv, 2022-10.

            Matsuda & Komaki (2017). Time Series Decomposition into Oscillation
            Components and Phase Estimation. Journal of Neural Computation, 29, 332-367.

            Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series
            smoothing and forecasting using the EM algorithm. Journal of time series
            analysis, 3(4), 253-264.

        Inputs:
        :param self: IterativeOscillatorModel class instance
        :param freq_res: minimal oscillator frequency spacing/resolution when adding a new oscillator
        :param keep_param: a tuple of strings for parameters to keep and not update
        :param reiterate: whether to perform reiterations by sorting log-likelihood increases in descending order
        :param sigma2_method: method used to initialize sigma2 of oscillators
        :param freq_hp: concentration hyperparameter in Von Mises Prior (freq_hp * data length)
        :param R_hp: hyperparameter in Inverse Gamma Prior determining weight on prior mode compared to MLE
        :param Q_hp: hyperparameter in Inverse Gamma Prior determining weight on prior mode compared to MLE
        :param R_sigma2: determine prior mode for R, if None it will be noise variance from previous iteration
        :param Q_sigma2: determine prior mode for Q, if None all components use noise variances from previous iteration
        :param no_priors: override all other prior hyperparameters and proceed with no priors
        :param track_params: save parameters from EM iterations
        :param plot_fit: plot the fitted oscillators and innovations at each iteration
        :param verbose: print oscillator information during iterative search iterations
        """
        # Store iteration settings as instance attributes
        self.freq_res = freq_res
        self.keep_param = keep_param

        # Copy the first added oscillator to preserve its parameters
        o1 = self.added_osc[0].copy()

        # Set up iteration variables
        start_add_osc = False
        innov_limit = None
        counter = 0

        # Begin iterations to search for more oscillators in the observed data
        while o1.ncomp < self.osc_range:
            if not start_add_osc:  # do EM learning on the initial oscillator first, don't add oscillators yet
                start_add_osc = True
                if verbose:
                    print('EM learning on the initial oscillator')

            else:  # add an oscillator
                # find existing frequency
                if abs(o1.freq[-1] - self.added_osc[-1].freq) >= self.freq_res:
                    existing_freqs = list(o1.freq) + [x.freq[0] for x in self.added_osc[:-1]]
                else:
                    existing_freqs = list(o1.freq) + [x.freq[0] for x in self.added_osc]

                add_freq, add_radius, add_sigma2, _, _, a, p = initialize_newosc(
                    o1.Fs, innovations, existing_freqs=np.abs(existing_freqs), freq_res=self.freq_res,  # noqa: F821
                    ar_order=self.osc_range * 2 - 1, sigma2=sigma2_method)

                if verbose:
                    print('AR: add freq = %0.2f Hz, add radius = %0.3f, add sigma2 = %0.3f'
                          % (add_freq, add_radius, add_sigma2))

                # plot the fitted oscillator from previous iteration and the innovation
                if plot_fit:
                    _, innov_limit = plot_innovations(o1, o1.y, innovations, a, p, add_freq, add_radius,  # noqa: F821
                                                      ax_limit=innov_limit, counter=counter)

                # construct an additional oscillator
                o2 = Osc(a=add_radius, freq=add_freq, sigma2=add_sigma2, Fs=o1.Fs)
                self.added_osc.append(o2)
                o1.append(o2)  # add the additional oscillator to the existing oscillator instance
                counter += 1  # accumulate iteration counter

            # Initialize priors for EM learning
            if no_priors:
                priors = None
            else:  # provide some informative priors to constrain parameter updates
                prior_params = {'R_hyperparameter': R_hp, 'Q_hyperparameter': Q_hp,
                                'R_sigma2': R_sigma2, 'Q_sigma2': Q_sigma2}
                if R_sigma2 == 'constant':  # use the same prior mode throughout iterations
                    prior_params['R_sigma2'] = self.added_osc[0].R[0, 0]
                if verbose:
                    print('Prior input parameters:', prior_params)
                priors = o1.initialize_priors(kappa=o1.ntime * freq_hp, **prior_params)
            self.priors.append(priors)  # store the priors dictionary

            # Run EM iterations
            em_params = []  # store the oscillator parameters throughout EM iterations
            for _ in range(50):
                _ = o1.m_estimate(**o1.dejong_filt_smooth(EM=True), priors=priors, keep_param=self.keep_param)
                if track_params:
                    em_params.append(o1.copy(drop_y=True))
            if track_params:
                self.all_params.append(em_params)

            self.fitted_osc.append(o1.copy(drop_y=True))

            if verbose:
                print('Iteration %d completed:' % counter)
                print(o1)

            # Find the innovation spectrum
            innovations, ll = get_innovations(o1)
            self.ll.append(ll)
            self.AIC.append(aic_calc(o1, ll))  # compute AIC

        if reiterate:
            # Perform reiterations
            self.reiterate(track_params=track_params, plot_fit=plot_fit, verbose=verbose)
        else:
            # Find the knee_index for the selected model of fitted oscillators
            self.knee_index = get_knee(self.ll)

    def reiterate(self, track_params=False, plot_fit=False, verbose=False):
        """ Sort and re-add the oscillators in descending order of log-likelihood increase """
        # Fast-forward to after fitting the first oscillator
        o1 = self.fitted_osc[0].copy()
        o1.y = self.added_osc[0].y.copy()  # add the observed data back in

        # Fill in the instance attributes for the first fitted oscillator
        added_osc_re = self.added_osc[:1]
        fitted_osc_re = self.fitted_osc[:1]
        priors_re = self.priors[:1]
        all_params_re = self.all_params[:1]
        innovations, ll = get_innovations(o1)
        ll_re = [ll]
        AIC_re = [aic_calc(o1, ll)]

        # Set up iteration variables
        add_indices = np.diff(self.ll).argsort()[::-1] + 1
        innov_limit = None
        counter = 0

        # Reiterate through all added oscillators
        for add_idx in add_indices:
            # construct the oscillator to be added next
            add_radius = self.fitted_osc[add_idx].a[-1]
            add_freq = np.abs(self.fitted_osc[add_idx].freq[-1])
            add_sigma2 = self.fitted_osc[add_idx].sigma2[-1]
            o2 = Osc(a=add_radius, freq=add_freq, sigma2=add_sigma2, Fs=o1.Fs)

            if verbose:
                print('Re-iteration: add freq = %0.2f Hz, add radius = %0.3f, add sigma2 = %0.3f'
                      % (o2.freq[0], o2.a[0], o2.sigma2[0]))

            # plot the fitted oscillator from previous iteration and the innovation
            if plot_fit:
                a, p = fit_ar(np.squeeze(innovations), self.osc_range * 2 - 1)
                _, innov_limit = plot_innovations(o1, o1.y, innovations, a, p, o2.freq[0], o2.a[0], reiterate=True,
                                                  ax_limit=innov_limit, counter=counter)

            added_osc_re.append(o2)
            o1.append(o2)  # add the additional oscillator to the existing oscillator instance
            counter += 1  # accumulate iteration counter

            # Initialize priors for EM learning
            if self.priors[add_idx] is None:  # then no_priors was True during iterate()
                priors = None
            else:
                prior_params = self.priors[add_idx][0].copy()
                kappa = prior_params['vmp_param']['kappa']
                del prior_params['vmp_param']
                if verbose:
                    print('Prior input parameters:', prior_params)
                priors = o1.initialize_priors(kappa=kappa, **prior_params)
            priors_re.append(priors)

            # Run EM iterations
            em_params = []  # store the oscillator parameters throughout EM iterations
            for x in range(50):
                _ = o1.m_estimate(**o1.dejong_filt_smooth(EM=True), priors=priors, keep_param=self.keep_param)
                if track_params:
                    em_params.append(o1.copy(drop_y=True))
            if track_params:
                all_params_re.append(em_params)

            fitted_osc_re.append(o1.copy(drop_y=True))

            if verbose:
                print('Re-iteration %d completed:' % counter)
                print(o1)

            # Find the innovation spectrum
            innovations, ll = get_innovations(o1)
            ll_re.append(ll)
            AIC_re.append(aic_calc(o1, ll))  # compute AIC

        # Overwrite the instance attributes from iterate()
        self.added_osc = added_osc_re
        self.fitted_osc = fitted_osc_re
        self.priors = priors_re
        self.all_params = all_params_re
        self.ll_xre = self.ll  # save the log likelihoods before reiterate()
        self.ll = ll_re  # overwrite the log likelihoods after reiterate()
        self.AIC = AIC_re

        # Find the new knee_index for the selected model of re-fitted oscillators
        self.knee_index = get_knee(self.ll)  # knee replacement

        # Update the reiterated flag
        self.reiterated = True

    def __repr__(self):
        """ Unambiguous and concise representation when calling IterativeOscillatorModel() """
        return 'IterOsc(' + str(len(self)) + ')<' + hex(id(self))[-4:] + '>'
