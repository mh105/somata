"""
Author: Amanda Beck <ambeck@mit.edu>
        Mingjian He <mh105@mit.edu>
"""

from .helper_functions import *
from ..basic_models import OscillatorModel as Osc
from ..utils import estimate_r


class IterativeOscillatorModel(object):
    """
    IterativeOscillatorModel is an object class containing fitted OscillatorModel objects
    from the iterative oscillator algorithm
    """
    def __init__(self, y, fs, estimate_R=False, noise_start=None, burg_flag=False, verbose=False):
        """
        Inputs:
            :param y: observed data
            :param fs: sampling frequency (Hz)
            :param estimate_R: True will estimate empirical observation noise R
            :param noise_start: frequency (Hz) above which white noise is assumed to estimate R
            :param burg_flag: True will use Burg algorithm to fit AR parameters instead of the default Yule-Walker
            :param verbose: print initial parameters
        """
        if noise_start is None:
            noise_start = fs / 2 - 20
            if noise_start <= 0:
                noise_start = fs / 2 - 5
        assert noise_start > 0, 'Please redefine noise_start. Entered or calculated frequency is less than zero.'

        # initialize the first oscillator
        add_freq, add_radius, _, _, _, b = initialize_newosc(fs, y, ar_order=13, burg_flag=burg_flag)
        R = estimate_r(y, fs, freq_cutoff=noise_start) if estimate_R else b
        init_params = {'y': y, 'a': add_radius, 'freq': add_freq, 'sigma2': b, 'Fs': fs, 'R': R}

        if verbose:
            print('Noise starting frequency is %0.1f Hz' % noise_start)
            print('Initial oscillator parameters:')
            print(init_params)

        # populate instance attributes
        self.burg_flag = burg_flag
        self.osc_range = None
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

    def iterate(self, osc_range=7, freq_res=1, keep_param=(), reiterate=True, freq_hp=3, R_hp=0.1, Q_hp=None,
                R_sigma2='b', Q_sigma2='MLE', no_priors=False, track_params=False, plot_innov=False, verbose=False):
        """
        Iterative Oscillator Algorithm

        Reference:
            Beck, A. M., He, M., Gutierrez, R. G. & Purdon, P. L. (in prep)

            Matsuda & Komaki (2017). Time Series Decomposition into Oscillation
            Components and Phase Estimation. Journal of Neural Computation, 29, 332-367.

            Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series
            smoothing and forecasting using the EM algorithm. Journal of time series
            analysis, 3(4), 253-264.

        Inputs:
        :param self: IterativeOscillatorModel class instance
        :param osc_range: maximum number of oscillators
        :param freq_res: minimal oscillator frequency spacing/resolution when adding a new oscillator
        :param keep_param: a tuple of strings for parameters to keep and not update
        :param reiterate: whether to perform reiterations by sorting log-likelihood increases in descending order
        :param freq_hp: concentration hyperparameter in Von Mises Prior (freq_hp * data length)
        :param R_hp: hyperparameter in Inverse Gamma Prior determining weight on prior mode compared to MLE
        :param Q_hp: hyperparameter in Inverse Gamma Prior determining weight on prior mode compared to MLE
        :param R_sigma2: determine prior mode for R, if None it will be noise variance from AR model
        :param Q_sigma2: determine prior mode for Q, if None it will be noise variance from AR model
        :param no_priors: override all other prior hyperparameters and proceed with no priors
        :param track_params: save parameters from EM iterations
        :param plot_innov: plot the innovations at each iteration
        :param verbose: print oscillator information during iterative search iterations
        """
        # Store iteration settings as instance attributes
        self.osc_range = osc_range
        self.freq_res = freq_res
        self.keep_param = keep_param

        # Copy the first added oscillator to preserve its parameters
        o1 = self.added_osc[0].copy()

        # Set up iteration variables
        start_add_osc = False
        innov_limit = None
        counter = 0

        # Begin iterations to search for more oscillators in the observed data
        while o1.ncomp < osc_range:
            if not start_add_osc:  # do EM learning on the initial oscillator first, don't add oscillators yet
                b = o1.R[0, 0]
                start_add_osc = True
                if verbose:
                    print('EM learning on the initial oscillator')

            else:  # add an oscillator
                # find existing frequency
                if abs(o1.freq[-1] - self.added_osc[-1].freq) >= self.freq_res:
                    existing_freqs = list(o1.freq) + [x.freq[0] for x in self.added_osc[:-1]]
                else:
                    existing_freqs = list(o1.freq) + [x.freq[0] for x in self.added_osc]

                # noinspection PyUnboundLocalVariable
                add_freq, add_radius, _, _, a, b = initialize_newosc(
                    o1.Fs, innovations, existing_freqs=existing_freqs, freq_res=self.freq_res,
                    ar_order=self.osc_range * 2 - 1, burg_flag=self.burg_flag)

                if verbose:
                    print('AR: add freq = %0.2f Hz, add radius = %0.3f, add sigma2 = %0.3f' % (add_freq, add_radius, b))

                # plot the fitted oscillator from previous iteration and the innovation
                if plot_innov:
                    innov_limit, _ = innovations_plot(o1, o1.y, innovations, a, b, add_freq, add_radius,
                                                      plot_all_poles=True, ax_limit=innov_limit, counter=counter)

                # construct an additional oscillator
                o2 = Osc(a=add_radius, freq=add_freq, sigma2=b, Fs=o1.Fs)
                self.added_osc.append(o2)
                o1.append(o2)  # add the additional oscillator to the existing oscillator instance
                counter += 1  # accumulate iteration counter

            # Initialize priors for EM learning
            if no_priors:
                priors = None
            else:
                prior_params = {}
                if R_hp is not None and R_sigma2 != 'MLE' and 'R' not in keep_param:
                    prior_params.update({'R_hyperparameter': R_hp})
                if Q_hp is not None and Q_sigma2 != 'MLE' and 'Q' not in keep_param:
                    prior_params.update({'Q_hyperparameter': Q_hp})
                if R_sigma2 is not None and 'R' not in keep_param:
                    prior_params.update({'R_sigma2': b if R_sigma2 == 'b' else R_sigma2})
                if Q_sigma2 is not None and 'Q' not in keep_param:
                    # TODO: add the case to handle multiple Q_sigma2
                    prior_params.update({'Q_sigma2': b if Q_sigma2 == 'b' else Q_sigma2})
                if verbose:
                    print('Prior input parameters:', prior_params)
                priors = o1.initialize_priors(kappa=o1.ntime * freq_hp, **prior_params)
            self.priors.append(priors)

            # Run EM iterations
            em_params = []  # store the oscillator parameters throughout EM iterations
            for x in range(50):
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
            innovations, ll = find_innovations(o1)
            self.ll.append(ll)
            self.AIC.append(aic_calc(o1, ll))  # compute AIC

        # Find the knee_index for the selected model of fitted oscillators
        self.knee_index = get_knee(self.ll)

        # Perform reiterations
        if reiterate:
            self.reiterate(track_params=track_params, plot_innov=plot_innov, verbose=verbose)

    def reiterate(self, track_params=False, plot_innov=False, verbose=False):
        """ Sort and re-add the oscillators in descending order of log-likelihood increase """
        # Fast-forward to after fitting the first oscillator
        o1 = self.fitted_osc[0].copy()
        o1.y = self.added_osc[0].y.copy()  # add the observed data back in

        # Fill in the instance attributes for the first fitted oscillator
        added_osc_re = self.added_osc[:1]
        fitted_osc_re = self.fitted_osc[:1]
        priors_re = self.priors[:1]
        all_params_re = [self.all_params[0]] if track_params else self.all_params[:1]
        innovations, ll = find_innovations(o1)
        ll_re = [ll]
        AIC_re = [aic_calc(o1, ll)]

        # Set up iteration variables
        add_indices = np.diff(self.ll).argsort()[::-1] + 1
        innov_limit = None
        counter = 0

        # Reiterate through all added oscillators
        for add_idx in add_indices:
            # look up the oscillator to be added next
            o2 = self.added_osc[add_idx]

            if verbose:
                print('Re-iteration AR: add freq = %0.2f Hz, add radius = %0.3f, add sigma2 = %0.3f'
                      % (o2.freq[0], o2.a[0], o2.sigma2[0]))

            # plot the fitted oscillator from previous iteration and the innovation
            if plot_innov:
                a, b, _ = fit_ar(np.squeeze(innovations), self.osc_range * 2 - 1, self.burg_flag)
                innov_limit, _ = innovations_plot(o1, o1.y, innovations, a, b, o2.freq[0], o2.a[0],
                                                  plot_all_poles=True, ax_limit=innov_limit, counter=counter)

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
            innovations, ll = find_innovations(o1)
            ll_re.append(ll)
            AIC_re.append(aic_calc(o1, ll))  # compute AIC

        # Overwrite the instance attributes from iterate()
        self.added_osc = added_osc_re
        self.fitted_osc = fitted_osc_re
        self.priors = priors_re
        self.all_params = all_params_re
        self.ll_xre = self.ll  # save the log likelihoods from before reiterate()
        self.ll = ll_re  # overwrite the log likelihoods after reiterate()
        self.AIC = AIC_re

        # Find the new knee_index for the selected model of re-fitted oscillators
        self.knee_index = get_knee(self.ll)  # knee replacement

        # Update the reiterated flag
        self.reiterated = True

    def get_aperiodic(self, plot_innov=False):
        """ Find the aperiodic component defined as prediction residual from the best model """
        y = self.added_osc[0].y
        osc = self.fitted_osc[self.knee_index]
        innovations, _ = find_innovations(osc, y=y)

        # Obtain the theoretical spectrum of a fitted AR model
        a, b, _ = fit_ar(np.squeeze(innovations), self.osc_range * 2 - 1, self.burg_flag)
        ar_psd, ar_hz = get_ar_psd(osc.Fs, a, b)

        # Fit a straight line in semi-log space, i.e., log(PSD) against linear frequency (Hz)
        from scipy import stats
        slope, intercept, *_ = stats.linregress(ar_hz, 10 * np.log10(ar_psd))

        if plot_innov:
            _, fig = innovations_plot(osc, y, innovations, a, b, counter=self.knee_index)
            plot_fit_line(fig, slope, intercept)

        return innovations, (slope, intercept)

    def __repr__(self):
        """ Unambiguous and concise representation when calling IterativeOscillatorModel() """
        return 'IterOsc(' + str(len(self)) + ')<' + hex(id(self))[-4:] + '>'

    def __str__(self):
        """ Helpful information when calling print(IterativeOscillatorModel()) """
        print_str = "number of oscillators = %d\n " % len(self)
        if self.fitted_osc:
            print_str += str(self.fitted_osc[-1])
        else:
            print_str += 'There is no fitted oscillator. Initial oscillator:\n '
            print_str += str(self.added_osc[0])
        return print_str

    def __len__(self):
        return self.added_osc[0].ncomp + len(self.added_osc) - 1
