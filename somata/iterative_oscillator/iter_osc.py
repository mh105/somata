""" Author: Amanda Beck <ambeck@mit.edu> """

from .helper_functions import *
from ..basic_models import OscillatorModel as Osc


class IterativeOscillatorModel(object):
    """
    IterativeOscillatorModel is an object class containing fitted OscillatorModel objects
    from the iterative oscillator algorithm
    """
    def __init__(self, y, fs, noise_start=None, ar_order=13, burg_flag=False, verbose=False):
        """
        Inputs:
            :param y: observed data
            :param fs: sampling frequency (Hz)
            :param noise_start: frequency (Hz) above which there should be only white noise, no oscillation
            :param ar_order: Order of autoregressive model used to model innovations / one-step prediction error
            :param burg_flag: True will use Burg algorithm to fit AR parameters instead of the default Yule-Walker
            :param verbose: print initial parameters
        """
        if noise_start is None:
            noise_start = fs / 2 - 20
            if noise_start <= 0:
                noise_start = fs / 2 - 5
        assert noise_start > 0, 'Please redefine noise_start. Entered or calculated frequency is less than zero.'

        # initialize the first oscillator
        init_params, scale_factor = initial_param(y, fs, noise_start, ar_order, burg_flag)
        if verbose:
            print('Noise starting frequency is %0.1f Hz' % noise_start)
            print('Initial oscillator parameters:')
            print(init_params)

        # populate instance attributes
        # noinspection PyProtectedMember
        self.y_original = Osc._must_be_row(Osc._process_constructor_input(y))  # make sure y is a 2D row vector
        self.scale = scale_factor
        self.ar_order = ar_order
        self.burg_flag = burg_flag
        self.added_osc = [Osc(**init_params)]
        self.fitted_osc = []
        self.scaled_osc = []
        self.AIC = []
        self.ll = []
        self.log_prior = []
        self.all_params = []  # a list of lists containing Osc objects with only parameters (y dropped)
        self.priors = []
        self.knee_index = []
        self.Q_innov = []

    def iterate(self, osc_range=7, freq_hp=3, R_hp=0.1, Q_hp=None, keep_param=(), R_sigma2='b', Q_sigma2='MLE',
                no_priors=False, track_params=False, plot_innov=False, verbose=False):
        """
        Apply Iterative Oscillator Algorithm

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
        :param freq_hp: concentration hyperparameter in Von Mises Prior (freq_hp * data length)
        :param R_hp: hyperparameter in Inverse Gamma Prior determining weight on prior mode compared to MLE
        :param Q_hp: hyperparameter in Inverse Gamma Prior determining weight on prior mode compared to MLE
        :param keep_param: a tuple of strings for parameters to keep and not update
        :param R_sigma2: determine prior mode for R, if None it will be noise variance from AR model
        :param Q_sigma2: determine prior mode for Q, if None it will be noise variance from AR model
        :param no_priors: override all other prior hyperparameters and proceed with no priors
        :param track_params: save parameters from EM iterations
        :param plot_innov: plot the innovations at each iteration
        :param verbose: print oscillator information during iterative search iterations
        """
        # Copy the last oscillator so that updated parameters during iterations do not override it
        o1 = self.added_osc[-1].copy()

        # Set up iteration variables
        start_add_osc = False
        innov_limit = None
        self.Q_innov = [o1.R.copy()[0, 0]]

        # Begin iterations to search for more oscillators in the observed data
        while o1.ncomp < osc_range:
            if not start_add_osc:  # do EM learning on the initial oscillator first, don't add oscillators yet
                b = o1.R.copy()[0, 0]
                start_add_osc = True
                if verbose:
                    print('EM learning on the initial oscillator')

            else:  # add an oscillator
                # noinspection PyUnboundLocalVariable
                add_freq, add_radius, all_freq, all_radii, a, b = initialize_newosc(o1.Fs, innovations,
                                                                                    existing_freqs=None,
                                                                                    ar_order=self.ar_order,
                                                                                    burg_flag=self.burg_flag)
                self.Q_innov.append(b)

                if verbose:
                    print('AR: add freq: %0.2f Hz, add radius: %0.3f, add q: %0.3f' % (add_freq, add_radius, b))

                # plot the fitted oscillator from previous iteration and the innovation
                if plot_innov:
                    innov_limit, _ = innovations_plot(o1, o1.y, innovations, a, b, add_freq, ax_limit=innov_limit)

                # construct an additional oscillator
                o2 = Osc(a=add_radius, freq=add_freq, sigma2=b, Fs=o1.Fs)
                self.added_osc.append(o2.copy(drop_y=True))
                o1.append(o2)  # add the additional oscillator to the existing oscillator instance

            # Initialize priors for EM learning
            input_params = {}
            if R_hp is not None:
                input_params.update({'R_hyperparameter': R_hp})
            if Q_hp is not None:
                input_params.update({'Q_hyperparameter': Q_hp})
            if R_sigma2 is not None:
                input_params.update({'R_sigma2': b if R_sigma2 == 'b' else R_sigma2})
            if Q_sigma2 is not None:
                # TODO: add the case to handle multiple Q_sigma2
                input_params.update({'Q_sigma2': b if Q_sigma2 == 'b' else Q_sigma2})
            if verbose:
                print('Prior input parameters:')
                print(input_params)

            priors = None if no_priors else o1.initialize_priors(kappa=o1.ntime * freq_hp, **input_params)
            self.priors.append(priors)

            # Run EM iterations
            em_params = []  # store the oscillator parameters throughout EM iterations
            for x in range(50):
                _ = o1.m_estimate(**o1.dejong_filt_smooth(EM=True), priors=priors, keep_param=keep_param)
                if track_params:
                    em_params.append(o1.copy(drop_y=True))
            if track_params:
                self.all_params.append(em_params)

            self.fitted_osc.append(o1.copy(drop_y=True))

            if verbose:
                print('Oscillator %d completed' % o1.ncomp)
                print(o1)

            # if plot_innov:
            #     _ = o1.visualize_freq('theoretical')

            # Find the innovation spectrum
            innovations, ll = find_innovations(o1)
            self.ll.append(ll)
            self.AIC.append(aic_calc(o1, ll))  # compute AIC

            # Store the parameters after scaling variance back to original y
            scaled_o1 = o1.copy(drop_y=True)
            scaled_o1.R *= self.scale
            scaled_o1.sigma2 *= self.scale
            scaled_o1.Q *= self.scale
            scaled_o1.Q0 *= self.scale
            self.scaled_osc.append(scaled_o1)

        # Find the knee_index for the selected model of fitted oscillators
        self.knee_index = get_knee(self.ll)

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
