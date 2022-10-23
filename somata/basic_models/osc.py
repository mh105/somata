"""
Author: Mingjian He <mh105@mit.edu>
        Amanda Beck <ambeck@mit.edu>

osc module contains Matsuda oscillator specific methods used in SOMATA
"""

from somata.basic_models import StateSpaceModel as Ssm
import numpy as np
import numbers
from sorcery import dict_of
from math import atan2, sin, cos, sqrt
from scipy.linalg import block_diag


class OscillatorModel(Ssm):
    """
    OscillatorModel is a subclass of StateSpaceModel class
    dedicated to Matsuda oscillators
    """
    type = 'osc'
    default_G = np.array([[1, 0]], dtype=np.float64)  # (Real, Imaginary)
    default_a = 0.99
    default_sigma2 = 3.
    a = None
    freq = None
    w = None
    sigma2 = None
    dc_idx = None

    # temporary attributes related to whitener, organize later
    D = None
    gamma = None
    K = None
    y_whiten = None

    def __init__(self, a=None, freq=None, w=None, sigma2=None, add_dc=False,
                 components='Osc', F=None, Q=None, mu0=None, Q0=None, G=None, R=None, y=None, Fs=None):
        """
        Constructor method for OscillatorModel class
        :param a: damping factor of oscillators
        :param freq: rotation frequency of oscillators in hertz
        :param w: rotation frequency of oscillators in radian
        :param sigma2: noise variance of oscillators
        :param add_dc: flag to add a DC oscillator
        :param components: a list of independent oscillator components
        :param F: transition matrix
        :param Q: state noise covariance matrix
        :param mu0: initial state mean vector
        :param Q0: initial state covariance matrix
        :param G: observation matrix (row major)
        :param R: observation noise covariance matrix
        :param y: observed data (row major, can be multivariate)
        :param Fs: sampling frequency in Hz
        """
        # Verify that components are of valid type
        assert components == 'Osc' or \
               all([x.type == 'osc' for x in components]), 'Encountered non-osc type component.'  # type: ignore

        # Oscillator models can be constructed by directly specifying
        # state equation oscillator parameters {a, w, sigma2}
        if freq is not None:  # one can provide frequency in Hz as well
            self.freq = np.asanyarray([freq], dtype=np.float64) if \
                isinstance(freq, numbers.Number) else np.asanyarray(freq, dtype=np.float64)
            assert Fs is not None, 'Input frequency in Hz but no sampling frequency provided.'
            w_temp = [OscillatorModel.hz_to_rad(x, Fs) for x in self.freq] if \
                self.freq.size > 1 else OscillatorModel.hz_to_rad(self.freq, Fs)
            if w is not None:
                assert (w_temp == w).all(), 'Input frequency parameters do not agree with each other.'
            w = w_temp

        if w is not None:
            w = np.asanyarray([w], dtype=np.float64) if \
                isinstance(w, numbers.Number) else np.asanyarray(w, dtype=np.float64)
            if a is not None:
                a = np.asanyarray([a], dtype=np.float64) if \
                    isinstance(a, numbers.Number) else np.asanyarray(a, dtype=np.float64)
                assert a.size == w.size, 'Different numbers of oscillator parameters provided.'
            else:
                a = np.ones_like(w, dtype=np.float64) * OscillatorModel.default_a
            if sigma2 is not None:
                sigma2 = np.asanyarray([sigma2], dtype=np.float64) if \
                    isinstance(sigma2, numbers.Number) else np.asanyarray(sigma2, dtype=np.float64)
                assert sigma2.size == w.size, 'Different numbers of oscillator parameters provided.'
            else:
                sigma2 = np.ones_like(w, dtype=np.float64) * OscillatorModel.default_sigma2

            self.a = a
            self.w = w
            self.sigma2 = sigma2

            F_tmp, Q_tmp = self._osc_to_ssm_param()
            if F is not None:
                assert (F == F_tmp).all(), 'Input state equation parameters do not agree with input F.'  # type: ignore
            if Q is not None:
                assert (Q == Q_tmp).all(), 'Input state equation parameters do not agree with input Q.'  # type: ignore
            F, Q = F_tmp, Q_tmp
        else:
            assert a is None, 'No frequency provided but damping factor a is given as input.'
            assert sigma2 is None, 'No frequency provided but noise variance sigma2 is given as input.'

        # Provide default values for mu0 and Q0
        mu0 = np.zeros((F.shape[1], 1), dtype=np.float64) if mu0 is None and F is not None else mu0
        Q0 = Q if Q0 is None and Q is not None else Q0

        # Call parent class constructor
        super().__init__(components=components, F=F, Q=Q, mu0=mu0, Q0=Q0, G=G, R=R, y=y, Fs=Fs)

        # Fill oscillator parameters
        self.fill_osc_param()

        # Add DC oscillator if specified
        if add_dc:
            if self.w is None:  # shortcut to create a DC oscillator using OscillatorModel(add_dc=True)
                o_dc = OscillatorModel(a=a, w=0., sigma2=sigma2, add_dc=False,
                                       F=F, Q=Q, mu0=mu0, Q0=Q0, R=R, y=y, Fs=Fs)
                attr_dict = o_dc.__dict__
                for attr in attr_dict.keys():
                    setattr(self, attr, getattr(o_dc, attr))
                self.dc_idx = 0
            else:
                self.add_dc()
        else:
            self.dc_idx = None

    # Dunder methods - magic methods
    def __repr__(self):
        return super().__repr__().replace('Ssm', 'Osc')

    def __str__(self):
        print_str = super().__str__().replace('<Ssm object at', '<Osc object at')
        # Append additional information about oscillator parameters
        np.set_printoptions(precision=3)
        print_str += "{0:9} = {1}\n ".format("damping a", str(self.a))
        if self.Fs is None:
            print_str += "{0:9} = {1}\n ".format("freq w", str(self.w))
        else:
            np.set_printoptions(precision=2)
            print_str += "{0:9} = {1}\n ".format("freq Hz", str(self.freq))
            np.set_printoptions(precision=3)
        print_str += "{0:9} = {1}\n ".format("sigma2", str(self.sigma2))
        print_str += "{0:9} = {1}\n ".format("obs noise R", str(self.R))
        print_str += "{0:9} = {1}\n".format("dc index", str(self.dc_idx))
        np.set_printoptions(precision=8)
        return print_str

    # Syntactic sugar methods - useful methods to make manipulations easier
    def concat_(self, other, skip_components=False):
        """
        Join two OscillatorModel objects together by concatenating the
        components.
        """
        assert self.type == 'osc' and other.type == 'osc', \
            'Both objects input to concat_() need to be of OscillatorModel class.'

        # Fill oscillator parameters in both objects first
        self.fill_osc_param()
        other.fill_osc_param()

        # Handle DC oscillators if present
        save_other_dc = None
        save_self_dc_index = None
        if other.dc_idx is not None:
            if self.dc_idx is None:
                save_other_dc = other.components[other.dc_idx]
            else:
                save_self_dc_index = self.dc_idx
            # Confirm that the DC oscillator has default values therefore can be added back with constructor
            assert other.a[other.dc_idx] == other.default_a, 'DC oscillator damping factor will be lost by concat_().'
            assert other.w[other.dc_idx] == 0, 'DC oscillator frequency will be lost by concat_().'
            assert other.sigma2[other.dc_idx] == other.default_sigma2, 'DC oscillator sigma2 will be lost by concat_().'
            # Temporarily remove the DC oscillator from the other object
            tmp_other = other.copy()
            tmp_other.remove_component(tmp_other.dc_idx)
            tmp_other.components = other.components[1:] if tmp_other.ncomp > 0 else tmp_other.components
            other = tmp_other

        # Frequency is a mandatory input for objects to call concat_()
        assert self.w is not None and other.w is not None, 'Both objects need at least w specified in order to concat.'
        w = np.hstack([self.w, other.w])

        # Configure the rest of attributes that are immutable
        # a
        if self.a is None and other.a is None:
            a = None
        elif self.a is None:
            a = np.hstack([np.ones_like(self.w, dtype=np.float64) * OscillatorModel.default_a, other.a])
        elif other.a is None:
            a = np.hstack([self.a, np.ones_like(other.w, dtype=np.float64) * OscillatorModel.default_a])
        else:
            a = np.hstack([self.a, other.a])

        # sigma2
        if self.sigma2 is None and other.sigma2 is None:
            sigma2 = None
        elif self.sigma2 is None:
            sigma2 = np.hstack([np.ones_like(self.w, dtype=np.float64) * OscillatorModel.default_sigma2, other.sigma2])
        elif other.sigma2 is None:
            sigma2 = np.hstack([self.sigma2, np.ones_like(other.w, dtype=np.float64) * OscillatorModel.default_sigma2])
        else:
            sigma2 = np.hstack([self.sigma2, other.sigma2])

        # Q0
        if self.Q0 is None and other.Q0 is None:
            Q0 = None
        elif self.Q0 is None:
            Q0 = block_diag(self.Q, other.Q0) if self.Q is not None else \
                block_diag(np.eye(self.w.size * 2, dtype=np.float64) * OscillatorModel.default_sigma2, other.Q0)
        elif other.mu0 is None:
            Q0 = block_diag(self.Q0, other.Q) if other.Q is not None else \
                block_diag(self.Q0, np.eye(other.w.size * 2, dtype=np.float64) * OscillatorModel.default_sigma2)
        else:
            Q0 = block_diag(self.Q0, other.Q0)

        # Call parent class method to obtain general concatenated attributes
        temp_obj = super().concat_(other, skip_components=skip_components)

        add_dc = save_other_dc is not None
        new_obj = OscillatorModel(a=a, w=w, sigma2=sigma2, mu0=temp_obj.mu0, Q0=Q0,
                                  R=temp_obj.R, y=temp_obj.y, Fs=temp_obj.Fs,
                                  components=temp_obj.components, add_dc=add_dc)
        new_obj.components[0] = save_other_dc if add_dc else new_obj.components[0]
        new_obj.dc_idx = save_self_dc_index if save_self_dc_index is not None else new_obj.dc_idx

        return new_obj

    def remove_component(self, comp_idx=0):
        """
        Remove a component from the OscillatorModel object,
        default to remove the left most component
        """
        super().remove_component(comp_idx=comp_idx)
        if self.dc_idx is not None:
            if comp_idx == self.dc_idx:
                self.dc_idx = None
            elif comp_idx < self.dc_idx:
                self.dc_idx -= 1
        if self.a is not None:
            self.a = np.delete(self.a, comp_idx)
        if self.w is not None:
            self.w = np.delete(self.w, comp_idx)
        if self.freq is not None:
            self.freq = np.delete(self.freq, comp_idx)
        if self.sigma2 is not None:
            self.sigma2 = np.delete(self.sigma2, comp_idx)

        # Set attributes to default values if nothing left
        self.a = None if len(self.a) == 0 else self.a
        self.w = None if len(self.w) == 0 else self.w
        self.freq = None if len(self.freq) == 0 else self.freq
        self.sigma2 = None if len(self.sigma2) == 0 else self.sigma2

    def add_dc(self):
        """ Add a DC oscillator to an existing OscillatorModel object """
        assert self.dc_idx is None, 'A DC oscillator already exists.'
        # usually DC oscillator has large variance therefore using max(self.sigma2)
        sigma2 = OscillatorModel.default_sigma2 if self.sigma2 is None else max(self.sigma2)
        self.rappend(OscillatorModel(w=0., sigma2=sigma2, add_dc=False))
        self.dc_idx = 0  # update the index for DC oscillator

    def fill_osc_param(self, F=None, Q=None, Fs=None):
        """ Attempt to fill oscillator parameters """
        F = self.F if F is None else F
        Q = self.Q if Q is None else Q
        Fs = self.Fs if Fs is None else Fs
        a, w, sigma2 = self._ssm_to_osc_param(F=F, Q=Q)
        self.a = a if F is not None else self.a
        self.w = w if F is not None else self.w
        self.sigma2 = sigma2 if Q is not None else self.sigma2
        if self.w is not None and Fs is not None:
            freq_hz = [OscillatorModel.rad_to_hz(x, Fs) for x in self.w] if self.w.size > 1 else \
                OscillatorModel.rad_to_hz(self.w, Fs)
            self.freq = np.asanyarray(freq_hz, dtype=np.float64)

    def _ssm_to_osc_param(self, F=None, Q=None):
        """ Convert from matrices F and Q to oscillator parameters """
        F = self.F if F is None else F
        Q = self.Q if Q is None else Q
        ncomp = self.ncomp
        a = np.zeros(ncomp, dtype=np.float64) if ncomp > 0 else None
        w = np.zeros(ncomp, dtype=np.float64) if ncomp > 0 else None
        sigma2 = np.zeros(ncomp, dtype=np.float64) if ncomp > 0 else None
        for ii in range(ncomp):
            a[ii], w[ii] = OscillatorModel.get_rot_param(
                F[ii * 2:ii * 2 + 2, ii * 2:ii * 2 + 2]) if F is not None else (None, None)
            sigma2[ii] = Q[ii * 2, ii * 2] if Q is not None else None  # type: ignore
        return a, w, sigma2

    def _osc_to_ssm_param(self, a=None, w=None, sigma2=None):
        """ Convert from oscillator parameters to matrices F and Q """
        # a, w, sigma2 must be index-able
        a = self.a if a is None else a
        w = self.w if w is None else w
        sigma2 = self.sigma2 if sigma2 is None else sigma2
        if w is not None:
            F_blocks = []
            Q_blocks = []
            for ii in range(w.size):
                F_blocks.append(OscillatorModel.get_rot_mat(a[ii], w[ii]))
                Q_blocks.append(np.eye(2, dtype=np.float64) * sigma2[ii])
            return block_diag(*F_blocks), block_diag(*Q_blocks)
        else:
            return None, None

    def get_default_q(self, components=None, E=None):
        """
        Get the default structure of state noise covariance
        matrix Q in the Q_basis block diagonal form
        """
        components = self.components if components is None else components
        if len(components) == 1 or type(components) is OscillatorModel:
            E = np.eye(1, dtype=np.float64) if E is None else E
            default_Q = block_diag(E, E)
        else:
            default_Q = block_diag(*[x.get_default_q(components=x, E=E) for x in components])
        return default_Q

    @staticmethod
    def tr(B):
        """ Compute the trace in a 2x2 matrix """
        assert B.shape == (2, 2), 'B is not a 2x2 matrix.'
        return B[0, 0] + B[1, 1]

    @staticmethod
    def rt(B):
        """ Compute the off-diagonal entry difference in a 2x2 matrix """
        assert B.shape == (2, 2), 'B is not a 2x2 matrix.'
        return B[1, 0] - B[0, 1]

    @staticmethod
    def get_rot_mat(a, w):
        """ F = a * R(w) where R is a rotation matrix with rotation radian w """
        F = np.zeros((2, 2), dtype=np.float64)
        F[0, 0] = a * cos(w)
        F[0, 1] = a * -sin(w)
        F[1, 0] = a * sin(w)
        F[1, 1] = a * cos(w)
        return F

    @staticmethod
    def get_rot_param(F):
        """ F = a * R(w) where R is a rotation matrix with rotation radian w """
        assert F.shape == (2, 2), 'F is not a 2x2 matrix.'
        a = sqrt(F[0, 0] ** 2 + F[1, 0] ** 2)
        w = atan2(F[1, 0], F[0, 0])
        return a, w

    @staticmethod
    def hz_to_rad(frequency_hz, sampling_frequency):
        """ Convert rotation frequency from hertz to radian """
        return 2 * np.pi * frequency_hz / sampling_frequency

    @staticmethod
    def rad_to_hz(frequency_rad, sampling_frequency):
        """ Convert rotation frequency from radian to hertz """
        return frequency_rad * sampling_frequency / (2 * np.pi)

    # Parameter estimation methods (M step)
    def m_estimate(self, priors=None, **kwargs):
        """
        Maximum likelihood or Maximum a posteriori estimation to update
        parameters. Calling super().m_estimate() for all Ssm parameters
        """
        # Handle DC oscillators if present
        if self.dc_idx is not None:
            if priors is None:
                priors: list = [None] * self.ncomp
                priors[self.dc_idx] = {'dc': True}
            elif type(priors) == dict:
                priors['dc'] = True
                priors = [priors]
            else:
                priors[self.dc_idx]['dc'] = True

        results = super().m_estimate(priors=priors, **kwargs)
        self.update_comp_param()
        return results

    def update_comp_param(self):
        """ Update OscillatorModel specific parameters """
        self.fill_osc_param()

    def initialize_priors(self, kappa=None, f_prior=None, Q_sigma2=None, Q_hyperparameter=None,
                          R_sigma2=None, R_hyperparameter=None):
        """
        Initialize priors for Matsuda oscillators

        - INPUT RULE: if None is given or no explicit argument is provided, default values
        will be initialized and used as prior parameters. To initialize priors only for
        some parameters but not others, use the input value 'MLE' on desired parameters.
        """
        # base case
        if self.ncomp <= 1:
            # [Von Mises prior] on rotation frequency w
            if f_prior is None:
                _, w_prior = OscillatorModel.get_rot_param(self.F)
                f_prior = OscillatorModel.rad_to_hz(w_prior, self.Fs)
            else:
                w_prior = OscillatorModel.hz_to_rad(f_prior, self.Fs)
            kappa = 10000. if kappa is None else kappa
            vmp_param = {'kappa': kappa, 'f_prior': f_prior, 'w_prior': w_prior}

            # [Inverse gamma prior] on state noise covariance Q diagonal entries
            if Q_sigma2 is None:
                assert self.Q[0, 0] == self.Q[1, 1], 'State noise sigma2 differs between real and imaginary parts'
                Q_sigma2 = self.Q[0, 0]
                Q_hyperparameter = 0.1 if Q_hyperparameter is None else Q_hyperparameter

            # [Inverse gamma prior] on observation noise variance R <--- TODO: update to Wishart
            if R_sigma2 is None:
                if self.R.shape[0] > 1:
                    raise NotImplementedError('Only uni-variate observation data is supported with prior for now.')
                else:
                    R_sigma2 = self.R[0, 0]
                    R_hyperparameter = 0.1 if R_hyperparameter is None else R_hyperparameter

            return dict_of(vmp_param, Q_sigma2, Q_hyperparameter, R_sigma2, R_hyperparameter)

        # recursive case
        else:
            assert self.components is not None, 'Cannot initialize priors outside base case when components is None.'
            components_prefill = self.fill_components(empty_comp=OscillatorModel(), deep_copy=True)

            # expand the specified prior values to the length of components
            f_prior = self._initialize_priors_recursive_list(f_prior)
            kappa = self._initialize_priors_recursive_list(kappa)
            Q_sigma2 = self._initialize_priors_recursive_list(Q_sigma2)
            R_sigma2 = self._initialize_priors_recursive_list(R_sigma2)
            R_hyperparameter = self._initialize_priors_recursive_list(R_hyperparameter)
            Q_hyperparameter = self._initialize_priors_recursive_list(Q_hyperparameter)

            # construct the final priors that is a list of dictionaries
            priors = []
            for ii in range(self.ncomp):
                current_component: OscillatorModel = self.components[ii]
                assert current_component.type == 'osc', 'Component type is not OscillatorModel class.'

                priors.append(current_component.initialize_priors(kappa=kappa[ii], f_prior=f_prior[ii],
                                                                  Q_sigma2=Q_sigma2[ii], R_sigma2=R_sigma2[ii],
                                                                  R_hyperparameter=R_hyperparameter[ii],
                                                                  Q_hyperparameter=Q_hyperparameter[ii]))

            # unfill the components
            self.unfill_components(components_prefill)

            return priors

    @staticmethod
    def ig_lp(mode, alpha, estimate):
        """ Compute inverse gamma log prior based on parameters (ignoring terms that don't depend on estimate) """
        return -(alpha + 1) * np.log(estimate) - mode * (alpha + 1) / estimate

    @staticmethod
    def vmp_lp(vmp_dict, estimate):
        """ Compute von mises prior based on parameters (ignoring terms that don't depend on estimate) """
        return vmp_dict['kappa'] * np.cos(estimate - vmp_dict['w_prior'])

    @staticmethod
    def _m_update_f(A=None, B=None, C=None, priors=None):
        """ Update transition matrix -- F """
        # Update the rotation radian -- w (adaptive Von Mises prior)
        if Ssm._m_update_if_mle(['dc', 'vmp_param'], priors):
            # MLE
            w_new = atan2(OscillatorModel.rt(B),
                          OscillatorModel.tr(B))
        elif 'dc' in priors:
            # fix DC oscillator frequency at 0
            w_new = 0.
        elif 'vmp_param' in priors:
            # MAP with Von Mises prior
            kappa = priors['vmp_param']['kappa']  # effective kappa is kappa * a_new/sigma2_Q_new
            w_prior = priors['vmp_param']['w_prior']
            w_new = atan2(OscillatorModel.rt(B) + kappa * sin(w_prior),
                          OscillatorModel.tr(B) + kappa * cos(w_prior))
        else:
            raise RuntimeError('Could not set m_update rule for rotation radian w. Please check.')

        # Update the rotation damping factor -- a (no prior)
        a_new = (cos(w_new) * OscillatorModel.tr(B) + sin(w_new) * OscillatorModel.rt(B)) / OscillatorModel.tr(C)

        # Construct transition matrix -- F
        F = OscillatorModel.get_rot_mat(a_new, w_new)
        return F

    @staticmethod
    def _m_update_q(A=None, B=None, C=None, T=None, F=None, priors=None):
        """ Update state noise covariance matrix -- Q """
        a_new, w_new = OscillatorModel.get_rot_param(F)
        Q_ss = (OscillatorModel.tr(C) - 2 * a_new * (cos(w_new) * OscillatorModel.tr(B) +
                                                     sin(w_new) * OscillatorModel.rt(B))
                + a_new ** 2 * OscillatorModel.tr(A))

        T *= 2  # double the data length since we sum over real and imaginary noise processes

        if Ssm._m_update_if_mle('Q_sigma2', priors):
            # MLE
            sigma2_Q_new = Q_ss / T
        else:
            # MAP with inverse gamma prior
            Q_init = priors['Q_sigma2']
            Q_hp = priors['Q_hyperparameter'] if 'Q_hyperparameter' in priors else 0.1
            alpha = T * Q_hp / 2  # scales with data length T according to the hyperparameter
            beta = Q_init * (alpha + 1)  # setting the mode of inverse gamma prior to be Q_init
            sigma2_Q_new = (beta + Q_ss / 2) / (alpha + T / 2 + 1)  # mode of inverse gamma posterior

        Q = sigma2_Q_new * np.eye(2, dtype=np.float64)
        return Q

    @staticmethod
    def _m_update_q0(x_0_n=None, P_0_n=None, mu0=None):
        """ Update initial state covariance -- Q0 """
        sigma2_Q0 = OscillatorModel.tr(P_0_n + x_0_n[:, None] @ x_0_n[:, None].T
                                       - x_0_n[:, None] @ mu0.T - mu0 @ x_0_n[:, None].T + mu0 @ mu0.T) / 2
        Q0 = sigma2_Q0 * np.eye(2, dtype=np.float64)
        return Q0

    @staticmethod
    def _m_update_g(y=None, x_t_n=None, P_t_n=None, h_t=None):
        """ Update observation matrix -- G (OscillatorModel has fixed G) """
        return None
