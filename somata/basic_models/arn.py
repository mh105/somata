"""
Author: Mingjian He <mh105@mit.edu>

arn module contains autoregressive model of order n methods used in SOMATA
"""

from somata.basic_models import StateSpaceModel as Ssm
from somata.exact_inference import inverse
import numpy as np
import numbers
from sorcery import dict_of
from scipy.linalg import block_diag


class AutoRegModel(Ssm):
    """
    AutoRegModel is a subclass of StateSpaceModel class dedicated
    to autoregressive models of order n
    """
    type = 'arn'
    default_G = None
    default_sigma2 = 3.
    order = None
    coeff = None
    sigma2 = None

    def __init__(self, coeff=None, sigma2=None,
                 components='Arn', F=None, Q=None, mu0=None, Q0=None, G=None, R=None, y=None, Fs=None):
        """
        Constructor method for AutoRegModel class
        :param coeff: coefficients of AR models
        :param sigma2: noise variance of AR models
        :param components: a list of independent AR components
        :param F: transition matrix
        :param Q: state noise covariance matrix
        :param mu0: initial state mean vector
        :param Q0: initial state covariance matrix
        :param G: observation matrix (row major)
        :param R: observation noise covariance matrix
        :param y: observed data (row major, can be multivariate)
        :param Fs: sampling frequency in Hz
        """
        # Autoregressive models can be constructed by directly specifying
        # the autoregressive parameters {order, coeff, sigma2}
        if coeff is not None:
            if isinstance(coeff, numbers.Number):
                coeff = (np.asanyarray([coeff], dtype=np.float64), )
            elif isinstance(coeff[0], numbers.Number):
                coeff = (np.asanyarray(coeff, dtype=np.float64),)
            else:
                coeff = tuple([np.asanyarray(x, dtype=np.float64) for x in coeff])

            order = np.asarray([len(x) for x in coeff])

            if sigma2 is not None:
                sigma2 = np.asanyarray([sigma2], dtype=np.float64) if \
                    isinstance(sigma2, numbers.Number) else np.asanyarray(sigma2, dtype=np.float64)
                assert sigma2.size == len(coeff), 'Different numbers of AR model parameters provided.'
            else:
                sigma2 = np.ones_like(order, dtype=np.float64) * AutoRegModel.default_sigma2

            self.order = order
            self.coeff = coeff
            self.sigma2 = sigma2

            F_tmp, Q_tmp = self._arn_to_ssm_param()
            if F is not None:
                assert (F == F_tmp).all(), 'Input state equation parameters do not agree with input F.'  # type: ignore
            if Q is not None:
                assert (Q == Q_tmp).all(), 'Input state equation parameters do not agree with input Q.'  # type: ignore
            F, Q = F_tmp, Q_tmp
        else:
            assert sigma2 is None, 'No coefficient provided but noise variance sigma2 is given as input.'

        # Provide default values for mu0 and Q0
        mu0 = np.zeros((F.shape[1], 1), dtype=np.float64) if mu0 is None and F is not None else mu0
        Q0 = Q if Q0 is None and Q is not None else Q0

        # Fill autoregressive parameters
        self.fill_arn_param(F=F, Q=Q)

        # Set up components input to parent class constructor
        if components == 'Arn':
            if self.order is None:
                components = None
            else:
                components = []
                for ii in range(len(self.order)):
                    component = AutoRegModel()
                    component.default_G = np.hstack([np.array([[1.]], dtype=np.float64),
                                                     np.zeros((1, self.order[ii]-1), dtype=np.float64)])
                    components.append(component)
                self.default_G = components[0].default_G if len(components) == 1 else self.default_G
        else:
            for ii in range(len(components)):
                current_component: AutoRegModel = components[ii]  # type: ignore
                assert current_component.type == 'arn', 'Encountered non-arn type component.'
                assert current_component.default_G.shape[1] == self.order[ii], 'Components mismatch AR order.'
            self.default_G = components[0].default_G if len(components) == 1 else self.default_G  # type: ignore

        # Call parent class constructor
        super().__init__(components=components, F=F, Q=Q, mu0=mu0, Q0=Q0, G=G, R=R, y=y, Fs=Fs)

    # Dunder methods - magic methods
    def __repr__(self):
        """ Dynamic display depending on whether a single AR model """
        if self.default_G is None:
            return super().__repr__().replace('Ssm', 'Arn')
        else:
            return 'Arn=' + str(self.default_G.shape[1]) + '<' + hex(id(self))[-4:] + '>'

    def __str__(self):
        print_str = super().__str__().replace('<Ssm object at', '<Arn object at')
        # Append additional information about autoregressive parameters
        np.set_printoptions(precision=3)
        print_str += "{0:9} = {1}\n ".format("AR order", str(self.order))

        # create the string for displaying coeff
        if self.coeff is None:
            coeff_str = 'None'
        else:
            coeff_str = '(' + str(self.coeff[0])
            for ii in range(1, len(self.coeff)):
                coeff_str += ', ' + str(self.coeff[ii])
            coeff_str += ')'
        print_str += "{0:9} = {1}\n ".format("AR coeff", coeff_str)

        print_str += "{0:9} = {1}\n ".format("sigma2", str(self.sigma2))
        np.set_printoptions(precision=8)
        return print_str

    # Syntactic sugar methods - useful methods to make manipulations easier
    def concat_(self, other, skip_components=False):
        """
        Join two AutoRegModel objects together by concatenating the
        components.
        """
        assert self.type == 'arn' and other.type == 'arn', \
            'Both objects input to concat_() need to be of AutoRegModel class.'

        # Fill autoregressive parameters in both objects first
        self.fill_arn_param()
        other.fill_arn_param()

        # Coefficient is a mandatory input for objects to call concat_()
        assert self.coeff is not None and other.coeff is not None, \
            'Both objects need at least coeff specified in order to concat.'
        tmp_coeff = list(self.coeff)
        for x in other.coeff:
            tmp_coeff.append(x)
        coeff = tuple(tmp_coeff)

        # Configure the rest of attributes that are immutable
        # sigma2
        if self.sigma2 is None and other.sigma2 is None:
            sigma2 = None
        elif self.sigma2 is None:
            sigma2 = np.hstack([np.ones_like(self.coeff, dtype=np.float64) * AutoRegModel.default_sigma2, other.sigma2])
        elif other.sigma2 is None:
            sigma2 = np.hstack([self.sigma2, np.ones_like(other.coeff, dtype=np.float64) * AutoRegModel.default_sigma2])
        else:
            sigma2 = np.hstack([self.sigma2, other.sigma2])

        # Q0
        if self.Q0 is None and other.Q0 is None:
            Q0 = None
        elif self.Q0 is None:
            tmp_sigma2 = np.ones_like(self.order, dtype=np.float64) * AutoRegModel.default_sigma2
            _, tmp_Q0 = self._arn_to_ssm_param(sigma2=tmp_sigma2)
            Q0 = block_diag(tmp_Q0, other.Q0)
        elif other.mu0 is None:
            tmp_sigma2 = np.ones_like(other.order, dtype=np.float64) * AutoRegModel.default_sigma2
            _, tmp_Q0 = AutoRegModel._arn_to_ssm_param(self=other, sigma2=tmp_sigma2)
            Q0 = block_diag(self.Q0, tmp_Q0)
        else:
            Q0 = block_diag(self.Q0, other.Q0)

        # Call parent class method to obtain general concatenated attributes
        temp_obj = super().concat_(other, skip_components=skip_components)

        return AutoRegModel(coeff=coeff, sigma2=sigma2, mu0=temp_obj.mu0, Q0=Q0, R=temp_obj.R,
                            y=temp_obj.y, Fs=temp_obj.Fs, components=temp_obj.components)

    def remove_component(self, comp_idx):
        """
        Remove a component from the AutoRegModel object,
        default to remove the left most component
        """
        super().remove_component(comp_idx=comp_idx)
        if self.order is not None:
            self.order = np.delete(self.order, comp_idx)
        if self.coeff is not None:
            self.coeff = tuple([self.coeff[x] for x in range(len(self.coeff)) if x != comp_idx])
        if self.sigma2 is not None:
            self.sigma2 = np.delete(self.sigma2, comp_idx)

        # Set attributes to default values if nothing left
        self.order = None if len(self.order) == 0 else self.order
        self.coeff = None if len(self.coeff) == 0 else self.coeff
        self.sigma2 = None if len(self.sigma2) == 0 else self.sigma2

    def fill_arn_param(self, F=None, Q=None):
        """ Attempt to fill autoregressive parameters """
        F = self.F if F is None else F
        Q = self.Q if Q is None else Q
        if Q is not None:
            non_zero_diagonal = np.nonzero(np.diagonal(Q))[0]
            self.order = np.asanyarray(
                np.diff(np.append(non_zero_diagonal, Q.shape[0])), dtype=np.int_)  # type: ignore
        elif F is not None:
            self.order = self._guess_ar_order(F=F)
        coeff, sigma2 = self._ssm_to_arn_param(F=F, Q=Q)
        self.coeff = coeff if F is not None else self.coeff
        self.sigma2 = sigma2 if Q is not None else self.sigma2

    def _guess_ar_order(self, F=None):
        """
        Try to guess the orders of AR models using F matrix,
        last resort method, should not be used normally
        """
        F = self.F if F is None else F
        assert F is not None, 'Cannot guess AR orders with None F matrix.'
        pointer = 0
        order = []
        while pointer < F.shape[0]:
            next_pointer = np.argmax(F[pointer, pointer:] == 0)
            current_order = F.shape[0] - pointer if next_pointer == 0 else next_pointer - pointer
            if current_order > 1:
                assert (F[pointer+1:pointer+current_order, pointer:pointer+current_order-1] ==
                        np.eye(current_order-1, dtype=np.float64)).all(), \
                    'Failed to guess the autoregressive model orders. Consider input order explicitly.'
            order.append(current_order)
            pointer += current_order
        return np.array(order, dtype=np.int_)

    def _ssm_to_arn_param(self, F=None, Q=None, order=None):
        """ Convert from matrices F and Q to autoregressive parameters """
        F = self.F if F is None else F
        Q = self.Q if Q is None else Q
        order = self.order if order is None else order
        if order is not None:
            coeff = []
            sigma2 = []
            for ii in range(len(order)):
                start_idx = sum(order[:ii])
                end_idx = sum(order[:ii+1])
                current_coeff = F[start_idx, start_idx:end_idx] if F is not None else None
                if isinstance(current_coeff, numbers.Number):
                    coeff.append(np.asanyarray([current_coeff], dtype=np.float64))
                else:
                    coeff.append(np.asanyarray(current_coeff, dtype=np.float64))  # type: ignore
                current_sigma2 = Q[start_idx, start_idx] if Q is not None else None  # type: ignore
                sigma2.append(current_sigma2)
            coeff = tuple(coeff) if len(order) > 1 else (coeff[0],)
            sigma2 = np.array(sigma2, dtype=np.float64)
            return coeff, sigma2
        else:
            return None, None

    def _arn_to_ssm_param(self, order=None, coeff=None, sigma2=None):
        """ Convert from autoregressive parameters to matrices F and Q """
        order = self.order if order is None else order
        coeff = self.coeff if coeff is None else coeff
        sigma2 = self.sigma2 if sigma2 is None else sigma2
        if coeff is not None:
            F_blocks = []
            Q_blocks = []
            for ii in range(len(coeff)):
                F_blocks.append(AutoRegModel.get_coeff_mat(coeff[ii]))
                Q_block = np.zeros((order[ii], order[ii]), dtype=np.float64)
                Q_block[0, 0] = sigma2[ii]
                Q_blocks.append(Q_block)
            return block_diag(*F_blocks), block_diag(*Q_blocks)
        else:
            return None, None

    def get_default_q(self, components=None, E=None):
        """
        Get the default structure of state noise covariance
        matrix Q in the Q_basis block diagonal form
        """
        components = self.components if components is None else components
        if len(components) == 1 or type(components) is AutoRegModel:
            order = components.default_G.shape[1] if type(components) is AutoRegModel else \
                components[0].default_G.shape[1]
            E = np.eye(1, dtype=np.float64) if E is None else E
            nsource = E.shape[0]
            default_Q = np.zeros((order * nsource, order * nsource), dtype=np.float64)
            default_Q[0:nsource, 0:nsource] = E
        else:
            default_Q = block_diag(*[x.get_default_q(components=x, E=E) for x in components])
        return default_Q

    @staticmethod
    def get_coeff_mat(coeff):
        """ Create a transition matrix F from AR coefficients """
        order = len(coeff)
        return np.vstack([coeff, np.hstack([np.eye(order - 1, dtype=np.float64),
                                            np.zeros((order - 1, 1), dtype=np.float64)])])

    # Parameter estimation methods (M step)
    def m_estimate(self, **kwargs):
        """
        Maximum likelihood or Maximum a posteriori estimation to update
        parameters. Calling super().m_estimate() for all Ssm parameters
        """
        results = super().m_estimate(**kwargs)
        self.update_comp_param()
        return results

    def update_comp_param(self):
        """ Update AutoRegModel specific parameters """
        self.fill_arn_param()

    def initialize_priors(self, Q_sigma2=None, Q_hyperparameter=None,
                          R_sigma2=None, R_hyperparameter=None):
        """ Initialize priors for autoregressive models """
        # base case
        if self.ncomp <= 1:
            # [Inverse gamma prior] on state noise covariance Q non-zero diagonal entries
            if Q_sigma2 is None:
                Q_sigma2 = self.Q[0, 0]
                Q_hyperparameter = 0.1 if Q_hyperparameter is None else Q_hyperparameter

            # [Inverse gamma prior] on observation noise variance R <--- TODO: update to Wishart
            if R_sigma2 is None:
                if self.R.shape[0] > 1:
                    raise NotImplementedError('Only uni-variate observation data is supported with prior for now.')
                else:
                    R_sigma2 = self.R[0, 0]
                    R_hyperparameter = 0.1 if R_hyperparameter is None else R_hyperparameter

            return dict_of(Q_sigma2, Q_hyperparameter, R_sigma2, R_hyperparameter)

        # recursive case
        else:
            assert self.components is not None, 'Cannot initialize priors outside base case when components is None.'
            components_prefill = self.fill_components(empty_comp=AutoRegModel(), deep_copy=True)

            # expand the specified prior values to the length of components
            Q_sigma2 = self._initialize_priors_recursive_list(Q_sigma2)
            R_sigma2 = self._initialize_priors_recursive_list(R_sigma2)
            R_hyperparameter = self._initialize_priors_recursive_list(R_hyperparameter)

            # construct the final priors that is a list of dictionaries
            priors = []
            for ii in range(self.ncomp):
                current_component: AutoRegModel = self.components[ii]
                assert current_component.type == 'arn', 'Component type is not AutoRegModel class.'
                priors.append(current_component.initialize_priors(Q_sigma2=Q_sigma2[ii], R_sigma2=R_sigma2[ii],
                                                                  R_hyperparameter=R_hyperparameter[ii]))

            # unfill the components
            self.unfill_components(components_prefill)

            return priors

    @staticmethod
    def _m_update_f(A=None, B=None, C=None, priors=None):
        """ Update transition matrix -- F """
        # Update the AR coefficients -- coeff (no prior)
        approach = 'svd' if A.shape[0] >= 5 else 'gaussian'
        coeff_new = B[0, :] @ inverse(A, approach=approach)

        # Construct transition matrix -- F
        F = AutoRegModel.get_coeff_mat(coeff_new)
        return F

    @staticmethod
    def _m_update_q(A=None, B=None, C=None, T=None, F=None, priors=None):
        """ Update state noise covariance matrix -- Q """
        coeff_new = F[0, :]
        Q_ss = C[0, 0] - 2 * B[0, :] @ coeff_new + coeff_new @ A @ coeff_new

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

        Q = np.zeros_like(F, dtype=np.float64)
        Q[0, 0] = sigma2_Q_new
        return Q

    @staticmethod
    def _m_update_mu0(x_0_n=None):
        """ Update initial state mean -- mu0 """
        mu0 = np.zeros((x_0_n.shape[0], 1), dtype=np.float64)
        mu0[0, 0] = x_0_n[0]
        return mu0

    @staticmethod
    def _m_update_mu0_src(x_0_n=None, nstate=None):
        """
        Update initial state mean mu0 in dynamic
        source localization
        """
        mu0 = np.zeros((x_0_n.shape[0], 1), dtype=np.float64)
        mu0[0::nstate, 0] = x_0_n[0::nstate]
        return mu0

    @staticmethod
    def _m_update_q0(x_0_n=None, P_0_n=None, mu0=None):
        """ Update initial state covariance -- Q0 """
        Q0 = np.zeros(P_0_n.shape, dtype=np.float64)
        Q0[0, 0] = P_0_n[0, 0] + x_0_n[0]**2 - 2 * x_0_n[0] * mu0[0, 0] + mu0[0, 0]**2
        return Q0

    @staticmethod
    def _m_update_g(y=None, x_t_n=None, P_t_n=None, h_t=None):
        """ Update observation matrix -- G (AutoRegModel has fixed G) """
        return None
