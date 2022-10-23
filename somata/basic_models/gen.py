"""
Author: Mingjian He <mh105@mit.edu>

gen module contains general Gaussian state-space model methods used in SOMATA
"""

from somata.basic_models import StateSpaceModel as Ssm
from somata.exact_inference import inverse
import numpy as np
from sorcery import dict_of
from scipy.linalg import block_diag


class GeneralSSModel(Ssm):
    """
    GeneralSSModel is a subclass of StateSpaceModel class dedicated
    to a general single component structure of linear Gaussian
    state-space models
    """
    type = 'gen'
    default_G = None

    def __init__(self, components='Gen', F=None, Q=None, mu0=None, Q0=None, G=None, R=None, y=None, Fs=None):
        """
        Constructor method for GeneralSSModel class
        :param components: add a single component or None
        :param F: transition matrix
        :param Q: state noise covariance matrix
        :param mu0: initial state mean vector
        :param Q0: initial state covariance matrix
        :param G: observation matrix (row major)
        :param R: observation noise covariance matrix
        :param y: observed data (row major, can be multivariate)
        :param Fs: sampling frequency in Hz
        """
        # Call parent class constructor
        super().__init__(F=F, Q=Q, mu0=mu0, Q0=Q0, G=G, R=R, y=y, Fs=Fs)
        self.default_G = np.ones((1, self.nstate), dtype=np.float64)

        # Fill in the components attribute
        if components == 'Gen':
            component = GeneralSSModel(components=None)  # type: ignore
            component.default_G = np.ones((1, self.nstate), dtype=np.float64)
            self.components = [component]
            self.ncomp = 1
            self.comp_nstates = [self.nstate]

    def __repr__(self):
        return super().__repr__().replace('Ssm', 'Gen')

    def __str__(self):
        print_str = super().__str__().replace('<Ssm object at', '<Gen object at')
        return print_str

    def get_default_q(self, components=None, E=None):
        """
        Get the default structure of state noise covariance
        matrix Q in the Q_basis block diagonal form
        """
        components = self.components if components is None else components
        if type(components) is not GeneralSSModel:
            assert len(components) == 1, 'More than one component for GeneralSSModel object is not permitted.'
            components = components[0]
            assert type(components) is GeneralSSModel, 'components type is not GeneralSSModel.'

        order = components.default_G.shape[1]
        E = np.eye(1, dtype=np.float64) if E is None else E
        default_Q = block_diag(*[E] * order)
        return default_Q

    # Parameter estimation methods (M step)
    def initialize_priors(self, R_sigma2=None, R_hyperparameter=None):
        """ Initialize priors for general state-space component """
        assert self.ncomp <= 1, 'GeneralSSModel object should not have components.'

        # [Inverse gamma prior] on observation noise variance R <--- TODO: update to Wishart
        if R_sigma2 is None:
            if self.R.shape[0] > 1:
                raise NotImplementedError('Only uni-variate observation data is supported with prior for now.')
            else:
                R_sigma2 = self.R[0, 0]
                R_hyperparameter = 0.1 if R_hyperparameter is None else R_hyperparameter

        return dict_of(R_sigma2, R_hyperparameter)

    @staticmethod
    def _m_update_f(A=None, B=None, C=None, q_old=None, priors=None):
        """ Update transition matrix -- F """
        approach = 'svd' if A.shape[0] >= 5 else 'gaussian'
        F = B @ inverse(A, approach=approach)
        return F

    @staticmethod
    def _m_update_q(A=None, B=None, C=None, T=None, F=None, priors=None):
        """ Update state noise covariance matrix -- Q """
        Q = (C - B @ F.T - F @ B.T + F @ A @ F.T) / T
        return Q
