"""
Author: Mingjian He <mh1@stanford.edu>

gen module contains general Gaussian state-space model methods used in SOMATA
"""

from somata.basic_models import StateSpaceModel as Ssm
from somata.exact_inference import inverse
import numpy as np
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
        super().__init__(components=components, F=F, Q=Q, mu0=mu0, Q0=Q0, G=G, R=R, y=y, Fs=Fs)
        self.default_G = np.ones((1, self.nstate), dtype=np.float64)

    def __repr__(self):
        """ Unambiguous and concise representation when calling GeneralSSModel() """
        return super().__repr__().replace('Ssm', 'Gen')

    def __str__(self):
        """ Helpful information when calling print(GeneralSSModel()) """
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

    def fill_components(self, empty_comp=None, deep_copy=True):
        """ Fill the components attribute with GeneralSSModel parameters """
        empty_comp = GeneralSSModel() if empty_comp is None else empty_comp
        return super().fill_components(empty_comp=empty_comp, deep_copy=deep_copy)

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

        return {'R_sigma2': R_sigma2, 'R_hyperparameter': R_hyperparameter}

    @staticmethod
    def _m_update_f(A=None, B=None, priors=None):
        """ Update transition matrix -- F """
        approach = 'svd' if A.shape[0] >= 5 else 'gaussian'
        F = B @ inverse(A, approach=approach)
        return F

    @staticmethod
    def _m_update_q(A=None, B=None, C=None, T=None, F=None, priors=None):
        """ Update state noise covariance matrix -- Q """
        Q = (C - B @ F.T - F @ B.T + F @ A @ F.T) / T
        return Q

    @staticmethod
    def _m_update_q_src(Q_ss=None, T=None, Q_basis=None, nsource=None, nstate=None, priors=None):
        """
        Update state noise covariance matrix Q in dynamic
        source localization, in the Q_basis block diagonal
        form
        """
        import torch
        from ..source_loc import SourceLocModel as Src
        assert Q_basis.shape == Q_ss.shape, 'GeneralSSModel has non-diagonal entries of component Q, ' \
                                            'therefore Q_basis must span the expanded space instead of nsource.'
        assert Q_ss.shape[0] == nsource * nstate, 'Q_ss does not match the dimension of GeneralSSModel with '\
                                                  + str(nstate) + ' states across ' + str(nsource) + ' sources.'

        # GeneralSSModel has a full matrix of sum of squares
        if type(Q_basis) is list:  # using non-orthonormal kernel
            Theta = Src.update_theta(Q_ss=Q_ss, T=T, Q_basis=Q_basis[1],
                                     nsource=nsource, npart=1, priors=priors)
            Q_new = Q_basis[0] @ torch.diag(Theta) @ Q_basis[0].T
        else:  # using orthonormal basis
            Theta = Src.update_theta(Q_ss=Q_ss, T=T, Q_basis=Q_basis,
                                     nsource=nsource, npart=1, priors=priors)
            Q_new = Q_basis @ torch.diag(Theta) @ Q_basis.T
        return Q_new
