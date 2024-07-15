"""
Author: Mingjian He <mh1@stanford.edu>

decimate module contains methods for decimated state-space model learning
"""

from somata.basic_models import StateSpaceModel as Ssm
from somata.exact_inference import run_em
import numpy as np


class DecimatedModel(object):
    """
    DecimatedModel is an object class created to organize
    processing for decimated state-space model learning

    The structure of this class is very similar to the
    somata.switching.vb.VBSwitchModel class since both
    algorithms assume a set of parallel and independent
    state-space models in the hidden states.
    """
    ssm_array = None
    K = None
    T = None
    q = None
    logL = None

    def __init__(self, ssm_array):
        """
        Constructor method for DecimatedModel class
        :param ssm_array: an array of Ssm class objects with decimated observations
        """
        # The array length is the decimation factor
        self.K = len(ssm_array)

        # Verify that each Ssm object contains observed data
        for ii in range(self.K):
            assert ssm_array[ii].y is not None, 'No observed data contained in the ssm_array[%d] object.' % ii
            assert ssm_array[ii].ntime > 0, 'Observed data have length 0 in the ssm_array[%d] object.' % ii
            assert ssm_array[ii].nchannel > 0, 'Observed data have 0 channel in the ssm_array[%d] object.' % ii
        self.ssm_array = ssm_array

        # Get the total number of time points
        self.T = np.sum([ssm.ntime for ssm in ssm_array])

        # Get the number of channels that should be the same across objects
        qs = [ssm.nchannel for ssm in ssm_array]
        assert len(np.unique(qs)) == 1, 'Objects in ssm_array have observed data with different numbers of channels.'
        self.q = qs[0]

    def __repr__(self):
        """ Unambiguous and concise representation when calling DecimatedModel() """
        return 'Dec(' + str(self.K) + ')<' + hex(id(self))[-4:] + '>'

    def __str__(self):
        """ Helpful information when calling print(DecimatedModel()) """
        print_str = "Dec model " + str(self.ssm_array)
        return print_str

    def learn(self, keep_param=(), priors=None, max_iter=100, logL_thresh=1e-6, show_pbar=False):
        """
        Decimated learning of state-space models

        Perform EM learning by pooling sums of squares across
        parallel state-space models to update a single set
        of parameters for the lagged sequences that decimate
        the original time series with a high sampling rate.

        Inputs:
        :param keep_param: a tuple of strings for parameters to keep and not update in M step
        :param priors: a single dictionary specifying priors for all models, if None -> MLE
        :param max_iter: maximal number of EM iterations
        :param logL_thresh: threshold below which EM iteration stops
        :param show_pbar: show progress bar during EM iterations
        """
        # EM iterations
        self.logL = [float('-inf')] if self.logL is None else self.logL
        m_kwargs = {'priors': priors, 'keep_param': keep_param}
        em_log = run_em(self, m_kwargs=m_kwargs, max_iter=max_iter, stop_thresh=logL_thresh,
                        return_dict=True, show_pbar=show_pbar)
        return em_log

    def e_step(self, y=None):
        """
        E step in decimated oscillator learning, exposed for run_em()

        Using parallel Kalman (DeJong version) filtering and smoothing.

        Inputs:
        :param y: observed data (should be None only)
        """
        assert y is None, 'A single sequence of observed data should not be inputted in decimated oscillator learning.'

        x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_all, *_ = Ssm.par_kalman(self.ssm_array, method='dejong',
                                                                             skip_check_observed=True)
        e_results = {'x_t_n_all': x_t_n_all, 'P_t_n_all': P_t_n_all, 'P_t_tmin1_n_all': P_t_tmin1_n_all}

        logL = float(np.sum([logL.sum() for logL in logL_all]))  # total logL summed across sequences
        stop_var = logL - self.logL[-1]  # logL should monotonically increase
        self.logL.append(logL)

        return e_results, stop_var

    def m_step(self, y=None, x_t_n_all=None, P_t_n_all=None, P_t_tmin1_n_all=None, priors=None, keep_param=()):
        """
        M step in decimated oscillator learning, exposed for run_em()

        Inputs:
        :param y: observed data (should be None only)
        :param x_t_n_all: a list of smoothed estimates (posterior) of state mean
        :param P_t_n_all: a list of smoothed estimates (posterior) of state conditional covariance
        :param P_t_tmin1_n_all: a list of smoothed estimates (posterior) of state lag1 conditional cross-covariance
        :param priors: a single dictionary specifying priors for all models, if None -> MLE
        :param keep_param: a tuple of strings for parameters to keep and not update
        """
        assert y is None, 'A single sequence of observed data should not be inputted in decimated oscillator learning.'

        # Compute sums of squares from each decimated sequence
        R_ss = np.zeros((self.q, self.q, self.K), dtype=np.float64)
        A, B, C, x_0_n, P_0_n = ([], [], [], [], [])
        for k in range(self.K):
            m_results = self.ssm_array[k].m_estimate(x_t_n=x_t_n_all[k], P_t_n=P_t_n_all[k],
                                                     P_t_tmin1_n=P_t_tmin1_n_all[k],
                                                     priors=priors, force_ABC=True, update_param=(), return_dict=True)
            # Store sum variables for joint estimations
            R_ss[:, :, k] = m_results['R_ss']
            A.append(m_results['A'])
            B.append(m_results['B'])
            C.append(m_results['C'])
            x_0_n.append(x_t_n_all[k][:, 0][:, None])
            P_0_n.append(P_t_n_all[k][:, :, 0])

        # TODO: an important assumption now is that there is only a single component in these ssm_array objects.
        #       We need to think about what to do when different components require different decimation rates.
        assert np.all([ssm.ncomp == 1 for ssm in self.ssm_array]), 'Only single component Ssm objects are supported.'

        # Obtain boolean flags of scopes of updates
        # noinspection PyProtectedMember
        update_FQ, update_mu0S0G = Ssm._m_estimate_scope(keep_param=keep_param)

        # Joint estimation of parameters
        if 'R' not in keep_param:
            R_ss = R_ss.sum(axis=2)
            # noinspection PyProtectedMember
            _ = [ssm._m_update_r(R_ss=R_ss, T=self.T, priors=priors) for ssm in self.ssm_array]

        # Call model specific _m_update_<param> methods
        if update_FQ:
            # Sum across decimated sequences
            A_tmp = np.dstack(A).sum(axis=2)
            B_tmp = np.dstack(B).sum(axis=2)
            C_tmp = np.dstack(C).sum(axis=2)

            if 'F' not in keep_param:
                # noinspection PyProtectedMember
                F = self.ssm_array[0]._m_update_f(A=A_tmp, B=B_tmp, priors=priors)
            else:
                F = self.ssm_array[0].F  # assume that all models already have the same F

            if 'Q' not in keep_param:
                # noinspection PyProtectedMember
                Q = self.ssm_array[0]._m_update_q(A=A_tmp, B=B_tmp, C=C_tmp, T=self.T, F=F, priors=priors)

        if update_mu0S0G:
            # noinspection PyUnboundLocalVariable
            C_tmp = np.dstack(C).sum(axis=2) if not update_FQ else C_tmp
            x_0_n_tmp = np.dstack(x_0_n).sum(axis=2) / self.K
            P_0_n_tmp = np.dstack(P_0_n).sum(axis=2) / self.K

            if 'mu0' not in keep_param:
                # noinspection PyProtectedMember
                mu0 = self.ssm_array[0]._m_update_mu0(x_0_n=x_0_n_tmp[:, 0])
            else:
                mu0 = self.ssm_array[0].mu0  # assume that all models already have the same mu0

            if 'S0' not in keep_param:
                # noinspection PyProtectedMember
                S0 = self.ssm_array[0]._m_update_S0(x_0_n=x_0_n_tmp[:, 0], P_0_n=P_0_n_tmp, mu0=mu0)

            if 'G' not in keep_param:
                # an additional sum of square needs to be calculated to update G
                D_tmp = np.dstack([self.ssm_array[k].y @ x_t_n_all[k][:, 1:].T for k in range(self.K)]).sum(axis=2)

                # noinspection PyProtectedMember
                G = self.ssm_array[0]._m_update_g(C=C_tmp, D=D_tmp)

        # Distribute to all models
        if update_FQ or update_mu0S0G:
            for k in range(self.K):
                if 'F' not in keep_param:
                    # noinspection PyUnboundLocalVariable
                    self.ssm_array[k].F = F

                if 'Q' not in keep_param:
                    # noinspection PyUnboundLocalVariable
                    self.ssm_array[k].Q = Q

                if 'mu0' not in keep_param:
                    # noinspection PyUnboundLocalVariable
                    self.ssm_array[k].mu0 = mu0

                if 'S0' not in keep_param:
                    # noinspection PyUnboundLocalVariable
                    self.ssm_array[k].S0 = S0

                if 'G' not in keep_param:
                    # noinspection PyUnboundLocalVariable
                    self.ssm_array[k].G = G if G is not None else self.ssm_array[k].G

                # make sure instance attributes reflect the updated parameters
                self.ssm_array[k].update_comp_param()
