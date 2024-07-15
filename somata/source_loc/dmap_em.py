"""
Author: Mingjian He <mh1@stanford.edu>

dmap_em module contains dynamic source localization codes used in SOMATA
"""

import torch
import mne
import numpy as np
import numbers
from collections.abc import Iterable
from copy import deepcopy
from joblib import cpu_count
from somata.basic_models import StateSpaceModel
from somata.exact_inference import run_em, inverse
from somata.utils import estimate_r
from somata.source_loc.source_loc_utils import smart_djkalman_conv_torch, get_whitener
from scipy.linalg import qr, block_diag, sqrtm
from scipy.stats import norm
import matplotlib.pyplot as plt

# Turn off tensor 32bit precision to allow full 32bit floating point
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SourceLocModel(object):
    """
    SourceLocModel is an object class dedicated to dynamic source
    localization inference in SOMATA, supporting all somata classes
    and multiple components
    """
    nstate = None  # hidden state dimension
    ncomp = None  # number of components
    nchannel = None  # number of channels
    nsource = None  # number of sources
    components = None  # a list of independent components
    comp_nstates = None  # a list of state dimensions of components
    proj = None  # projection matrix on observed data
    lfm = None  # lead field gain matrix
    G = None  # observation matrix
    Ps = None  # permutation matrix
    Ms = None  # mixing matrix
    B = None  # adjacency matrix
    Q_basis = None  # orthonormal basis of adjacency matrix
    ssm_params = None  # state-space model parameters
    em_log = None  # dictionary of EM iteration numbers and changes in log likelihoods

    def __init__(self, components, proj=None, lfm=None, B=None, src=None, fwd=None, **kwargs):
        """
        Constructor method for SourceLocModel class
        :param components: a list of independent components in the model
        :param proj: projection matrix on observed data (default to average referencing)
        :param lfm: forward model lead field gain matrix (lfm < fwd)
        :param B: adjacency matrix for the basis of source noise covariance (B < src)
        :param src: MNE SourceSpaces instance with anatomical information (src < fwd)
        :param fwd: MNE Forward instance with source information (fwd takes the highest priority)
        """
        # Process the list of independent components underlying all sources
        self.ncomp = len(components)
        self.components = list(components) if isinstance(components, Iterable) else [components]
        self.comp_nstates = [x.default_G.shape[1] for x in self.components]

        # Process Forward if provided (takes precedence over lfm and src inputs)
        if fwd is not None:
            assert lfm is None and src is None, 'Forward provided in fwd. lfm and src inputs will be overwritten.'
            lfm, src = self._process_fwd_input(fwd)

        # Set up forward model and related attributes
        self.lfm = deepcopy(np.asanyarray(lfm, dtype=np.float64))
        self.nchannel, self.nsource = lfm.shape
        self.nstate = sum(self.comp_nstates) * self.nsource
        self.G = self._build_observation_matrix()
        self.Ps = [self._build_permutation_matrix(nstate=x) for x in self.comp_nstates]

        # Process the projection matrix
        if proj is None:  # default to average referencing projector
            indi_vec = np.ones((self.nchannel, 1), dtype=np.float64)
            self.proj = np.eye(self.nchannel, dtype=np.float64) - indi_vec @ indi_vec.T / (indi_vec.T @ indi_vec)
        else:
            self.proj = proj

        # Process SourceSpaces if provided to set up spatial dependencies
        if src is not None:
            assert B is None, 'SourceSpaces provided in src. adjacency matrix input B will be overwritten.'
            assert len(src[0]['vertno']) + len(src[1]['vertno']) == self.nsource, \
                'Total vertno is different from nsource.'
            d1, d2, m1, m2, scale, approach, orthonormal = self._process_src_scalars(**kwargs)
            B = self._build_adjacency_matrix(src=src, d1=d1, d2=d2, scale=scale, approach=approach)
            self.Ms = [self._build_mixing_matrix(nstate=x, src=src, m1=m1, m2=m2) for x in self.comp_nstates]
        else:
            self.Ms = [np.eye(x * self.nsource, dtype=np.float64) for x in self.comp_nstates]
            orthonormal = None

        # Create the basis for noise covariance across sources
        self.B = np.eye(self.nsource, dtype=np.float64) if B is None else B
        self.Q_basis = self._get_q_basis(self.B, orthonormal=orthonormal)

    # Dunder methods - magic methods
    def __repr__(self):
        """ Unambiguous and concise representation when calling SourceLocModel() """
        return 'Src(' + str(self.ncomp) + ')<' + hex(id(self))[-4:] + '>'

    def __str__(self):
        """ Helpful information when calling print(SourceLocModel()) """
        print_str = "<Src object at " + hex(id(self)) + ">\n " + \
                    "{0:8} = {1: <5} ".format("nstate", str(self.nstate)) + \
                    "{0:8} = {1}\n ".format("ncomp", str(self.ncomp)) + \
                    "{0:8} = {1: <5} ".format("nchannel", str(self.nchannel)) + \
                    "{0:8} = {1}\n ".format("nsource", str(self.nsource)) + \
                    "{0:12} = {1}\n ".format("comp_nstates", str(self.comp_nstates)) + \
                    "components = {}\n ".format(str(self.components)) + \
                    "{0:3}.shape = {1}\n ".format("lfm", [str(x.shape) if x is not None else 'None' for x
                                                          in [self.lfm]][0]) + \
                    "{0:3}.shape = {1}\n ".format("G", [str(x.shape) if x is not None else 'None' for x
                                                        in [self.G]][0]) + \
                    "{0:8} = {1} ".format("ssm_params", [str(x.keys()) if x is not None else 'None' for x
                                                         in [self.ssm_params]][0])
        return print_str

    def __len__(self):
        return self.ncomp

    # EM methods - dynamic source localization with Expectation-Maximization algorithm
    def learn(self, y, rank=None, SNR=3, R=None, Q=None, mu0=None, S0=None,
              priors=None, Q_prior_option='MLE', R_prior_option='MLE',
              dtype=torch.float32, update_param=('F', 'Q', 'R'), keep_param=(),
              max_iter=10, logL_thresh=1e-6, show_pbar=True):
        """
        Dynamic Source Localization with MAP EM Learning

        The algorithm is based on the following reference, with substantial
        modifications made to flexibly handle all somata classes and compute
        more efficiently and robustly for other SSM structures beyond AR1.

        Reference:
            Pirondini, E., Babadi, B., Obregon-Henao, G., Lamus, C., Malik, W. Q.,
            Hämäläinen, M. S., & Purdon, P. L. (2017). Computationally efficient
            algorithms for sparse, dynamic solutions to the EEG source localization
            problem. IEEE Transactions on Biomedical Engineering, 65(6), 1359-1372.

        Inputs:
        :param y: observed data (row major, should be multivariate)
        :param rank: rank of observed data (use of mne.compute_rank() is recommended)
        :param SNR: signal-to-noise ratio of amplitude in scalp measurement
        :param R: observation noise covariance matrix or a scalar
        :param Q: state noise covariance matrix
        :param mu0: initial state mean vector
        :param S0: initial state covariance matrix
        :param priors: a list of dictionaries specifying priors for each component, if None -> initialize_priors()
        :param Q_prior_option: a string selecting a prior to use for MAP estimating Q
        :param R_prior_option: a string selecting a prior to use for MAP estimating R
        :param dtype: numerical precision data type for GPU processing
        :param update_param: a tuple of strings for parameters to update
        :param keep_param: a tuple of strings for parameters to keep and not update in M step
        :param max_iter: maximal number of EM iterations in dMAP-EM
        :param logL_thresh: threshold below which EM iteration stops
        :param show_pbar: show progress bar during EM iterations
        """
        # Initialize state-space model parameters
        R, rank = self._initialize_r(y=y, rank=rank, R=R, cutoff_frequency=35)
        mu0 = np.zeros((self.nstate, 1), dtype=np.float64) if mu0 is None else mu0
        S0, sigma2_S0 = self._initialize_S0(R=R, rank=rank, S0=S0, SNR=SNR, diagonal=True)
        Q, sigma2_Q = self._initialize_q(S0=S0, sigma2_S0=sigma2_S0, Q=Q)

        # Move attribute matrices to cuda device
        self.Ps = [torch.as_tensor(P, dtype=dtype).cuda() for P in self.Ps]
        self.Ms = [torch.as_tensor(M, dtype=dtype).cuda() for M in self.Ms]
        if isinstance(self.Q_basis, list):
            self.Q_basis = [torch.as_tensor(q, dtype=dtype).cuda() for q in self.Q_basis]
        else:
            self.Q_basis = torch.as_tensor(self.Q_basis, dtype=dtype).cuda()

        # Initialize E-step using a dictionary of state-space model parameters
        y = torch.as_tensor(y, dtype=dtype).cuda()
        F = self._build_transition_matrix()
        ssm_params = {'F': F, 'Q': Q, 'mu0': mu0, 'S0': S0, 'G': self.G, 'R': R}
        for elem in ssm_params.keys():  # convert to torch data type for GPU processing
            ssm_params[elem] = torch.as_tensor(ssm_params[elem], dtype=dtype).cuda()
        self.ssm_params = ssm_params

        # Initialize priors dictionary for M-step
        priors = self.initialize_priors(sigma2_Q=sigma2_Q, sigma2_S0=sigma2_S0, Q_prior_option=Q_prior_option,
                                        R=ssm_params['R'].detach().clone(), T=y.shape[1],
                                        R_prior_option=R_prior_option) if priors is None else priors

        # Learn with EM algorithm
        e_kwargs = {'logL_list': [float('-inf')], 'rank': rank}
        m_kwargs = {'priors': priors, 'update_param': update_param, 'keep_param': keep_param}
        self.em_log = run_em(self, y=y, e_kwargs=e_kwargs, m_kwargs=m_kwargs, max_iter=max_iter,
                             stop_thresh=logL_thresh, ignore_numerr=True, return_dict=True, show_pbar=show_pbar)

        # Final Kalman smoothing to get hidden state estimates
        x_t_n, P_t_n, *_, = smart_djkalman_conv_torch(y=y, rank=rank, proj=self.proj, **self.ssm_params)

        return x_t_n.cpu().numpy(), P_t_n.cpu().numpy()

    def e_step(self, y, rank=None, logL_list=None):
        """ E-step in dMAP-EM, exposed for run_em() """
        x_t_n, P_t_n, P_t_tmin1_n, logL, break_conv, *_ = smart_djkalman_conv_torch(
            y=y, rank=rank, proj=self.proj, **self.ssm_params)
        e_logL = torch.sum(logL)
        stop_var = abs(e_logL - logL_list[-1])  # due to convergent Kalman, may not monotonically increase
        logL_list.append(e_logL)

        # Announce the flag of reaching numerical precision limit
        stop_var = True if break_conv or torch.isnan(e_logL) else stop_var

        return {'x_t_n': x_t_n, 'P_t_n': P_t_n, 'P_t_tmin1_n': P_t_tmin1_n}, stop_var

    def m_step(self, y, x_t_n=None, P_t_n=None, P_t_tmin1_n=None, priors=None,
               update_param=('F', 'Q', 'R'), keep_param=()):
        """ M-step in dMAP-EM, exposed for run_em() """
        # Obtain boolean flags of scopes of updates
        update_FQ, update_mu0S0 = self._m_step_scope(update_param, keep_param)

        # Initialize parameters
        T = y.shape[1]
        if priors is None:
            priors = [None] * self.ncomp
        elif isinstance(priors, dict):
            priors = [priors]  # so that it can be indexed to position 0

        # Attempt to skip sums of squares computation in the steady-state version
        if update_FQ:
            A = P_t_n * T + x_t_n[:, :-1] @ x_t_n[:, :-1].T
            B = P_t_tmin1_n * T + x_t_n[:, 1:] @ x_t_n[:, :-1].T
            C = P_t_n * T + x_t_n[:, 1:] @ x_t_n[:, 1:].T
        else:
            A, B, C = (None, None, None)

        # Update parameters for each independent component -- F, Q, mu0, S0 (component specific priors)
        if update_FQ or update_mu0S0:
            # Initialize variables to store updated parameters
            if 'Q' in update_param and 'Q' not in keep_param:
                Q_new = torch.zeros(P_t_n.shape, dtype=P_t_n.dtype).cuda()
            else:
                Q_new = None

            if 'mu0' in update_param and 'mu0' not in keep_param:
                mu0_new = torch.zeros((self.nstate, 1), dtype=x_t_n.dtype).cuda()
            elif 'S0' in update_param and 'S0' not in keep_param:
                mu0_new = self.ssm_params['mu0']
            else:
                mu0_new = None

            if 'S0' in update_param and 'S0' not in keep_param:
                S0_new = torch.zeros(P_t_n.shape, dtype=P_t_n.dtype).cuda()
            else:
                S0_new = None

            # Iterate through independent components
            for ii in range(self.ncomp):
                current_component = self.components[ii]
                start_idx = sum(self.comp_nstates[:ii]) * self.nsource
                end_idx = sum(self.comp_nstates[:ii + 1]) * self.nsource
                if update_FQ:
                    A_tmp = A[start_idx:end_idx, start_idx:end_idx]
                    B_tmp = B[start_idx:end_idx, start_idx:end_idx]
                    C_tmp = C[start_idx:end_idx, start_idx:end_idx]
                else:
                    A_tmp, B_tmp, C_tmp = (None, None, None)

                # Call the _m_update_<param> methods specific to the component subclass
                if 'F' in update_param and 'F' not in keep_param:
                    # Combine sums of squares across sources for the same component
                    ABC = self._combine_src_ss(self.comp_nstates[ii], self.Ms[ii], A_tmp, B_tmp, C_tmp)
                    # noinspection PyProtectedMember
                    current_component.F = current_component._m_update_f(A=ABC['A'], B=ABC['B'], priors=priors[ii])
                    current_component.update_comp_param()  # update component specific parameters

                if 'Q' in update_param and 'Q' not in keep_param:
                    # Compute the sum of squares inside the trace term of complete loglikelihood that involves Q
                    F_tmp = self._build_transition_matrix(components=[current_component], Ms=[self.Ms[ii]])
                    Q_ss = C_tmp - B_tmp @ F_tmp.T - F_tmp @ B_tmp.T + F_tmp @ A_tmp @ F_tmp.T

                    # Re-order sum of squares into Q_basis block diagonal form
                    P = self.Ps[ii]
                    Q_ss = P.T @ Q_ss @ P

                    # Call the _m_update_q_src method specific to the component subclass
                    # noinspection PyProtectedMember
                    Q_new_ii = current_component._m_update_q_src(Q_ss=Q_ss, T=T, Q_basis=self.Q_basis,
                                                                 nsource=self.nsource,
                                                                 nstate=self.comp_nstates[ii],
                                                                 priors=priors[ii])

                    # Re-order Q_new_ii back into the Kalman form
                    Q_new[start_idx:end_idx, start_idx:end_idx] = P @ Q_new_ii @ P.T

                if 'mu0' in update_param and 'mu0' not in keep_param:
                    # noinspection PyProtectedMember
                    mu0_new[start_idx:end_idx, 0] = \
                        current_component._m_update_mu0_src(x_0_n=x_t_n[start_idx:end_idx, 0],
                                                            nstate=self.comp_nstates[ii])[:, 0]

                if 'S0' in update_param and 'S0' not in keep_param:
                    x_0_n = x_t_n[start_idx:end_idx, 0][:, None]
                    mu0_tmp = mu0_new[start_idx:end_idx, 0][:, None]

                    # Compute the posterior covariance at t=0
                    S0_ss = P_t_n + x_0_n @ x_0_n.T - x_0_n @ mu0_tmp.T - mu0_tmp @ x_0_n.T + mu0_tmp @ mu0_tmp.T

                    # Re-order the covariance into Q_basis block diagonal form
                    P = self.Ps[ii]
                    S0_ss = P.T @ S0_ss @ P

                    # Re-use the _m_update_q_src method specific to the component subclass
                    # noinspection PyProtectedMember
                    S0_new_ii = current_component._m_update_q_src(Q_ss=S0_ss, T=1, Q_basis=self.Q_basis,
                                                                  nsource=self.nsource,
                                                                  nstate=self.comp_nstates[ii],
                                                                  priors=None)

                    # Re-order S0_new_ii back into the Kalman form
                    S0_new[start_idx:end_idx, start_idx:end_idx] = P @ S0_new_ii @ P.T

        # Update the dictionary of ssm parameters
        if 'F' in update_param and 'F' not in keep_param:
            F_new = self._build_transition_matrix()
            self.ssm_params.update({'F': F_new})

        if 'Q' in update_param and 'Q' not in keep_param:
            # noinspection PyUnboundLocalVariable
            self.ssm_params.update({'Q': Q_new})

        if 'mu0' in update_param and 'mu0' not in keep_param:
            # noinspection PyUnboundLocalVariable
            self.ssm_params.update({'mu0': mu0_new})

        if 'S0' in update_param and 'S0' not in keep_param:
            # noinspection PyUnboundLocalVariable
            self.ssm_params.update({'S0': S0_new})

        if 'R' in update_param and 'R' not in keep_param:
            R_new = self._m_update_r(y=y, x_t_n=x_t_n, P_t_n=P_t_n, T=T, priors=priors[0])
            self.ssm_params.update({'R': R_new})

        return

    def _combine_src_ss(self, nstate, M, A_tmp, B_tmp, C_tmp):
        """ Combine sums of squares across sources for the same component """
        # Apply the mixing matrix associated with transition matrix
        A_hat = M @ A_tmp @ M.T
        B_hat = B_tmp @ M.T
        C_hat = C_tmp

        # Combine sums of squares across sources
        A = torch.zeros((nstate, nstate), dtype=A_tmp.dtype).cuda()
        B = torch.zeros((nstate, nstate), dtype=B_tmp.dtype).cuda()
        C = torch.zeros((nstate, nstate), dtype=C_tmp.dtype).cuda()
        for vidx in range(self.nsource):
            A += A_hat[vidx * nstate:(vidx + 1) * nstate, vidx * nstate:(vidx + 1) * nstate]
            B += B_hat[vidx * nstate:(vidx + 1) * nstate, vidx * nstate:(vidx + 1) * nstate]
            C += C_hat[vidx * nstate:(vidx + 1) * nstate, vidx * nstate:(vidx + 1) * nstate]
        return {'A': A, 'B': B, 'C': C}

    def _m_update_r(self, y=None, x_t_n=None, P_t_n=None, T=None, priors=None):
        """ Update observation noise covariance """
        G = self.ssm_params['G']
        pred_err = y - G @ x_t_n[:, 1:]
        R_ss = pred_err @ pred_err.T + G @ (P_t_n * T) @ G.T

        # Due to convergent Kalman, R_ss diagonal could be negative. We lower-bound it to 0
        lower_bound_index = torch.where(torch.lt(R_ss, 0))[0]
        R_ss[lower_bound_index, lower_bound_index] = 0.
        # if len(lower_bound_index) > 0:
        #     print('Encountered negative diagonal elements when updating R!')

        # noinspection PyProtectedMember
        if StateSpaceModel._m_update_if_mle('R_wishart', priors):
            # MLE
            R_new = R_ss / T
        else:
            # MAP with Wishart prior
            R_new = (R_ss + priors['R_wishart']['v'] * priors['R_wishart']['R_prior']) / (
                    T + priors['R_wishart']['v'] + priors['R_wishart']['p'] + 1)

        return R_new

    # Matrix building methods - construct important matrices for dmap_em()
    def _build_observation_matrix(self):
        """ Construct observation matrix G """
        stacked_G = []
        for comp in self.components:
            stacked_G.append(np.hstack([self.lfm[:, ii][:, None] * comp.default_G
                                        for ii in range(self.lfm.shape[1])]))
        G = np.hstack(stacked_G)
        assert G.shape[0] == self.nchannel, 'Incorrect channel dimension of observation matrix G.'
        assert G.shape[1] == self.nstate, 'Incorrect state dimension of observation matrix G.'
        return G

    def _build_transition_matrix(self, components=None, Ms=None):
        """ Construct transition matrix F """
        components = self.components if components is None else components
        Ms = self.Ms if Ms is None else Ms
        F_blocks = [torch.block_diag(*[torch.as_tensor(components[x].F, dtype=Ms[x].dtype).cuda()] * self.nsource
                                     ) @ Ms[x] for x in range(len(components))]
        F = torch.block_diag(*F_blocks).cuda()
        return F

    def _build_permutation_matrix(self, nstate):
        """
        Construct the elementary row/column permutation matrix
        to transform a block diagonal Q basis into the Q block
        for one independent component.

        The relation is: P @ Q_basis @ P.T = Q
        """
        P = np.zeros((self.nsource * nstate, self.nsource * nstate), dtype=int)
        for vidx in range(self.nsource):
            for j in range(nstate):
                to_idx = vidx * nstate + j
                from_idx = j * self.nsource + vidx
                P[to_idx, from_idx] = 1
        return P

    def _build_mixing_matrix(self, nstate, src, m1=0.5, m2=0.5):
        """
        Construct the mixing matrix to linearly combine hidden states
        from first- and second-order neighbors when evolving the state
        equation, i.e., provide a right-multiplying mixing transform
        to the transition matrix.

        If a diagonal transition matrix is of the form:
             X_t = F @ X_t-1
        With a mixing matrix M, the state equation becomes:
             X_t = F @ M @ X_t-1

        The weighting on a source and its neighbors is as following:
            m1 * x + m2 * (d_N1 * N1(x)) + (1 - m1 - m2) * (d_N2 * N2(x))

        N.B.: Here the d_N1 are normalized to sum to 1 among first-order
        neighbors. Similarly, d_N2 are normalized to sum to 1 among
        second-order neighbors. For a justification of this, see Appendix A
        in the following reference.

        Reference:
            Lamus, C., Hämäläinen, M. S., Temereanca, S., Brown, E. N., & Purdon,
            P. L. (2012). A spatiotemporal dynamic distributed solution to the
            MEG inverse problem. NeuroImage, 63(2), 894-909.
        """
        # Default weighting ignores second-order neighbors
        m1 = 1.0 if m1 is None else m1
        m2 = 0.0 if m2 is None else m2
        assert m1 + m2 <= 1, 'The total weights must be less than or equal to 1.'

        # Compute the pairwise-distance between all sources in use
        if src[0]['dist'] is None:
            mne.add_source_space_distances(src, n_jobs=cpu_count())

        # Create the index mapping between vertex and source numbering
        vert_to_source, source_to_vert = self._vertex_source_mapping(src)

        # Define neighbors of all used vertices in original vertex numbering
        neighbors = self._define_neighbors(src)

        # Build the mixing matrix
        M = np.zeros((nstate * self.nsource, nstate * self.nsource), dtype=np.float64)

        for hemi in range(len(src)):
            for vidx in source_to_vert[hemi]:
                vert = source_to_vert[hemi][vidx]  # vertex indexing

                # Add the mixing block for the source itself
                M[vidx * nstate:(vidx + 1) * nstate, vidx * nstate:(vidx + 1) * nstate] = \
                    m1 * np.eye(nstate, dtype=np.float64)

                for order, m in zip(['first', 'second'], (m2, 1 - m1 - m2)):
                    vert_neighbor = np.asarray([vert_to_source[hemi].get(x, float('nan'))
                                                for x in neighbors[hemi][vert][order]])
                    vert_distance = np.asarray([src[hemi]['dist'][vert, x]
                                                for x in neighbors[hemi][vert][order]])

                    # Filter out neighbor vertices that are not sources
                    valid_idx = np.invert(np.isnan(vert_neighbor))
                    vert_neighbor = vert_neighbor[valid_idx].astype(dtype=int)
                    vert_distance = vert_distance[valid_idx]

                    # Normalize the inverse distances so they sum to one
                    inv_distance = 1 / vert_distance
                    inv_distance = inv_distance / inv_distance.sum()

                    # Add the mixing block for the neighbors
                    for v, d in zip(vert_neighbor, inv_distance):
                        M[vidx * nstate:(vidx + 1) * nstate, v * nstate:(v + 1) * nstate] = \
                            m * d * np.eye(nstate, dtype=np.float64)

        return M

    def _build_adjacency_matrix(self, src, d1=0.5, d2=0.25, scale=None, plot_kernel=False, approach=None):
        """
        Create adjacency matrix to capture spatial dependencies
        between sources based on the distance and neighboring
        relationship, i.e., the local topology of SourceSpaces
        """
        # Default scaling on first- and second-order neighbors
        d1 = 0.0 if d1 is None else d1
        d2 = 0.0 if d2 is None else d2
        approach = approach or 'normalize_neighbors'

        # Compute the pairwise-distance between all sources in use
        if src[0]['dist'] is None:
            mne.add_source_space_distances(src, n_jobs=cpu_count())

        # Create the index mapping between vertex and source numbering
        vert_to_source, source_to_vert = self._vertex_source_mapping(src)

        # Define neighbors of all used vertices in original vertex numbering
        neighbors = self._define_neighbors(src)

        # Build the adjacency matrix
        """
        There are different ways to construct the adjacency matrix and obtain
        a set of diagonal heavy kernels for update state noise covariance
        matrix Q.
        """
        if approach == 'normalize_neighbors':
            # (1) normalized and scaled distances from neighbors (Pirondini et al. 2017)
            B = np.eye(self.nsource, dtype=np.float64)

            # Populate the adjacency matrix with normalized inverse distances
            for hemi in range(len(src)):
                for vidx in source_to_vert[hemi]:
                    vert = source_to_vert[hemi][vidx]  # vertex indexing

                    for order, delta in zip(['first', 'second'], (d1, d2)):
                        vert_neighbor = np.asarray([vert_to_source[hemi].get(x, float('nan'))
                                                    for x in neighbors[hemi][vert][order]])
                        vert_distance = np.asarray([src[hemi]['dist'][vert, x]
                                                    for x in neighbors[hemi][vert][order]])

                        # Filter out neighbor vertices that are not sources
                        valid_idx = np.invert(np.isnan(vert_neighbor))
                        vert_neighbor = vert_neighbor[valid_idx].astype(dtype=int)
                        vert_distance = vert_distance[valid_idx]

                        # Normalize the inverse distances so they sum to one
                        inv_distance = 1 / vert_distance
                        inv_distance = inv_distance / inv_distance.sum()

                        # Fill the scaled and normalized inverse distances
                        B[vidx, vert_neighbor] = delta * inv_distance

        elif approach == 'symmetric_neighbors':
            # (2) symmetric and diagonal scaling from neighbors (Behtash Babadi)
            B = np.zeros((self.nsource, self.nsource), dtype=np.float64)
            N = np.zeros((self.nsource, self.nsource), dtype=int)

            # STEP 1: Populate the adjacency matrix with inverse distances
            for hemi in range(len(src)):
                for vidx in source_to_vert[hemi]:
                    vert = source_to_vert[hemi][vidx]  # vertex indexing

                    for order, n in zip(['first', 'second'], (1, 2)):
                        vert_neighbor = np.asarray([vert_to_source[hemi].get(x, float('nan'))
                                                    for x in neighbors[hemi][vert][order]])
                        vert_distance = np.asarray([src[hemi]['dist'][vert, x]
                                                    for x in neighbors[hemi][vert][order]])

                        # Filter out neighbor vertices that are not sources
                        valid_idx = np.invert(np.isnan(vert_neighbor))
                        vert_neighbor = vert_neighbor[valid_idx].astype(dtype=int)
                        vert_distance = vert_distance[valid_idx]

                        # Store the inverse distances and neighborhood status
                        B[vidx, vert_neighbor] = 1 / vert_distance
                        N[vidx, vert_neighbor] = n  # mark the neighbors for scaling

            # STEP 2: Add the sum of each row / column as the diagonal
            B_diagonal = B.sum(axis=0)
            B[range(self.nsource), range(self.nsource)] = B_diagonal

            # STEP 3: Normalize by the square root of diagonal row- and column-wise
            sqrt_B_diag = np.sqrt(B_diagonal)
            B = B / sqrt_B_diag[:, None] / sqrt_B_diag[None, :]

            # STEP 4: Scale the first- and second-order neighbors
            B[N == 1] *= d1
            B[N == 2] *= d2

            # STEP 5: Sanity checks
            assert np.allclose(B, B.T), 'Adjacency matrix is not symmetric.'
            assert np.allclose(np.diag(B), np.ones(self.nsource)), 'Diagonal of adjacency matrix is not unity.'

        elif approach == 'symmetric_gaussian':
            # (3) symmetric and Gaussian kernels from all sources (Mingjian He and Proloy Das)
            B = np.zeros((self.nsource, self.nsource), dtype=np.float64)

            if scale is None:
                # Depending on the icosahedron resolution, set Gaussian std so that
                # the adjacency matrix entry for a "typically" distanced first-order
                # neighbor of a source is fixed at 0.25
                if src[0]['nuse'] < 1000:  # ico3 has 642 srcs per hemi
                    scale = np.sqrt(0.0124 ** 2 / (-2 * np.log(0.25)))  # std = 0.00745
                elif src[0]['nuse'] < 3000:  # ico4 has 2562 srcs per hemi
                    scale = np.sqrt(0.0062 ** 2 / (-2 * np.log(0.25)))  # std = 0.00372
                elif src[0]['nuse'] < 11000:  # ico5 has 10242 srcs per hemi
                    scale = np.sqrt(0.0031 ** 2 / (-2 * np.log(0.25)))  # std = 0.00186
                else:
                    raise ValueError('Ico resolution is non-standard. Cannot set the std of Gaussian kernel.')

            # Visualize the Gaussian kernel on top of src-src distances
            if plot_kernel:
                all_dist = np.hstack([src[0]['dist'].data, src[1]['dist'].data])
                fig, ax = plt.subplots()
                ax.hist(all_dist, bins=1000)
                plt.xlabel('Distance (m)', fontsize='x-large')
                plt.ylabel('Counts', fontsize='x-large')
                plt.title('Histogram of src-src distances', fontsize='xx-large')
                ax2 = ax.twinx()
                x = np.linspace(start=0, stop=all_dist.max(initial=None), num=10000)
                ax2.plot(x, norm.pdf(x, loc=0, scale=scale) / norm.pdf(0, loc=0, scale=scale), color='red')
                plt.ylabel('Gaussian kernel weights', fontsize='x-large')

            # Populate the adjacency matrix with normalized Gaussian kernels
            for hemi in range(len(src)):
                hemi_verts = src[hemi]['vertno']  # vertex indexing
                hemi_vidxs = [vert_to_source[hemi].get(x, float('nan')) for x in hemi_verts]
                for vidx, vert in zip(hemi_vidxs, hemi_verts):
                    B[vidx, hemi_vidxs] = norm.pdf(src[hemi]['dist'][vert, hemi_verts].toarray(),
                                                   loc=0, scale=scale) / norm.pdf(0, loc=0, scale=scale)

            # Sanity checks
            assert np.allclose(B, B.T), 'Adjacency matrix is not symmetric.'
            assert np.allclose(np.diag(B), np.ones(self.nsource)), 'Diagonal of adjacency matrix is not unity.'

        else:
            raise ValueError('Specified approach to build adjacency matrix is not recognized.')

        return B

    # Initialization methods - initialize parameters for dMAP-EM
    def initialize_priors(self, sigma2_Q=None, sigma2_S0=None, Q_prior_option='MLE',
                          R=None, T=None, R_prior_option='MLE'):
        """
        Initialize priors for covariance matrices
        used for dMAP-EM source localization
        """
        priors = [{} for _ in self.components]

        # Add source localization specific priors to each component
        for prior in priors:
            # State noise covariance
            if Q_prior_option == 'inverse gamma':
                """ Non-informative inverse gamma prior (Lamus et al. 2012) """
                E_theta = sigma2_Q  # magnitude of source activity
                Var_theta = 1 / E_theta  # Variance >> Expectation
                alpha = E_theta ** 2 / Var_theta + 2  # approximately 2
                beta = E_theta * (E_theta ** 2 / Var_theta + 1)  # approximately E_theta
                prior['theta_ig'] = {'alpha': alpha, 'beta': beta}
            elif Q_prior_option == 'log-sum':
                """ Sparse L0 log-sum prior (Pirondini et al. 2017) """
                prior['theta_l0'] = {'gamma': 1 / sigma2_S0}
            elif Q_prior_option == 'exponential':
                """ Sparse L1 exponential prior (Pirondini et al. 2017) """
                gamma = np.asarray([-np.log(1 - (x + 0.5) / self.nsource) for x in range(self.nsource)]
                                   ).sum() / (self.nsource * sigma2_S0)
                prior['theta_l1'] = {'gamma': gamma}
            elif Q_prior_option == 'jeffreys':
                """ Sparse Jeffreys prior (Pirondini et al. 2017) """
                prior['theta_jeff'] = True

            # Observation noise covariance - Wishart prior
            if R_prior_option == 'wishart':
                prior['R_wishart'] = {'R_prior': R, 'p': self.nchannel, 'v': T}

        return priors

    def _initialize_r(self, y, rank=None, R=None, cutoff_frequency=35, plot_freqz=False):
        """ Initialize observation noise covariance matrix R """
        assert y.shape[0] == self.nchannel, 'Dimensions of y are incompatible with lfm.'

        # Check the rank of observed data
        rank = self.nchannel if rank is None else rank
        assert rank <= self.nchannel, 'Incorrect rank number. Cannot be greater than the number of channels.'

        # Estimate the observation noise covariance matrix R
        if isinstance(R, numbers.Number):
            R = R * np.eye(self.nchannel, dtype=np.float64)

        elif R is None:
            R = estimate_r(y, self.components[0].Fs, freq_cutoff=cutoff_frequency, plot_freqz=plot_freqz)

        # Update rank to be the rank of projected data
        eig, eigvec = mne.utils.linalg.eigh(R, overwrite_a=False)
        eig = np.ones(eig.shape, dtype=np.float64)
        eig[:-rank] = 0.0
        temp_R = self.proj @ eigvec * eig @ eigvec.T @ self.proj.T
        eig, _ = mne.utils.linalg.eigh(temp_R, overwrite_a=True)
        rank = np.sum(eig > 1e-2)

        return R, rank

    def _initialize_S0(self, R, rank=None, S0=None, SNR=3, diagonal=True):
        """
        Initialize state covariance matrix S0 at t=0

        Inputs:
        :param R: observation noise covariance matrix
        :param rank: rank of projected data
        :param SNR: amplitude SNR in scalp measurement
        :param diagonal: whether to initialize S0 as a diagonal matrix

        About SNR:
            - MNE uses amplitude SNR of 3, i.e., power SNR of 9
            - Lamus et al. 2012 uses power SNR of 5
            - Pirondini et al. 2017 uses power SNR of 3
        """
        if S0 is not None:
            sigma2_S0 = np.trace(S0) / S0.shape[0]

        else:
            # Lamus et al. 2012
            W, _ = get_whitener(R, rank=rank, proj=self.proj)
            WG = W @ self.G
            scale_S0 = rank / np.sum(WG ** 2)

            """
            The above is a more efficient and accurate calculation of the expression:

                scale_S0 = self.nchannel / np.trace(self.G @ self.G.T @ np.linalg.inv(R))

            MNE-Python as of v0.24.1 used the following expression:

                scale_S0 = np.trace(R) / np.trace(self.G @ self.G.T)

            that is the same as above when data is whitened.

            Note that Pirondini et al. 2017 used the second expression
            without whitening the data, resulting in an approximation
            with only the diagonal of R, which is inaccurate.
            """

            lambda2 = 1 / SNR ** 2
            sigma2_S0 = scale_S0 / lambda2

            S0_blocks = []
            for ii in range(self.ncomp):
                current_component = self.components[ii]
                if diagonal:
                    E = np.eye(self.nsource, dtype=np.float64)
                else:
                    E = self.Q_basis @ self.Q_basis.T
                S0_tmp = sigma2_S0 * current_component.get_default_q(components=current_component, E=E)
                S0_blocks.append(self.Ps[ii] @ S0_tmp @ self.Ps[ii].T)  # re-order into the Kalman form

            S0 = block_diag(*S0_blocks)

        return S0, sigma2_S0

    @staticmethod
    def _initialize_q(S0, sigma2_S0, Q=None, src_scalp_SNR_ratio=0.1):
        """ Initialize state covariance matrix Q """
        if Q is not None:
            sigma2_Q = np.trace(Q) / Q.shape[0]

        else:
            # Lamus et al. 2012 introduced this heuristic scaling, assuming
            # SNR in the source space is 1/10 of the SNR in scalp measurement
            Q = src_scalp_SNR_ratio * S0
            sigma2_Q = src_scalp_SNR_ratio * sigma2_S0

        return Q, sigma2_Q

    @staticmethod
    def _process_fwd_input(fwd):
        """ Process MNE Forward input to constructor """
        # Ensure fixed normal direction is used for lead field gain matrix
        if fwd['sol']['data'].shape == fwd['_orig_sol'].shape:
            fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)

        lfm = fwd['sol']['data']
        src = fwd['src']

        return lfm, src

    @staticmethod
    def _process_src_scalars(**kwargs):
        """
        Process the scalars used to construct adjacency
        matrix for state noise covariance matrix and to
        construct mixing matrix for transition matrix
        """
        d1 = kwargs['d1'] if 'd1' in kwargs else None
        d2 = kwargs['d2'] if 'd2' in kwargs else None
        m1 = kwargs['m1'] if 'm1' in kwargs else None
        m2 = kwargs['m2'] if 'm2' in kwargs else None
        scale = kwargs['scale'] if 'scale' in kwargs else None
        approach = kwargs['approach'] if 'approach' in kwargs else None
        orthonormal = kwargs['orthonormal'] if 'orthonormal' in kwargs else None
        return d1, d2, m1, m2, scale, approach, orthonormal

    @staticmethod
    def _vertex_source_mapping(src):
        """
        Create dictionaries to map between vertex numbering contained
        as 'vertno' attribute of MNE SourceSpaces and numbering of
        sources in the forward model lead field gain matrix, i.e.,
        the numbering of columns in lfm
        """
        assert len(src) == 2, 'Expecting two hemispheres contained in SourceSpaces instance.'
        vert_to_source, source_to_vert = ([{}] * len(src), [{}] * len(src))

        # Left (hemi=0) and right (hemi=1) hemispheres
        for hemi in range(len(src)):
            nsource = src[hemi]['nuse']
            source_indices = (np.asarray(range(nsource)) + src[0]['nuse']).tolist() if hemi == 1 else range(nsource)
            vertex_indices = src[hemi]['vertno']
            vert_to_source[hemi] = dict(zip(vertex_indices, source_indices))
            source_to_vert[hemi] = dict(zip(source_indices, vertex_indices))

        return vert_to_source, source_to_vert

    @staticmethod
    def _define_neighbors(src):
        """
        Define the indices of first and second order neighbors of all
        vertices in the source triangulation as in the 'use_tris'
        attribute of MNE SourceSpaces
        """
        assert len(src) == 2, 'Expecting two hemispheres contained in SourceSpaces instance.'
        neighbors = [{}] * len(src)

        # Left (hemi=0) and right (hemi=1) hemispheres
        for hemi in range(len(src)):
            hemi_neighbors: dict = {}

            # Define first-order neighbors
            for tri in src[hemi]['use_tris']:
                for vert in tri:
                    # Set default key-value pairs if not already present
                    hemi_neighbors.setdefault(vert, {})
                    hemi_neighbors[vert].setdefault('first', [])

                    # Add the other two vertices in the triangulation to first-order neighbors
                    first_list = hemi_neighbors[vert]['first']
                    vert_neighbors = tri.copy().tolist()
                    vert_neighbors.remove(vert)
                    hemi_neighbors[vert]['first'] = np.unique(first_list + vert_neighbors).tolist()

            # Define second-order neighbors
            for vert in hemi_neighbors:
                first_list = hemi_neighbors[vert]['first']
                unique_second = np.unique(np.hstack([hemi_neighbors[x]['first'] for x in first_list]))
                hemi_neighbors[vert]['second'] = np.setdiff1d(unique_second, first_list + [vert]).tolist()

            neighbors[hemi] = hemi_neighbors

        return neighbors

    @staticmethod
    def _get_q_basis(B, orthonormal=None):
        """
        Decompose the adjacency matrix into an orthonormal
        basis for noise covariance, or use the matrix square
        root of the adjacency matrix as a non-orthonormal kernel
        """
        orthonormal = orthonormal or True
        if orthonormal:
            QR = qr(np.float64(B), pivoting=False)
            return QR[0]  # orthonormal basis, N.B. not eigen-basis!
        else:
            B_sqrt = sqrtm(B)
            return [B_sqrt, inverse(B_sqrt)]  # non-orthonormal kernel

    @staticmethod
    def update_theta(Q_ss=None, T=None, Q_basis=None, nsource=None, npart=None, priors=None):
        """ Update the expansion coefficients for the state noise covariance matrix Q """
        Theta = torch.zeros(nsource, dtype=Q_ss.dtype).cuda()
        for vidx in range(nsource):
            theta_ss = Q_basis[:, vidx] @ Q_ss @ Q_basis[:, vidx]

            # noinspection PyProtectedMember
            if StateSpaceModel._m_update_if_mle(['theta_ig', 'theta_l0', 'theta_l1', 'theta_jeff'], priors):
                # MLE
                Theta[vidx] = theta_ss / (npart * T)
            elif 'theta_ig' in priors:
                # inverse gamma prior
                alpha = priors['theta_ig']['alpha']
                beta = priors['theta_ig']['beta']
                Theta[vidx] = (beta + (theta_ss / npart) / 2) / (alpha + T / 2 + 1)
            elif 'theta_l0' in priors:
                gamma = priors['theta_l0']['gamma']
                d = theta_ss * gamma / npart - T
                Theta[vidx] = (d / 2 + torch.sqrt(
                    d ** 2 / 4 + 2 * gamma * (2 / (npart ** 2) + (T / npart) / 2) * theta_ss)
                               ) / (gamma * (4 / npart + T))
            elif 'theta_l1' in priors:
                # exponential prior
                gamma = priors['theta_l1']['gamma']
                Theta[vidx] = (torch.sqrt(T ** 2 + 8 * theta_ss * gamma / (npart ** 2)) - T) / (4 * gamma / npart)
            elif 'theta_jeff' in priors:
                # Jeffreys prior
                Theta[vidx] = theta_ss / (npart * T + 2)
            else:
                raise RuntimeError('Could not set m_update rule for theta. Please check.')

        # assert np.all(Theta > 0), 'Theta coefficients should be non-negative.'
        # Due to convergent Kalman, Theta could be negative. We lower-bound it to 0
        Theta[Theta < 0] = 0

        return Theta

    @staticmethod
    def _m_step_scope(update_param, keep_param):
        """ Sort out the scopes of updates in m_step() """
        # noinspection PyProtectedMember
        update_FQ, update_mu0S0 = StateSpaceModel._m_estimate_scope(update_param, keep_param)
        assert 'G' not in update_param, 'Observation matrix should not be updated during source localization.'
        return update_FQ, update_mu0S0
