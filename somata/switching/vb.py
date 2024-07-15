"""
Author: Mingjian He <mh1@stanford.edu>

vb module contains variational bayesian methods for switching inference
"""

from somata.basic_models import StateSpaceModel as Ssm
from somata.exact_inference import forward_backward, viterbi, run_em, logdet, inverse
from somata.switching import switching
import numpy as np
from codetiming import Timer


class VBSwitchModel(object):
    """
    VBSwitchModel is an object class created to help
    Variational Bayesian Switching Models inference
    """
    ssm_array = None
    K = None
    T = None
    A = None
    h_t_m = None
    q_t_m = None
    logL_bound = None
    x_t_n_all = None
    P_t_n_all = None

    def __init__(self, ssm_array, y=None):
        """
        Constructor method for VBSwitchModel class
        :param ssm_array: an array of Ssm class objects or a single object with stacked models
        :param y: observed data
        """
        self.ssm_array, self.K, self.T = Ssm.setup_array(ssm_array, y=y)

    def __repr__(self):
        """ Unambiguous and concise representation when calling VBSwitchModel() """
        return 'Vbs(' + str(len(self.ssm_array)) + ')<' + hex(id(self))[-4:] + '>'

    def __str__(self):
        """ Helpful information when calling print(VBSwitchModel()) """
        print_str = "Vbs model " + str(self.ssm_array)
        return print_str

    def learn(self, y=None, dwell_prob=0.99, A=None, keep_param=(),
              maxE_iter=100, maxVB_iter=100, h_t_thresh=1e-6, q_t_thresh=1e-6,
              shared_R=False, shared_comp=False, priors_all='auto', normalize_q_t=False,
              plot_E=False, verbose=True, return_dict=None, original=False, show_pbar=True):
        """
        Variational Bayesian learning of segmental state-space
        models with Markov switching

        This is an implementation of Ghahramani & Hinton 2000
        learning algorithm with variational bayesian approximation.
        This algorithm has a generalized EM structure where the E
        step will lower-bound the posterior distribution of the
        hidden variables (both continuous hidden state and discrete
        switching variables), and the M step will re-estimate the
        state-space model (linear dynamical system) and hidden Markov
        model parameters using Shumway & Stoffer 1982 EM on the SSM
        and Baum-Welch algorithm on the HMM.

        Figure 5: Learning algorithm for switching state-space models
        [E step] - Repeat until convergence of KL(Q||P):
                E.1 Compute q_t_m from the prediction error of
                    state-space model m on observation Y_t
                E.2 Compute h_t_m using the forward-backward algorithm
                    on the HMM, with observation probabilities q_t_m
                E.3 For m = 1 to M (we use K instead of M)
                    Run Kalman smoothing recursions, using the data
                    weighted by h_t_m

        [M step]
                M.1 Re-estimate parameters for each state-space model
                    using the data weighted by h_t_m
                M.2 Re-estimate parameters for the switching process
                    using Baum-Welch update equations.

        Reference:
            He, M., Das, P., Hotan, G., & Purdon, P. L. (2023). Switching
            state-space modeling of neural signal dynamics. PLoS
            Computational Biology, 19(8), e1011395.

            Ghahramani, Z., & Hinton, G. E. (2000). Variational learning
            for switching state-space models. Neural computation, 12(4),
            831-864.

        If original is False, we use a new implementation of the original
        G&H2000 learning algorithm with variational bayesian approximation.
        This new implementation introduces various improvements across
        the VB learning iterations. These features include:
            - smart initialization of E-step with interpolated density
              and avoid deterministic annealing
            - semi-hard (converged h_t), soft (smoothed estimates of
              discrete HMM using interpolated density), and hard
              (Viterbi) segmentations are available
            - parallelized Kalman filtering and smoothing in De Jong
              version that avoids inverting conditional state noise
              covariance P_t_tmin1 and handles degenerate matrices
              automatically
            - parameters for shared components can be jointly estimated
            - MAP estimation instead of ML to incorporate priors

        Inputs:
        :param y: observed data
        :param dwell_prob: dwell probability for HMM transitions to initialize A
        :param A: HMM transition matrix, if provided will override dwell_prob
        :param keep_param: a tuple of strings for parameters to keep and not update
        :param maxE_iter: maximal number of iterations to run for fixed-point iteration in E step
        :param maxVB_iter: maximal number of EM iterations in VB learning
        :param h_t_thresh: threshold below which EM iteration stops
        :param q_t_thresh: threshold below which E step fixed-point iteration stops
        :param shared_R: boolean flag to do joint estimation of observation noise
        :param shared_comp: False to do independent updates, or a tuple of lists of indices for which component
                            in each model to estimate jointly. E.g. ([1,1], [2,3]) will force the 1st component
                            of model 1 to be linked to the 1st component of model 2, and the 2nd component of
                            model 1 to be linked to the 3rd component of model 2. Lists inside the tuple must
                            have the same length as the number of models in ssm_array
        :param priors_all: a list of lists of dictionaries specifying priors for each component,
                           if 'auto' -> Auto MAP, if None -> MLE
        :param normalize_q_t: whether to normalize interpolated density at the first E-step iteration
        :param plot_E: boolean flag to plot during fixed-point iteration in E step
        :param verbose: boolean flag for printing messages of changes in h_t_m and time taken
        :param return_dict: None -> no return, True -> return dict, False -> return tuple of variables
        :param original: boolean flag to use the original Ghahramani & Hinton algorithm
        :param show_pbar: show progress bar during EM iterations

        Outputs:
        :returns: a dictionary or tuple of variables depending on return_dict
            - h_t_m: a 2D array of model responsibilities for each model at each time point
            - h_t_m_soft: a 2D array of soft segmentations for each model at each time point
            - h_t_m_hard: a 2D array of hard segmentations for each model at each time point
            - q_t_m: a 2D array of emission probability densities for each model at each time point
            - A: HMM transition matrix
            - ssm_array: an array of Ssm class objects
            - VB_iter: number of EM iterations
            - logL_bound: a list of lower bounds on log likelihood
            - x_t_n_all: a list of smoothed estimates (posterior) of state mean
            - P_t_n_all: a list of smoothed estimates (posterior) of state conditional covariance
        """
        t = Timer(name="time taken", text="{name}: {seconds:.2f} seconds")
        t.start()

        """ Initialization """
        self.A = abs(np.eye(self.K, dtype=np.float64) - np.ones((self.K, self.K), dtype=np.float64)
                     * (1-dwell_prob) / (self.K-1)) if A is None else A
        self.logL_bound = []
        if original:
            shared_R = True
            shared_comp = False
            priors_all = None
            self.h_t_m = np.ones((self.K, self.T), dtype=np.float64) / self.K  # initialize with equal responsibilities
        else:
            self.h_t_m = float('inf')
        # Initialize priors for SSM if 'auto' is specified
        priors_all = [x.initialize_priors() for x in self.ssm_array] if priors_all == 'auto' else priors_all

        e_kwargs = {'original': original, 'maxE_iter': maxE_iter, 'q_t_thresh': q_t_thresh,
                    'normalize_q_t': normalize_q_t, 'plot_E': plot_E}
        m_kwargs = {'priors_all': priors_all, 'keep_param': keep_param,
                    'shared_R': shared_R, 'shared_comp': shared_comp}

        """ EM iterations """
        VB_iter, _ = run_em(self, y=y, e_kwargs=e_kwargs, m_kwargs=m_kwargs,
                            max_iter=maxVB_iter, stop_thresh=h_t_thresh, ignore_numerr=True, show_pbar=show_pbar)

        """ Switching linear segmentation """
        # Provide different segmentation methods (soft, semi-hard (VB h_t_m), hard)
        h_t_m_soft, fy_t = switching(self.ssm_array, method='ab pari', A=self.A)  # soft segmentation
        _, h_t_m_hard = viterbi(self.A, fy_t, ignore_numerr=True)  # apply Viterbi on parallel interpolated density

        """ EOF """
        if verbose:
            print('VBSwitchModel.learn()')
            t.stop()
            print('iterations:', str(VB_iter))

        if return_dict is None:
            pass
        elif return_dict:
            return {'h_t_m': self.h_t_m, 'h_t_m_soft': h_t_m_soft, 'h_t_m_hard': h_t_m_hard, 'q_t_m': self.q_t_m,
                    'A': self.A, 'ssm_array': self.ssm_array, 'VB_iter': VB_iter, 'logL_bound': self.logL_bound,
                    'x_t_n_all': self.x_t_n_all, 'P_t_n_all': self.P_t_n_all}
        else:
            return self.h_t_m, h_t_m_soft, h_t_m_hard, self.q_t_m, \
                self.A, self.ssm_array, VB_iter, self.logL_bound, self.x_t_n_all, self.P_t_n_all

    def e_step(self, y=None, original=False, maxE_iter=100, q_t_thresh=1e-6, normalize_q_t=False, plot_E=False):
        """
        Generalized E step in VB learning, exposed for run_em()

        Depending on whether the original algorithm is used, the E step
        has a different initialization of the fixed-point iteration
        """
        # Save the current h_t_m to update EM stopping variable at the end
        last_h_t_m = self.h_t_m

        # Initialize the fixed-point iteration
        delta_q_t_m = float('inf')
        self.q_t_m = float('inf')
        E_iter = 0
        E_logL_bound = []

        # Different initialization based on whether the original algorithm is used
        if original:
            method = 'kalman'
            temperature = 100  # initial temperature parameter for deterministic annealing
            # Initialize E step using converged h_t_m from the last EM iteration as a warm start
            x_t_n_all, P_t_n_all, *_ = Ssm.par_kalman(self.ssm_array, y=y, method=method, R_weights=1/self.h_t_m)
        else:
            method = 'dejong'
            temperature = 1  # no deterministic annealing

        # minimize KL(Q||P)
        while delta_q_t_m > q_t_thresh and E_iter < maxE_iter:
            # Store q_t_m to check for convergence of E step
            E_iter += 1
            last_q_t_m = self.q_t_m

            # E.1 Compute q_t_m using smoothed estimates
            if E_iter == 1 and not original:
                # Initialize q_t_m in E.1 using interpolated density
                _, P_t_n_all, _, _, _, _, _, _, _, fy_t_interp_all = Ssm.par_kalman(self.ssm_array, y=y,
                                                                                    method=method, skip_interp=False)
                self.q_t_m = np.vstack(fy_t_interp_all)

                # Normalize interpolated density to better initialize q_t_m
                if normalize_q_t:
                    # Scale q_t_m such that the sum of interpolated densities are equal across models
                    sum_density_weights = self.q_t_m.sum(axis=1)
                    self.q_t_m = self.q_t_m * (min(sum_density_weights) / sum_density_weights)[:, None]

                    # Scale q_t_m by the effective smoothed conditional covariance
                    P_t_n_weights = np.zeros(self.K, dtype=np.float64)
                    for m in range(self.K):
                        P_t_n_weights[m] = np.squeeze(
                            self.ssm_array[m].G @ P_t_n_all[m][:, :, self.T//2] @ self.ssm_array[m].G.T)
                    self.q_t_m = self.q_t_m * (min(P_t_n_weights) / P_t_n_weights)[:, None]

                g_t_m = np.log(self.q_t_m)
            else:
                # noinspection PyUnboundLocalVariable
                self.q_t_m, g_t_m = _compute_q_t_m(self.ssm_array, x_t_n_all, P_t_n_all, temperature=temperature)

            # E.2 Compute h_t_m using forward-backward algorithm
            _, self.h_t_m, h_t_tmin1_m, logL_HMM = forward_backward(A=self.A, py_x=self.q_t_m, compute_edge=True)

            # Modify h_t_m and h_t_tmin1_m with deterministic annealing
            self.h_t_m = self.h_t_m / temperature
            h_t_tmin1_m = h_t_tmin1_m / temperature
            temperature = temperature / 2 + 0.5  # decreasing towards asymptote at temperature=1

            # E.3 Run Kalman smoothing recursions with weighted data
            x_t_n_all, P_t_n_all, P_t_tmin1_n_all, logL_all, *_ = \
                Ssm.par_kalman(self.ssm_array, y=y, method=method, R_weights=1/self.h_t_m)
            logL_SSM = np.vstack(logL_all)

            # Check for convergence using the average change in elements of q_t_m
            delta_q_t_m = np.mean(abs(last_q_t_m - self.q_t_m))

            # Compute the negative variational free energy / lower bound on likelihood / evidence lower bound (ELBO)
            E_logL_bound.append(_compute_logl_bound(self.ssm_array, logL_HMM, logL_SSM, self.h_t_m, g_t_m))

            # Plotting E step for the first two models
            _ = _plot_e(self.h_t_m, self.q_t_m, E_iter) if plot_E else None

        # Finish the E step
        self.logL_bound.append(E_logL_bound)
        # noinspection PyUnboundLocalVariable
        self.x_t_n_all = x_t_n_all
        # noinspection PyUnboundLocalVariable
        self.P_t_n_all = P_t_n_all

        # noinspection PyUnboundLocalVariable
        e_results = {'x_t_n_all': x_t_n_all, 'P_t_n_all': P_t_n_all, 'P_t_tmin1_n_all': P_t_tmin1_n_all,
                     'h_t_tmin1_m': h_t_tmin1_m}
        stop_var = np.mean(abs(last_h_t_m - self.h_t_m))

        return e_results, stop_var

    def m_step(self, y=None, x_t_n_all=None, P_t_n_all=None, P_t_tmin1_n_all=None, h_t_tmin1_m=None,
               priors_all=None, keep_param=(), shared_R=False, shared_comp=False):
        """ Generalized M step in VB learning, exposed for run_em() """
        # M.1 ML estimates of SSM parameters with weighted data
        _m1(self.ssm_array, x_t_n_all, P_t_n_all, P_t_tmin1_n_all, self.h_t_m, y=y,
            priors_all=priors_all, keep_param=keep_param, shared_R=shared_R, shared_comp=shared_comp)

        # M.2 ML estimate of transition matrix of HMM
        self.A, _ = _m2(self.h_t_m, h_t_tmin1_m)


def _compute_q_t_m(ssm_array, x_t_n_all, P_t_n_all, temperature=1, y=None):
    """
    Compute the un-normalized Gaussian density function
    of expected error for VB learning fixed-point iteration

    Reference:
        Ghahramani, Z., & Hinton, G. E. (2000). Variational learning
        for switching state-space models. Neural computation, 12(4),
        831-864.
    """
    # Array dimensions
    y = ssm_array[0].y if y is None else y
    K = len(ssm_array)  # number of models
    q, T = y.shape  # number of channels and number of time points
    approach = 'svd' if q >= 5 else 'gaussian'
    g_t_m = np.zeros((K, T), dtype=np.float64)

    for m in range(K):
        x_t_n = x_t_n_all[m][:, 1:]  # (index 1 corresponds to t=0, etc.)
        P_t_n = P_t_n_all[m][:, :, 1:]
        for ii in range(T):  # t=1 -> t=T
            g_t_m[m, ii] = 1/temperature * (
                    -0.5 * y[:, ii].T @ inverse(ssm_array[m].R, approach=approach) @ y[:, ii]
                    + y[:, ii].T @ inverse(ssm_array[m].R, approach=approach) @ ssm_array[m].G @ x_t_n[:, ii]
                    + -0.5 * np.trace(ssm_array[m].G.T @ inverse(ssm_array[m].R, approach=approach) @ ssm_array[m].G
                                      @ (P_t_n[:, :, ii] + x_t_n[:, ii][:, None] @ x_t_n[:, ii][:, None].T)))

    q_t_m = np.exp(g_t_m)  # Equation 4.13
    return q_t_m, g_t_m


def _compute_logl_bound(ssm_array, logL_HMM, logL_SSM, h_t_m, g_t_m):
    """
    Compute the bound on the marginal log likelihood of observed data.
    Also known as Evidence Lower BOund (ELBO) or negative variational
    free energy.
    """
    # Precompute the log determinant of R
    R_logdet = np.asarray([logdet(2 * np.pi * x.R) for x in ssm_array])

    # First term: determinant of R
    F1 = -0.5 * R_logdet @ h_t_m.sum(axis=1)

    # Second term: marginal log likelihood from HMM
    F2 = logL_HMM.sum()

    # Third term: -ht*gt
    F3 = -(h_t_m * g_t_m).sum()

    # Fourth term: marginal log likelihood from SSM combined with log(ht) to deal with infinity values
    F4_tmp = logL_SSM + 0.5 * (R_logdet[None, :].T - ssm_array[0].nchannel * np.log(h_t_m))
    F4_tmp[np.isinf(F4_tmp)] = float('nan')
    F4 = np.nansum(F4_tmp)

    logL_bound = F1 + F2 + F3 + F4
    return logL_bound


def _m1(ssm_array, x_t_n_all, P_t_n_all, P_t_tmin1_n_all, h_t_m, y=None, priors_all=None,
        keep_param=(), shared_R=False, shared_comp=False):
    """
    ML/MAP estimation of SSM parameters for the M.1 step.
    This updates SSM parameters with weighted data, and
    executes the M.1 step in VB learning of switching
    linear segments.

    Inputs:
    :param ssm_array: an array of Ssm class objects
    :param y: observed data
    :param x_t_n_all: a list of smoothed estimates (posterior) of state mean
    :param P_t_n_all: a list of smoothed estimates (posterior) of state conditional covariance
    :param P_t_tmin1_n_all: a list of smoothed estimates (posterior) of state lag1 conditional cross-covariance
    :param h_t_m: a 2D array of responsibility vectors
    :param priors_all: a list of lists of dictionaries specifying priors for each component, if None -> MLE
    :param keep_param: a tuple of strings for parameters to keep and not update
    :param shared_R: boolean flag to do joint estimation of observation noise
    :param shared_comp: False to do independent updates, or a tuple of lists of indices for which component
                        in each model to estimate jointly. E.g. ([1,1], [2,3]) will force the 1st component
                        of model 1 to be linked to the 1st component of model 2, and the 2nd component of
                        model 1 to be linked to the 3rd component of model 2. Lists inside the tuple must
                        have the same length as the number of models in ssm_array
    """
    y = ssm_array[0].y if y is None else y
    K = len(ssm_array)  # number of models
    T = y.shape[1]  # number of time points
    priors_all = [[None] * x.ncomp for x in ssm_array] if priors_all is None else priors_all

    """ Estimate model parameters individually for each model """
    R_ss = np.zeros((y.shape[0], y.shape[0], K), dtype=np.float64)
    A, B, C = ([], [], [])
    for m in range(K):
        m_results = ssm_array[m].m_estimate(y=y, x_t_n=x_t_n_all[m], P_t_n=P_t_n_all[m], P_t_tmin1_n=P_t_tmin1_n_all[m],
                                            h_t=h_t_m[m, :], priors=priors_all[m],
                                            keep_param=keep_param, return_dict=True)
        # Store sum variables for joint estimations
        R_ss[:, :, m] = m_results['R_ss']
        A.append(m_results['A'])
        B.append(m_results['B'])
        C.append(m_results['C'])

    """ Joint estimation of parameters """
    # Update observation noise R shared across models
    if shared_R and 'R' not in keep_param:
        R_ss = R_ss.sum(axis=2)
        # noinspection PyProtectedMember
        _ = [x._m_update_r(R_ss=R_ss, T=T, priors=priors_all[0][0]) for x in ssm_array]

    # Update F and Q matrices shared across models
    if shared_comp and not ('F' in keep_param and 'Q' in keep_param):
        shared_comp = (shared_comp,) if not isinstance(shared_comp, tuple) else shared_comp
        assert all([k == K for k in [len(x) for x in shared_comp]]), \
            'Index of component shared is not specified for all models. Use index=None if a model does not share.'

        for j in range(len(shared_comp)):  # iterate through each independent component that is shared
            # Obtain the component indices for a shared component in each model
            shared_comp_indices: list = shared_comp[j]  # type: ignore

            # Identify which models are joining
            joint_models = [x is not None for x in shared_comp_indices]

            # Total T is multiplied by the number of models sharing the components
            shared_T = T * sum(joint_models)

            # Get the state dimension of the current component
            first_model_index = np.argmax(joint_models)
            nstate = ssm_array[first_model_index].comp_nstates[shared_comp_indices[first_model_index]]

            # Create temporary trellis variables
            update_index_store = [(None, None)] * K
            A_tmp = np.zeros((nstate, nstate, K), dtype=np.float64)
            B_tmp = np.zeros((nstate, nstate, K), dtype=np.float64)
            C_tmp = np.zeros((nstate, nstate, K), dtype=np.float64)

            # Accumulate the sums of squares for joint estimation
            for m in range(K):
                comp_index = shared_comp_indices[m]
                if comp_index is not None:
                    # Identify the state indices for the component
                    start_idx = sum(ssm_array[m].comp_nstates[:comp_index])
                    end_idx = sum(ssm_array[m].comp_nstates[:comp_index + 1])
                    update_index_store[m] = (start_idx, end_idx)

                    # Accumulate sums of squares for the shared component
                    A_tmp[:, :, m] = A[m][start_idx:end_idx, start_idx:end_idx]
                    B_tmp[:, :, m] = B[m][start_idx:end_idx, start_idx:end_idx]
                    C_tmp[:, :, m] = C[m][start_idx:end_idx, start_idx:end_idx]

            A_tmp = A_tmp.sum(axis=2)
            B_tmp = B_tmp.sum(axis=2)
            C_tmp = C_tmp.sum(axis=2)

            # Assume all components being shared have the same priors specified
            comp_priors = priors_all[first_model_index][shared_comp_indices[first_model_index]]

            # Call component specific _m_update_<param> methods
            comp = ssm_array[first_model_index].components[shared_comp_indices[first_model_index]]
            if 'F' not in keep_param:
                # noinspection PyProtectedMember
                F = comp._m_update_f(A=A_tmp, B=B_tmp, priors=comp_priors)
            else:
                # assume that all models already have the same F entries for the shared component
                start_idx, end_idx = update_index_store[first_model_index]
                F = ssm_array[first_model_index].F[start_idx:end_idx, start_idx:end_idx]

            if 'Q' not in keep_param:
                # noinspection PyProtectedMember
                Q = comp._m_update_q(A=A_tmp, B=B_tmp, C=C_tmp, T=shared_T, F=F, priors=comp_priors)

            # Distribute to all models sharing the component
            for m in range(K):
                comp_index = shared_comp_indices[m]
                if comp_index is not None:
                    start_idx, end_idx = update_index_store[m]
                    if 'F' not in keep_param:
                        ssm_array[m].F[start_idx:end_idx, start_idx:end_idx] = F
                    if 'Q' not in keep_param:
                        # noinspection PyUnboundLocalVariable
                        ssm_array[m].Q[start_idx:end_idx, start_idx:end_idx] = Q

        for m in range(K):  # make sure instance attributes reflect the updated F and Q
            ssm_array[m].update_comp_param()


def _m2(h_t_m, h_t_tmin1_m):
    """ ML estimation of HMM parameters for the M.2 step """
    # M step - MLE of parameters {A, p1}
    p1 = h_t_m[:, 0]  # updated p1
    edges = h_t_tmin1_m.sum(axis=2)  # sum over time points
    A = edges / edges.sum(axis=0)  # updated A
    return A, p1


def _plot_e(h_t_m, q_t_m, E_iter):
    """ Plot the h_t_m and q_t_m for two models in E step """
    import matplotlib.pyplot as plt

    plt.subplots_adjust(hspace=0.5)
    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1.clear()
    ax1.plot(h_t_m[0, :])
    ax1.plot(h_t_m[1, :])
    ax1.set_title('h_t_m: iter ' + str(E_iter))
    ax1.set_ylim([-0.1, 1.1])

    ax2 = plt.subplot(2, 1, 2)
    ax2.clear()
    ax2.plot(q_t_m[0, :])
    ax2.plot(q_t_m[1, :])
    ax2.set_title('q_t_m: iter ' + str(E_iter))

    plt.pause(0.05)
    return fig
