"""
Authors: Mingjian He <mh1@stanford.edu>
         Proloy Das <pd640@mgh.harvard.edu>

source_loc_utils module implements supporting functions for dynamic source localization
"""

import torch
import mne
import numpy as np
from somata.exact_inference import djkalman_conv_torch


def patch_decompose(fwd, src_patch, dist_metric='geodesic', normalize=False):
    """
    Patch decomposition for constructing the forward model of a
    source space with reduced dimensionality.

    Reference:
        Limpiti, T., Van Veen, B. D., & Wakai, R. T. (2006). Cortical patch
        basis model for spatially extended neural activity. IEEE Transactions
        on Biomedical Engineering, 53(9), 1740-1754.

        Babadi, B., Obregon-Henao, G., Lamus, C., Hämäläinen, M. S.,
        Brown, E. N., & Purdon, P. L. (2014). A subspace pursuit-based
        iterative greedy hierarchical solution to the neuromagnetic inverse
        problem. NeuroImage, 87, 427-443.

    Inputs:
    :param fwd: MNE Forward instance with source information
    :param src_patch: MNE SourceSpaces instance with reduced dimensionality
                      - Note: the recommended way to construct src_patch is
                      to use the same SourceSpaces construction pipeline in
                      creating fwd and use a smaller number of sources. If
                      the surfaces in fwd are simply decimated, there is no
                      guarantee that the smaller source space is a subset
                      of the one with higher dimension
    :param dist_metric: the distance metric to define Voronoi cell.
                        Options: 'geodesic' or 'Euclidean'
    :param normalize: whether to normalize the lead field matrix to have unit norms
    """
    # Make sure both SourceSpaces instances have the same coordinate frame
    src = fwd['src']
    assert src_patch[0]['coord_frame'] == fwd['coord_frame'] == src[0]['coord_frame'], \
        'Different coordinate frames between SourceSpaces.'

    # Ensure fixed normal direction is used for lead field gain matrix
    if fwd['sol']['data'].shape == fwd['_orig_sol'].shape:
        fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)

    # Compute the pairwise-distance between all sources in use
    if dist_metric == 'geodesic' and src[0]['dist'] is None:
        mne.add_source_space_distances(src)

    # Process one hemisphere at a time
    lfm_patch_hemi = [np.array([None])] * len(src_patch)
    sval_hemi = [np.array([None])] * len(src_patch)
    nmra_hemi = [np.array([None])] * len(src_patch)
    for hemi in range(len(src_patch)):
        # Since the sources included in a patch are selected based on the Voronoi Cell
        # on the cortical surface of the source space with higher dimension, each patch
        # model source must also be a source in the source space with higher dimension
        assert src_patch[hemi]['np'] == src[hemi]['np'], 'Mismatch in the number of vertices.'
        assert src_patch[hemi]['ntri'] == src[hemi]['ntri'], 'Mismatch in the number of faces.'
        assert np.all(src[hemi]['inuse'][src_patch[hemi]['vertno']] == 1), 'Patch is not a subset of sources.'

        # Find the Voronoi cell of patch sources
        if dist_metric == 'geodesic':
            # using geodesic distances over cortical surface
            dist = src[hemi]['dist'][np.tile(src_patch[hemi]['vertno'], reps=(len(src[hemi]['vertno']), 1)).T,
                                     np.tile(src[hemi]['vertno'], reps=(len(src_patch[hemi]['vertno']), 1))].toarray()
            min_indices = np.argmin(dist, axis=0)  # Voronoi cell source indices
        else:
            # using Euclidean distances over 3D space
            min_indices = np.zeros(len(src[hemi]['vertno']), dtype=np.int32)
            for jj in range(len(min_indices)):
                vec_diff = np.linalg.norm(src[hemi]['rr'][src[hemi]['vertno'][jj], :]
                                          - src_patch[hemi]['rr'][src_patch[hemi]['vertno'], :], axis=1)
                min_indices[jj] = np.argmin(vec_diff)

        # Initialize the patch basis forward model lead field matrix
        lfm_hemi = fwd['sol']['data'][:, :src[0]['nuse']] if hemi == 0 else fwd['sol']['data'][:, src[0]['nuse']:]
        lfm_patch_hemi[hemi] = np.zeros((lfm_hemi.shape[0], src_patch[hemi]['nuse']))
        nmra_hemi[hemi] = np.zeros(lfm_patch_hemi[hemi].shape[1])  # normalized mean representation accuracy

        # Loop through each patch
        for ii in range(lfm_patch_hemi[hemi].shape[1]):
            # SVD of the patch forward model gain matrix
            U, s, *_ = np.linalg.svd(lfm_hemi[:, min_indices == ii])
            if normalize:
                lfm_patch_hemi[hemi][:, ii] = U[:, 0]  # first left singular vector
            else:
                lfm_patch_hemi[hemi][:, ii] = U[:, 0] * s[0]  # multiplied by first singular value

            sval_hemi[hemi][ii] = s[0]  # store the first singular value
            nmra_hemi[hemi][ii] = s[0] / s.sum()  # first singular value over the sum of all singular values

    # Concatenate the lead field matrices from two hemispheres
    lfm_patch = np.hstack(lfm_patch_hemi)
    sval = np.hstack(sval_hemi)
    nmra = np.hstack(nmra_hemi)

    # Note that the patch basis model does not support free orientation,
    # therefore we modify the '_orig_sol' and '_orig_source_ori' attributes
    # in the output fwd_patch instance accordingly to support saving with
    # mne.write_forward_solution() and reading with mne.read_forward_solution()
    fwd_patch = fwd.copy()
    fwd_patch['source_ori'] = mne.io.constants.FIFF.FIFFV_MNE_FIXED_ORI
    fwd_patch['nsource'] = lfm_patch.shape[1]
    fwd_patch['sol']['ncol'] = lfm_patch.shape[1]
    fwd_patch['sol']['data'] = lfm_patch
    fwd_patch['_orig_sol'] = lfm_patch
    fwd_patch['src'] = src_patch.copy()
    fwd_patch['source_rr'] = np.vstack(
        [src_patch[hemi]['rr'][src_patch[hemi]['vertno'], :] for hemi in range(len(src_patch))])
    fwd_patch['_orig_source_ori'] = mne.io.constants.FIFF.FIFFV_MNE_FIXED_ORI
    fwd_patch['source_nn'] = np.vstack(
        [src_patch[hemi]['nn'][src_patch[hemi]['vertno'], :] for hemi in range(len(src_patch))])

    return fwd_patch, sval, nmra


def simulate_oscillation(center_frequency_Hz, damping_factor, state_noise_variance, initial_state_mean,
                         initial_state_variance, sampling_frequency, duration, oscillation_type='oscillator'):
    """ Simulate oscillation activity time series """
    from somata.basic_models import OscillatorModel as Osc

    time_step = 1 / sampling_frequency
    time_axis = np.arange(0, duration + time_step, time_step)
    n_time_steps = time_axis.shape[0]

    if oscillation_type == 'oscillator':
        transition_matrix = Osc.get_rot_mat(damping_factor, Osc.hz_to_rad(center_frequency_Hz, sampling_frequency))
        simulated = np.zeros((2, n_time_steps))
        simulated[:, 0] += np.matmul(transition_matrix,
                                     np.random.multivariate_normal(initial_state_mean,
                                                                   initial_state_variance * np.eye(2, 2))
                                     ) + np.random.multivariate_normal([0, 0], state_noise_variance * np.eye(2, 2))
        for ii in range(n_time_steps - 1):
            simulated[:, ii + 1] = np.matmul(transition_matrix, simulated[:, ii]) + \
                                   np.random.multivariate_normal([0, 0], state_noise_variance * np.eye(2, 2))
        simulated = simulated[1, :]  # keep only the real component

    elif oscillation_type == 'sinusoid':
        # amplitude = 2 * np.pi * damping_factor * np.sqrt(state_noise_variance)  # this might not be correct
        # simulated = amplitude * np.sin(2 * np.pi * center_frequency_Hz * time_axis) + \
        #             np.random.normal(0, np.sqrt(state_noise_variance), n_time_steps)
        simulated = np.sin(2 * np.pi * center_frequency_Hz * time_axis)

    else:
        raise ValueError('Unrecognized oscillation_type value')

    return simulated


def resolution_matrix_metrics(K, src):
    """
    Compute the resolution matrix metrics:
        - Spatial dispersion (SD)
        - Dipole localization error (DLE)
        - Resolution index (RI)
    """
    from somata.source_loc import SourceLocModel as Src

    nsources = K.shape[0]

    # Compute the pairwise-distance between all triangulation vertices
    if src[0]['dist'] is None:
        mne.add_source_space_distances(src)

    # Create a distance matrix between sources
    D = np.zeros((nsources, nsources), dtype=np.float64)
    # noinspection PyProtectedMember
    _, source_to_vert = Src._vertex_source_mapping(src)

    for hemi in range(len(src)):
        for vidx in source_to_vert[hemi]:
            vert = source_to_vert[hemi][vidx]  # vertex indexing

            for vidx2 in source_to_vert[hemi]:
                vert2 = source_to_vert[hemi][vidx2]  # vertex indexing

                D[vidx, vidx2] = src[hemi]['dist'][vert, vert2] * 100  # get into cm

    # Calculate SD, DLE, RI
    SD = np.zeros(nsources, dtype=np.float64)
    DLE = np.zeros(nsources, dtype=np.float64)
    RI = np.zeros(nsources, dtype=np.float64)
    max_dist = np.max(D)

    for hemi in range(len(src)):
        for vidx in source_to_vert[hemi]:

            vidx2_list = np.asarray(list(source_to_vert[hemi].keys()))

            numerator = ((D[vidx, vidx2_list] * K[vidx2_list, vidx]) ** 2).sum()
            denominator = (K[vidx2_list, vidx] ** 2).sum()
            SD[vidx] = np.sqrt(numerator / denominator)

            max_r_idx = np.argmax(abs(K[vidx2_list, vidx]))
            DLE[vidx] = D[vidx2_list[max_r_idx], vidx]

            max_c_idx = np.argmax(abs(K[vidx, vidx2_list]))
            RI[vidx] = ((max_dist - D[vidx, vidx2_list[max_c_idx]]) * abs(K[vidx, vidx])
                        ) / (max_dist * abs(K[vidx, vidx2_list[max_c_idx]]))

    return SD, DLE, RI


def smart_djkalman_conv_torch(F, Q, mu0, S0, G, R, y, rank, proj, conv_steps=100):
    """
    Wrapper of djkalman_conv_torch() to pre-whiten observed data

    `smart_djkalman_conv_torch()` handles average referencing by combining it with
    whitening. This also handles the rank deficiency that comes with average
    referencing. Following is a brief explanation of why and how:

    djkalman_conv_torch() uses the following state space model:
        x(t) = F*x(t-1)+eta(t)   eta ~ N(0,Q) (state or transition equation)
        y(t) = G*x(t)+eps(t)     eps ~ N(0,R) (observation or measurement equation)

    In EEG, such y(t) is not observable because electrical potentials can only be
    measured upto a baseline (i.e, with recording referencing). Consider average
    referencing is captured by a matrix H, the actual observed data is then:
        z(t) = H*y(t) = H*(G*x(t) + eps(t))

    smart_djkalman_conv_torch() adds a whitening matrix that includes the referencing
    matrix to the observation equation:
        W*y(t) = W*(G*x(t)+eps(t))      where W = M*pinv(H), and M*R*M.T = I
               = W*G*x(t)+lambda(t)     lambda(t) = W*eps(t) ~ N(0,C)

    Because H is rank deficient, C is also rank-deficient. However, C is only
    involved in djkalman_conv_torch() when computing the Kalman gain:
        K = F*P*G.T*W.T(W*G*P*G.T*W.T + C)^{-1}     where C = W*R*W.T

    We can apply the Woodbury matrix identity (matrix inversion lemma) twice in
    terms that involve the Kalman gain:
        K(W*y - F*x)    when updating the mean
        K*W*G           when updating the covariance

    and replace C with an identity matrix. It is ok to use an identity matrix to
    replace a singular identity matrix (some diagonal values being zero) because
    the whitener W will preserve the rank-deficiency in these terms.

    Input/Output:
    ------------
    rank: int
         estimated rank of *projected* data
    proj: Ny x Ny matrix
         projection matrix computed from `mne.Info` object (use accompanied `make_projector()`
         to compute the matrix)

    see the rest in function header of djkalman_conv_torch()
    """
    dtype = y.dtype

    W, logdet_by_n2 = get_whitener(R=R.cpu().numpy(), rank=rank, proj=proj)  # W = M * pinv(H)
    W = torch.as_tensor(W, dtype=dtype).cuda()
    logdet_by_n2 = torch.as_tensor(logdet_by_n2, dtype=dtype).cuda()

    G = torch.matmul(W, G)
    y = torch.matmul(W, y)
    R = torch.eye(R.shape[0], dtype=dtype).cuda()

    x_t_n, P_t_n, P_t_tmin1_n, logL, break_conv, K_t, x_t_tmin1, P_t_tmin1 = djkalman_conv_torch(
        F, Q, mu0, S0, G, R, y, conv_steps, dtype)

    # adjust log-likelihood by adding back the -1/2 logdet term
    qlog2pi_by_n2 = -y.shape[0] * torch.log(torch.as_tensor(2 * torch.pi, dtype=dtype).cuda()) / 2
    logL += y.shape[1] * (qlog2pi_by_n2 + logdet_by_n2)

    return x_t_n, P_t_n, P_t_tmin1_n, logL, break_conv, K_t, x_t_tmin1, P_t_tmin1


def get_whitener(R, rank, proj):
    """ Compute a whitening matrix based on the observation noise covariance """
    # Apply the projector on observation noise covariance matrix
    R = proj @ R @ proj.T

    # Eigen-decomposition for whitening
    eig, eigvec = mne.utils.linalg.eigh(R, overwrite_a=True)
    eig[:-rank] = 0.0
    eig_pos_idx = eig > 0
    eig[~eig_pos_idx] = 0.0

    # Some sanity checks on eigenvalues
    assert eig[-1] > 0, 'The largest eigenvalue is non-positive.'
    assert np.all(eig[eig_pos_idx] / eig[-1] > 1e-15), \
        'Some eigenvalues are very small. Whitener matrix may be inaccurate.'

    # Create the whitener matrix
    eig[eig_pos_idx] = 1.0 / np.sqrt(eig[eig_pos_idx])
    logdet_by_n2 = np.log(eig[eig_pos_idx]).sum()
    W = eigvec * eig @ eigvec.T

    return W, logdet_by_n2


def make_projector(info, projs=None, ch_names=None):
    """
    Create the projection operator

    Author: Proloy Das <pd640@mgh.harvard.edu>

    Input:
    -----
    info: mne.Info
    projs : list | None (default)
        List of projection vectors.
    ch_names: list of str | None (default)
        List of channels to include in the projection matrix.
    """
    from mne.io.pick import pick_info
    from mne.io.proj import make_projector

    projs = info['projs'] if projs is None else projs
    ch_names = info['ch_names'] if ch_names is None else ch_names
    if info['ch_names'] != ch_names:
        info = pick_info(info, [info['ch_names'].index(c) for c in ch_names])
    assert info['ch_names'] == ch_names

    proj, ncomp, _ = make_projector(projs, ch_names)
    return proj, ncomp
