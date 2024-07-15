"""
Author: Mingjian He <mh1@stanford.edu>

exact_inference module contains basic dynamic programming algorithms used in SOMATA
"""

from .dp_func import forward_backward, baum_welch, viterbi, logdet, logdet_torch, \
    kalman, djkalman, djkalman_conv_torch, inverse, inverse_torch  # noqa: F401
from .em import run_em  # noqa: F401
