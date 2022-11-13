"""
Author: Mingjian He <mh105@mit.edu>

exact_inference module contains basic dynamic programming algorithms used in SOMATA
"""

from .dp_func import forward_backward, baum_welch, viterbi, logdet, \
    kalman, djkalman, inverse
from .em import run_em
