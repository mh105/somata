"""
Author: Mingjian He <mh105@mit.edu>

em module contains a general run_em() function that works across SOMATA
"""

import numpy as np
from tqdm import tqdm
from sorcery import dict_of


def run_em(obj: object, y=None, init_from_data=False, e_kwargs=None, m_kwargs=None,
           max_iter=10, stop_thresh=np.finfo(float).eps, ignore_numerr=False, return_dict=False, show_pbar=False):
    """
    run_em() is a general purpose function to organize EM algorithms used
    throughout SOMATA. The class object that is input as the first argument
    is expected to have e_step() and m_step() properly defined. Additional
    inputs to the E step and M step can be provided as dictionaries for
    e_kwargs and m_kwargs to make further specifications such as priors.
    This function is implemented here as a high-level organizer of EM
    iterations to bring together different signal processing algorithms
    that all depend on the generalized EM iteration structure, despite
    their specific E, M steps can be vastly different.

    Inputs:
    :param obj: a SOMATA-compatible class object, could be one of basic_models
                objects, or a module specific class object
    :param y: observed data
    :param init_from_data: boolean flag to call initialize_from_data() method on obj
    :param e_kwargs: arguments for exposed e_step() method
    :param m_kwargs: arguments for exposed m_step() method
    :param max_iter: maximal number of EM iterations to run
    :param stop_thresh: a stopping criterion threshold, flexible depending on the
                        stopping variable used by the e/m_step() methods of obj
    :param ignore_numerr: whether to ignore warning messages of numerical errors
    :param return_dict: None -> no return, True -> return dict, False -> return tuple of variables
    :param show_pbar: show progress bar during EM iterations
    """
    # Initialize from data
    if init_from_data:
        obj.initialize_from_data(y=y)

    # Initialize EM iterations
    em_iter = 0
    stop_var = float('inf')
    stop_var_tally = []

    if ignore_numerr:
        old_settings = np.seterr(divide='ignore', over='ignore', invalid='ignore')

    # Run EM to learn model parameters
    with tqdm(total=max_iter, desc='run_em()', colour='green', disable=not show_pbar) as pbar:
        while stop_var > stop_thresh and em_iter < max_iter:
            # E step
            e_results, stop_var = obj.e_step(y=y, **e_kwargs)
            stop_var_tally.append(stop_var)
            if type(stop_var) is bool and stop_var:
                break  # break EM while loop
            else:
                assert stop_var >= 0, 'Stopping variable should be non-negative. Current value = ' + str(stop_var) + '!'

            # M step
            obj.m_step(y=y, **e_results, **m_kwargs)

            em_iter += 1
            pbar.update(1)

    if ignore_numerr:
        # noinspection PyUnboundLocalVariable
        np.seterr(**old_settings)

    if return_dict is None:
        pass
    elif return_dict:
        return dict_of(em_iter, stop_var_tally)
    else:
        return em_iter, stop_var_tally
