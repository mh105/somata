"""
Author: Mingjian He <mh1@stanford.edu>

Testing function for loading data in tests/test_*.py
"""

import os
from importlib.util import find_spec
from scipy.io.matlab import loadmat


def _load_data(filename, return_path=False):
    """ Load data files for test functions """
    somata_init_path = find_spec('somata').origin  # find the package root __init__.py filepath
    test_data_dir = os.path.join(os.path.dirname(os.path.dirname(somata_init_path)), 'tests/_test_data')
    full_fn = os.path.join(test_data_dir, filename)
    if return_path:
        return full_fn
    else:
        data_dict = loadmat(full_fn)
        for elem in ('__header__', '__version__', '__globals__'):
            data_dict.pop(elem)  # pop the extra system info key-value pairs
        return data_dict


def test_load_data():
    _load_data("kalman_inputs.mat")


if __name__ == "__main__":
    test_load_data()
    print('Test data loading finished without exception.')
