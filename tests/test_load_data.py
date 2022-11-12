"""
Author: Mingjian He <mh105@mit.edu>

Testing function for loading data in tests/test_*.py
"""

import os
from scipy.io.matlab import loadmat


def _load_data(filename, return_path=False):
    """ Load data files for dp functions """
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if dir_path.find('tests') > 0:
            test_data_dir = os.path.join(dir_path, '_test_data')
        else:
            test_data_dir = os.path.join(dir_path, 'tests/_test_data')
    except NameError:  # __file__ isn't available for iPython console sessions
        dir_path = os.getcwd()
        test_data_dir = os.path.join(dir_path[:dir_path.find('somata') + 6],
                                     'tests/_test_data')  # assumes the top level repo name is somata
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
