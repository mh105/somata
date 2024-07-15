"""
Author: Mingjian He <mh1@stanford.edu>

Testing function for loading data in tests/test_*.py
"""

import pathlib
from importlib.util import find_spec
from scipy.io.matlab import loadmat


def test_anchor_func():
    """ Anchor function for locating this .py file """
    return


def get_test_path():
    """ Find the path to the directory containing the test_load_data.py file """
    f_path = pathlib.Path(test_anchor_func.__code__.co_filename)
    if f_path.is_file():
        return f_path.parents[0].resolve()
    else:
        somata_path = pathlib.Path(find_spec('somata').origin)  # find the package root __init__.py filepath
        return (somata_path.parents[1] / 'tests').resolve()  # 2-levels up


def _load_data(filename, return_path=False):
    """ Load data files for test functions """
    full_fn = get_test_path() / '_test_data' / filename
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
