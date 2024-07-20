"""
Author: Mingjian He <mh1@stanford.edu>

Testing function for importing numpy and torch libraries
"""
import torch
import numpy as np


def test_import_numpy_torch(dim=2):
    """
    Due to a strange dependency on sys lib5.dll between numpy and torch in
    macOS(osx-arm64), one must import torch first then numpy to avoid the error:
        >>> zsh: segmentation fault  python

    This is a very problematic issue because it fails silently without any
    error message unless the offending function call is asked directly in a
    python session because it is a system-level error that does not return an
    error code to parent stacks, leading to all superordinate processes hanging.

    This unit test checks that numpy and torch can be imported without error.
    Note that we do not want to import within the function as it is executed
    in runtime during test suite collection such as by pytest.
    """

    np.random.seed(1)
    A = np.random.rand(dim, dim)
    A = A @ A.T + np.eye(dim)
    _ = np.linalg.inv(A)  # this call depends on lib5.dll
    A = torch.as_tensor(data=A, dtype=torch.float32)  # does not call lib5.dll
    _ = torch.linalg.qr(A, mode='complete')  # this call also invokes lib5.dll

    return


if __name__ == "__main__":
    test_import_numpy_torch()
    print('numpy and torch import test finished without exception.')
