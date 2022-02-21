import numpy as np
import tensorflow as tf
import gpflow as gpf

from demo.data.toy_data import create_toy_data
from kernel import LfmKernel


def test_psd():
    _, X, Y = create_toy_data()
    n_output, n_latent = 2, 3
    kernel = LfmKernel(num_output=n_output, num_latent=n_latent, )
    gpf.utilities.print_summary(kernel)

    # Test if the matrix is positive definite
    kmm = kernel(X, X)
    kmm_jitter = kmm + 1e-8 * np.eye(kmm.shape[0])
    L = np.linalg.cholesky(kmm_jitter)


def test_diag():
    _, X, Y = create_toy_data()
    n_output, n_latent = 2, 3
    kernel = LfmKernel(num_output=n_output, num_latent=n_latent, )
    var = kernel.K_diag(X)
    k_diag_true = tf.linalg.tensor_diag_part(kernel(X, X))  # (N1,)
    assert np.allclose(var, k_diag_true)
