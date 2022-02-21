import numpy as np
import tensorflow as tf
import gpflow as gpf

from demo.data.toy_data import create_toy_data
from kernel import LfmKernel
from model import LfmModel


def cross_covariance_yf_single(kernel, X, Xnew, ell, S):
    """
    Cross-covariance function for single f from LFM paper [Alvarez et al., 2009].
    There is one type in the derivation of the paper, X_dist should have plus sign in both places.

    :param kernel: LFM Kernel
    :param X:
    :param Xnew:
    :param ell: lengthscale for single r
    :param S: sensitivities for single r
    :return:
    """
    X_val = X[:, 0]  # (N1,)
    X_p_idx = tf.cast(X[:, 1], tf.int32)  # (N1,)
    Xnew_val = Xnew[:, 0]  # (N2, )

    X_dist = X_val[:, None] - Xnew_val[None, :]  # (N1, N2)
    Dp = tf.gather(kernel.D, X_p_idx)[:, None]  # (N1, 1)
    Sp = tf.gather(S, X_p_idx)[:, None]  # (N1, 1)
    v = ell * Dp / 2  # (N1, 1)

    mult = 0.5 * np.sqrt(np.pi) * Sp * ell * tf.exp(tf.square(v))  # (N1, 1)
    mult_exp = tf.exp(-Dp * X_dist)  # (N1, N2)
    erf_term1 = X_dist / ell - v  # (N1, N2)
    erf_term2 = Xnew_val[None, :] / ell + v  # (N1, N2)
    return mult * mult_exp * (tf.math.erf(erf_term1) + tf.math.erf(erf_term2))


def predict_lf_single(model, X, Y, Xnew, kmn, full_cov=False,):
    Xnew_val = Xnew[:, 0]  # (N2, )
    kmm = model.kernel(X)
    se_kernel = gpf.kernels.SquaredExponential(lengthscales=model.kernel.lengthscales)
    knn = se_kernel(Xnew_val[:, None], full_cov=full_cov)
    kmm_plus_s = model._add_noise_cov(kmm)

    conditional = gpf.conditionals.base_conditional
    f_mean, f_var = conditional(
        kmn, kmm_plus_s, knn, Y, full_cov=full_cov, white=False
    )
    return f_mean, f_var


def test_cross_covariance_vectorized():
    X, X_train, Y_train = create_toy_data()
    n_output, n_latent = 2, 3
    kernel = LfmKernel(num_output=n_output, num_latent=n_latent,)
    model = LfmModel(data=(X_train, Y_train), kernel=kernel, num_output=n_output, num_latent=n_latent)

    gpf.utilities.print_summary(kernel)

    X_new = np.hstack([X, np.ones_like(X)])
    k_cross_covar = model.cross_covariance_yf(X_train, X_new)  # (N1, R, N2)
    f_mean, f_var = model.predict_lf(X_new)

    for i in range(n_latent):
        kmn_single = cross_covariance_yf_single(kernel, X_train, X_new,
                                                kernel.lengthscales[i],
                                                kernel.S[:, i])
        assert np.allclose(k_cross_covar[:, i, :], kmn_single)
        f_mean_single, f_var_single = predict_lf_single(model, X_train, Y_train, X_new, kmn_single)
        assert np.allclose(f_mean[i, :, :], f_mean_single)
        assert np.allclose(f_var[i, :, :], f_var_single)


if __name__ == '__main__':
    test_cross_covariance_vectorized()

