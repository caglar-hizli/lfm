import numpy as np
import tensorflow as tf
import gpflow as gpf
from gpflow.utilities import to_default_float
from scipy.integrate import odeint


class LfmKernel(gpf.kernels.Kernel):

    def __init__(self, num_output, num_latent, lengthscales=1.0, B=0.0, D=1.0, S=1.0, **kwargs):
        """

        :param num_output: The number of outputs P.
        :param num_latent: The number of latent factors R.
        :param lengthscales: The kernel lengthscale.
        :param B: The base
        :param D: The decay rate.
        :param S: Coupling constants (sensitivities)
        :param kwargs:
        """
        for kwarg in kwargs:
            if kwarg not in {"name", "active_dims"}:
                raise TypeError(f"Unknown keyword argument: {kwarg}")

        super().__init__(**kwargs)
        self.num_output = num_output
        self.num_latent = num_latent
        # TODO better init
        self.lengthscales = gpf.Parameter(np.ones(self.num_latent) * lengthscales +
                                          np.random.uniform(0.0, 0.5, size=self.num_latent),
                                          transform=gpf.utilities.positive())
        # self.B = gpf.Parameter(tf.ones(self.num_output) * B)
        self.D = gpf.Parameter(np.ones(self.num_output) * D + np.random.uniform(0.0, 0.5, size=self.num_output))
        self.S = gpf.Parameter(np.ones((self.num_output, self.num_latent)) * S
                               + np.random.uniform(0.0, 0.5, size=(self.num_output, self.num_latent)))  # (P, R)
        self._validate_ard_active_dims(self.lengthscales)

    def K(self, X, X2=None):
        """
        LFM kernel.

        :param X: (N1 x 2), 1st column provides dimension value, 2nd column provides output type p in {0, ..., P-1}.
        :param X2: (N2 x 2)
        :return: (N1 x N2 x R) matrix
        """
        if X2 is None:
            X2 = X
        h_term = self._h(X, X2) + tf.transpose(self._h(X2, X), (1, 0, 2))  # (N1, N2, R)
        X_p_idx = tf.cast(X[:, 1], tf.int32)  # (N1,)
        X2_q_idx = tf.cast(X2[:, 1], tf.int32)  # (N2,)
        ell = self.lengthscales[None, None, :]  # (1, 1, R)
        Spr = tf.gather(self.S, X_p_idx)[:, None, :]  # (N1, 1, R)
        Sqr = tf.gather(self.S, X2_q_idx)[None, :, :]  # (1, N2, R)
        mult = Spr * Sqr * 0.5 * np.sqrt(np.pi) * ell  # (N1, N2, R)
        k = tf.reduce_sum(mult * h_term, axis=2)  # (N1, N2)
        return k  # this returns a 2D tensor

    def K_diag(self, X):
        h_term = self._h_diag(X) * 2  # (N1, R)
        X_p_idx = tf.cast(X[:, 1], tf.int32)  # (N1,)
        ell = self.lengthscales[None, :]  # (1, R)
        Spr = tf.gather(self.S, X_p_idx)  # (N1, R)
        mult = tf.square(Spr) * 0.5 * np.sqrt(np.pi) * ell  # (N1, R)
        k_diag = tf.reduce_sum(mult * h_term, axis=1)
        return k_diag  # this returns a 1D tensor

    def _h_diag(self, X):
        X_val = tf.gather(X, 0, axis=1)[:, None]  # (N1, 1)
        X_p_idx = tf.cast(X[:, 1], tf.int32)  # (N1,)

        ell = self.lengthscales[None, :]  # (1, R)
        Dp = tf.gather(self.D, X_p_idx)[:, None]  # (N1, 1)

        vp = (ell * Dp / 2)  # (N1, R)
        D_sum = Dp + Dp  # (N1, 1)
        mult = tf.exp(tf.square(vp)) / D_sum  # (N1, R)
        erf_term1 = - vp  # (N1, R)
        erf_term2 = X_val / ell + vp  # (N1, R)
        erf_term3 = X_val / ell - vp  # (N1, R)
        term1 = (tf.math.erf(erf_term1) + tf.math.erf(erf_term2))  # (N1, R)
        term2 = tf.exp(-2 * Dp * X_val) * (tf.math.erf(erf_term3) + tf.math.erf(vp))  # (N1, R)
        return mult * (term1 - term2)  # (N1, R)

    def _h(self, X, X2=None):
        """
        Implementation of h_{pq} function from LFM paper [Alvarez et al, 2009].

        :param X: (N1 x 2), first dimension value, second dimension type
        :param X2: (N2 x 2)
        :return: (N1 x N2 x R) matrix
        """
        if X2 is None:
            X2 = X
        X_val = tf.gather(X, 0, axis=1)[:, None, None]  # (N1, 1, 1)
        X2_val = tf.gather(X2, 0, axis=1)[None, :, None]  # (1, N2, 1)
        X_p_idx = tf.cast(X[:, 1], tf.int32)  # (N1,)
        X2_q_idx = tf.cast(X2[:, 1], tf.int32)  # (N2,)

        ell = self.lengthscales[None, None, :]  # (1, 1, R)
        X_dist = X_val - X2_val  # (N1, N2, 1)
        Dp = tf.gather(self.D, X_p_idx)[:, None, None]  # (N1, 1, 1)
        Dq = tf.gather(self.D, X2_q_idx)[None, :, None]  # (1, N2, 1)

        vp = (ell * Dp / 2)  # (N1, 1, R)
        D_sum = Dp + Dq  # (N1, N2, 1)
        mult = tf.exp(tf.square(vp)) / D_sum  # (N1, N2, R)
        erf_term1 = X_dist / ell - vp  # (N1, N2, R)
        erf_term2 = X2_val / ell + vp  # (N1, N2, R)
        erf_term3 = X_val / ell - vp  # (N1, 1, R)
        term1 = tf.exp(-Dp * X_dist) * (tf.math.erf(erf_term1) + tf.math.erf(erf_term2))  # (N1, N2, R)
        term2 = tf.exp(-Dp * X_val - Dq * X2_val) * (tf.math.erf(erf_term3) + tf.math.erf(vp))  # (N1, N2, R)
        return mult * (term1 - term2)  # (N1, N2, R)
