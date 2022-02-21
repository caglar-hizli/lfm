import numpy as np
import tensorflow as tf
import gpflow as gpf

from kernel import LfmKernel


# TODO add docs
class LfmModel(gpf.models.GPR):

    def __init__(
        self,
        data,
        kernel,
        num_output,
        num_latent,
        mean_function=None,
        noise_variance=1.0,
    ):
        """
        LFM Model implementation for a simple GP Regression case.

        :param data:
        :param kernel:
        :param num_output:
        :param num_latent:
        :param mean_function:
        :param noise_variance:
        """
        if not isinstance(kernel, LfmKernel):
            raise NotImplementedError('LfmModel only implemented for convolved square exponential kernel!')

        super().__init__(data, kernel, mean_function=mean_function, noise_variance=noise_variance)
        self.num_output = num_output  # P
        self.num_latent = num_latent  # R

    def cross_covariance_yf(self, X, Xnew):
        """
        Cross-covariance function for single f from LFM paper [Alvarez et al., 2009].
        There is one type in the derivation of the paper, X_dist should have plus sign in both places.

        :param X:
        :param Xnew:
        :return:
        """
        X_val = X[:, 0]  # (N1,)
        X_p_idx = tf.cast(X[:, 1], tf.int32)  # (N1,)
        Xnew_val = Xnew[:, 0]  # (N2, )

        ell = self.kernel.lengthscales[:, None, None]  # (R, 1, 1)

        X_dist = X_val[:, None] - Xnew_val[None, :]  # (N1, N2)
        X_dist = X_dist[None, :, :]  # (1, N1, N2)
        Dp = tf.gather(self.kernel.D, X_p_idx)[None, :, None]  # (1, N1, 1)
        Sp = tf.transpose(tf.gather(self.kernel.S, X_p_idx))[:, :, None]  # (R, N1, 1)
        vp = (ell * Dp / 2)  # (R, N1, 1)

        mult = 0.5 * np.sqrt(np.pi) * Sp * ell * tf.exp(tf.square(vp))  # (R, N1, 1)
        mult_exp = tf.exp(-Dp * X_dist)  # (1, N1, N2)
        erf_term1 = X_dist / ell - vp  # (R, N1, N2)
        erf_term2 = Xnew_val[None, None, :] / ell + vp  # (R, N1, N2)
        cc = mult * mult_exp * (tf.math.erf(erf_term1) + tf.math.erf(erf_term2))
        # TODO extra transpose
        return tf.transpose(cc, (1, 0, 2))

    def predict_lf(
            self, Xnew, full_cov=False,
    ):
        """

        :param Xnew: (N, 2)
                Same Xnew is assumed for all latent factors f_r(.). This is a reasonable assumption, as we're
                mostly interested in latent factors in the same space. Besides, if they're active on different regions
                one can pass the union of regions as Xnew (though this would result in extra computation),
                or pass them as Xnew one by one.
        :param full_cov: To get the full covariance or just the variance (diag).
        :return:
        """
        X, Y = self.data
        Xnew_val = Xnew[:, 0]  # (N2, )

        kmm = self.kernel(X)
        se_kernel = gpf.kernels.SquaredExponential(lengthscales=self.kernel.lengthscales)
        knn = se_kernel(Xnew_val[:, None], full_cov=full_cov)
        kmm_plus_s = self._add_noise_cov(kmm)

        kmn = self.cross_covariance_yf(X, Xnew,)  # (N1, R, N2)
        conditional = gpf.conditionals.base_conditional
        f_mean, f_var = conditional(
            kmn, kmm_plus_s, knn, Y, full_cov=full_cov, white=False
        )
        return f_mean, f_var
