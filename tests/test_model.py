import gpflow as gpf
import tensorflow_probability as tfp

from gpflow.utilities import to_default_float

from demo.data.toy_data import create_toy_data
from kernel import LfmKernel
from model import LfmModel


def test_model():
    X, X_train, Y_train = create_toy_data()
    n_output, n_latent = 2, 2
    kernel = LfmKernel(num_output=n_output, num_latent=n_latent, lengthscales=5.0, B=0.0, D=0.5, S=1.0)
    model = LfmModel(data=(X_train, Y_train), kernel=kernel, num_output=n_output, num_latent=n_latent)
    model.kernel.D.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(1.0))
    model.kernel.S.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(1.0))
    model.kernel.lengthscales.prior = tfp.distributions.HalfNormal(to_default_float(2.0))
    # Optimize
    opt = gpf.optimizers.Scipy()
    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
    gpf.utilities.print_summary(model)

