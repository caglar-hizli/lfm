import numpy as np
import gpflow as gpf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from gpflow.utilities import to_default_float

from demo.data.toy_data import create_toy_data, cos_fnc
from kernel import LfmKernel
from model import LfmModel


def run_toy_example1():
    X, Y, X_train, Y_train = create_toy_data()
    X1, Y1 = X_train[X_train[:, 1] == 0, 0], Y_train[X_train[:, 1] == 0]
    X2, Y2 = X_train[X_train[:, 1] == 1, 0], Y_train[X_train[:, 1] == 1]

    plt.figure(figsize=(8, 4))
    plt.plot(X, Y, alpha=0.5)
    plt.plot(X1, Y1, 'x', label='Y1')
    plt.plot(X2, Y2, 'x', label='Y2')
    plt.legend(fontsize=18)
    plt.show()

    n_output, n_latent = 2, 1
    kernel = LfmKernel(num_output=n_output, num_latent=n_latent, lengthscales=5.0, B=0.0, D=0.5, S=1.0)
    model = LfmModel(data=(X_train, Y_train), kernel=kernel, num_output=n_output, num_latent=n_latent)
    model.kernel.D.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(1.0))
    model.kernel.S.prior = tfp.distributions.Normal(to_default_float(0.0), to_default_float(1.0))
    model.kernel.lengthscales.prior = tfp.distributions.HalfNormal(to_default_float(2.0))

    opt = gpf.optimizers.Scipy()
    opt_logs = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1000))
    gpf.utilities.print_summary(model)

    # Plot ys
    Xnew = np.vstack([np.hstack([X.reshape(-1, 1), np.zeros((100, 1))]),
                      np.hstack([X.reshape(-1, 1), np.ones((100, 1))])])
    y_mean, y_var = model.predict_y(Xnew)

    plt.figure(figsize=(8, 4))
    plt.plot(X1, Y1, 'x', label='Y1')
    plt.plot(X2, Y2, 'x', label='Y2')
    plt.plot(X, y_mean[:100, 0], "C0", lw=2, label='Y1 pred')
    plt.plot(X, y_mean[100:, 0], "red", lw=2, label='Y2 pred')
    plt.legend(fontsize=18)
    _ = plt.xlim(-0.1, 5 + 0.1)
    plt.tight_layout()
    plt.show()

    # Plot fs
    f_mean, f_var = model.predict_lf(Xnew[:100, :])
    mean = f_mean[0]
    var = f_var[0]

    plt.figure(figsize=(8, 4))
    # plt.plot(X_train, y_train, "kx", mew=2)
    plt.plot(X, -cos_fnc(X), 'r', label='f true')
    plt.plot(X, mean, "C0", lw=2, label='f pred')
    plt.fill_between(
        X[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color="C0",
        alpha=0.2,
    )
    plt.legend(fontsize=18)
    _ = plt.xlim(-0.1, 5 + 0.1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_toy_example1()


