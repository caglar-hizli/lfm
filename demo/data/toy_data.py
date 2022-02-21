import numpy as np
from scipy.integrate import odeint


cos_fnc = lambda t: np.cos(t)


def create_toy_data():
    np.random.seed(123)
    S = np.array([[-0.5], [.2]])  # S_{rp} = S_{11} and S_{12} for y1 and y2
    D = np.array([.5, .3])
    # Dense set of time points to solve the model on
    ttd = np.linspace(0., 5., 100)

    # Time-Evolution equation
    def dXdt(X, t):
        # equivalent to diag(-D).dot(X) + S.dot(g)
        return -D * X + S.dot(cos_fnc(t))[:, 0]

    # numerically solve the ODE
    # This could also be done by a discrete sum
    sol = odeint(dXdt, [0., 0.], ttd)

    # subsample the time vector to create our training data
    N1 = 8  # no. of samples from y1(t)
    N2 = 15  # no. of samples from y2(t)
    y1_ind = np.sort(np.random.choice(ttd.size, size=N1, replace=False))
    # sample only from the 1st two thirds of y(t)
    y2_ind = np.sort(np.random.choice(ttd.size * 2 // 3, size=N2, replace=False))
    t1, Y1 = ttd[y1_ind], sol[y1_ind, 0]
    t2, Y2 = ttd[y2_ind], sol[y2_ind, 1]
    #
    y_idx = np.repeat([0, 1], repeats=[N1, N2])
    T = np.hstack([t1, t2])
    X_train = np.hstack([T.reshape(-1, 1), y_idx.reshape(-1, 1)])
    Y_train = np.hstack([Y1, Y2]).reshape(-1, 1)
    X = ttd.reshape(-1, 1)
    return X, sol, X_train, Y_train
