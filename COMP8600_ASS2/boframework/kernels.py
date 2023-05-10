import numpy as np
from recordclass import recordclass


# Class Structure


class Matern:
    """
    Matern kernel.

    Arguments:
    ----------
    nu : float
        The parameter nu controlling the smoothness of the learned function.

    length_scale : float, default=1.0
        The length scale of the kernel.

    length_scale_bounds : pair of floats >= 0, default=(1e-5, 1e3)
        The lower and upper bound on 'length_scale'.

    variance : float, default=1.0
        The signal variance of the kernel

    variance_bounds : pair of floats >= 0, default=(1e-5, 1e2)
        The lower and upper bound on 'variance'.
    """

    def __init__(self, nu: float, length_scale: float = 1.0, length_scale_bounds: tuple = (1e-5, 1e3),
                 variance: float = 1.0, variance_bounds: tuple = (1e-5, 1e2)) -> None:
        self.nu = nu
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.variance = variance
        self.variance_bounds = variance_bounds

        hyper_structure = recordclass('Hyperparameter', ('name', 'value', 'bounds'))

        self.hyperparameter_struct = [hyper_structure('length_scale', length_scale, self.length_scale_bounds),
                                      hyper_structure('variance', variance, self.variance_bounds)]

    def get_theta(self):
        """
        Returns the log-transformed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns:
        --------
        theta : ndarray of shape (n_dims,)
            The log-transformed hyperparameters of the kernel
        """
        theta = []
        hyperparams = self.get_hyperparameters()
        for hyperparameter_struct in self.hyperparameter_struct:
            theta.append(hyperparams[hyperparameter_struct.name])

        return np.log(np.hstack(theta))

    def set_theta(self, theta):
        hyperparams = self.get_hyperparameters()
        i = 0
        for hyperparameter_struct in self.hyperparameter_struct:
            hyperparams[hyperparameter_struct.name] = np.exp(theta[i])
            i += 1

        self.set_hyperparameters(**hyperparams)

    def get_bounds(self):
        """ 
        Returns the log-transformed hyperparameter bounds.

        Returns:
        --------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed hyperparameters of the kernel. The second
            dimension indicates the value of lower and upper bounds.
        """
        bounds = []
        for hyperparameter_struct in self.hyperparameter_struct:
            bounds.append(hyperparameter_struct.bounds)

        return np.log(np.vstack(bounds))

    def get_hyperparameters(self):
        """
        Get the hyperparameters of this kernel.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        hyperparams = dict()

        args = []
        for hyperparameter_struct in self.hyperparameter_struct:
            args.append(hyperparameter_struct.name)

        for arg in args:
            hyperparams[arg] = getattr(self, arg)

        return hyperparams

    def set_hyperparameters(self, **hyperparams):
        for key, value in hyperparams.items():
            setattr(self, key, value)

    def __call__(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        Return the kernel k(X, Y).

        Arguments:
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            should be evaluated instead.

        Returns:
        --------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        """

        X = np.atleast_2d(X)
        length_scale = np.squeeze(self.length_scale).astype(float)

        # TODO Q2.2b
        # Uncomment the code and implement the Matern class covariance functions for different values of nu
        # FIXME
        # if self.nu == :
        #     pass
        # elif self.nu == :
        #     pass
        # ...
        # else:
        #     # Do not change
        #     raise NotImplementedError

        raise NotImplementedError
