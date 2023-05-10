from typing import Callable, Tuple
import numpy as np
from scipy.optimize import minimize
from kernels import Matern
from gp import GPR
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib import animation, gridspec
from matplotlib.lines import Line2D
from IPython.display import HTML, clear_output
from functools import partial

# Class Structure


class BO:
    """
    Class the performs Bayesian Global Optimisation

    Arguments:
    ----------
        X_init: ndarray of shape (2, 1)
            The two initial starting points for X

        Y_init: ndarray of shape (2, 1)
            The two initial starting points for y, evaluated under f

        f: function 
            The black-box expensive function to evaluate

        noise_level: float
            Gaussian noise added to the function

        bounds: tuple
            Bounds for variable X 
    """
    def __init__(self, X_init: np.ndarray, Y_init: np.ndarray, f: Callable,
                 noise_level: float, bounds: Tuple, n_iter: int, xi: float, title: str, **kwargs) -> None:
        
        self.X_sample = X_init
        self.Y_sample = Y_init
        
        self.noise_level = noise_level
        self.bounds = bounds
        self.f = f
        self.n_iter = n_iter
        self.xi = xi

        # You don't need variables from kwargs. Skip and do not change
        self.X = kwargs['X']
        self.Y = kwargs['Y']

        # TODO Q2.7b
        # ------------------------------------------------------------------------
        # FIXME
        # m = Matern(nu=?, length_scale=1.0, variance=1.0)
        # ------------------------------------------------------------------------
        self.gpr = GPR(kernel=m, noise_level=self.noise_level, n_restarts=5)

        self.fig = plt.figure(figsize=(12, 6), clear=True)
        gs = gridspec.GridSpec(2, 1)
        plt.title(title)
        self.ax1 = self.fig.add_subplot(gs[0])
        plt.setp(self.ax1.get_xticklabels(), visible=False)
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)

        self.ax1.plot(self.X, self.Y, color='darkred',
                      label='Noise-free objective')

        self.ax1_lines = [
            Line2D([], [], marker='o', ls='None', label='Observations'),
            Line2D([], [], ls='--', linewidth=2.0,
                   color='black', label='Surrogate Function'),
            Line2D([], [], ls='-', linewidth=2.0, color='black', alpha=0.4)]

        self.ax2_lines = [
            Line2D([], [], ls='-', color='blue', label='Acquisition Function'),
            Line2D([], [], marker='x', ls='None', color='blue', label='Maximum / Next Sampling Position')]
        for l in self.ax1_lines:
            self.ax1.add_line(l)
        for l in self.ax2_lines:
            self.ax2.add_line(l)

    def __call__(self, step, acquisition: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of BO algorithm

        The X_sample and Y_sample are creating the dataset D and 
        need to be updated as per line 5 of Algorithm 3 (see __init__).


        Arguments:
        ----------
            acquisition: function
                The chosen acquisition function (EI or PI)

            xi: float
                Trade-off between exploitation and exploration

            n_iter: int
                Number of iterations to run

        Returns:
        ----------
        Animation arguments (do not change)

        """
        # TODO Q2.7a
        # Implement the Bayesian optimisation algorithm following the
        # Algorithm (3) from the assignment sheet.

        # As a surrogate model you should use your GP regression model, and
        # to sample new points using the `sample_next_point` function implemented
        # in the previous questions.

        # FIXME
        # HINT (I): X_next is the result of the `sample_next_point` function
        # HINT (II): You do not need to implement the for-loop operation as this
        #            is implicitly done for you in the call_animation() function
        # HINT (III): Given the current codebase, an efficient implementation would first perform a GP update
        #             and then continue with the next algorithmic steps.
        # HINT (IV): Perform your xi scheduling policy here.
        X_next = None

        # DO NOT CHANGE
        # Plot samples, surrogate function, noise-free objective and next sampling location
        plt.title(f'Iteration {step+1} | xi = ' + str(self.xi))

        mu, std = self.gpr.predict(self.X, return_std=True)
        self.ax1.collections.clear()
        self.ax1_lines[0].set_data(self.X_sample, self.Y_sample)
        self.ax1_lines[1].set_data(self.X, mu)
        self.ax1.fill_between(self.X.ravel(),
                              mu.ravel() - std.ravel(),
                              mu.ravel() + std.ravel(),
                              color='#82bfbc')
        acq = acquisition(self.X, self.X_sample, self.gpr, self.xi)
        self.ax2_lines[0].set_data(self.X, acq)
        self.ax2_lines[1].set_data([self.X[np.argmax(acq)]], [max(acq)])

        return self.ax1_lines + self.ax2_lines

    def sample_next_point(self, acquisition_func: Callable, gpr: object,
                          xi: float, n_restarts: int = 25) -> np.ndarray:
        """
        Proposes the next point to sample the loss function for 
        by optimising the acquisition function using the L-BFGS-B algorithm.
        Initial values should be sampled uniformly with respect to the bounds (see __init__).

        Arguments:
        ----------
            acquisition_func: function.
                Acquisition function to optimise.

            gpr: GPR object.
                Gaussian process trained on previously evaluated hyperparameters.

            n_restarts: integer.
                Number of times to run the minimiser with different starting points.

        Returns:
        --------
            best_x: ndarray of shape (k, 1), where k is the shape of optimal solution x.
        """
        best_x = None
        best_acquisition_value = 1
        n_params = self._X_sample.shape[1]

        # TODO Q2.6
        # Implement the 'sampling' step via acquisition maximisation
        # HINT: Again, formulate the equivalent minimisation problem and solve with the built-in L-BFGS.
        
        # FIXME

        raise NotImplementedError
    
    def init_animation(self):
        self.ax1.set_xlim([-3, 6])
        x1min, x1max = self.ax1.get_xlim()
        self.ax1.set_xticks(np.round(np.linspace(x1min, x1max, 5), 2))
        self.ax1.set_ylim([-2, 2])
        y1min, y1max = self.ax1.get_ylim()
        self.ax1.set_yticks(np.round(np.linspace(y1min, y1max, 4), 2))
        self.ax1.legend(loc='upper left')
        self.ax1.grid()
        self.ax2.legend(loc='upper left')
        self.ax2.set_xlim([-3, 6])
        x2min, x2max = self.ax2.get_xlim()
        self.ax2.set_xticks(np.round(np.linspace(x2min, x2max, 5), 2))
        self.ax2.set_ylim([1e-10, 0.4])
        y2min, y2max = self.ax2.get_ylim()
        self.ax2.set_yticks(np.round(np.linspace(y2min, y2max, 4), 2))
        self.ax2.set_yscale('symlog')
        self.ax2.grid()

        return self.ax1_lines+self.ax2_lines

    def call_animation(self, acquisition):
        anim = animation.FuncAnimation(self.fig, partial(self.__call__, acquisition=acquisition), frames=range(
            self.n_iter), interval=1000, init_func=self.init_animation, blit=True)
        clear_output()
        html_anim = HTML(anim.to_jshtml())
        self.plot_convergence(self.X_sample, self.Y_sample)

        return html_anim

    def plot_convergence(self, X_sample, Y_sample, n_init=2):
        plt.figure(figsize=(20, 5))

        x = X_sample[n_init:].ravel()
        y = Y_sample[n_init:].ravel()
        r = range(1, len(x)+1)

        x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
        y_max_watermark = np.maximum.accumulate(y)

        star = mpath.Path.unit_regular_star(6)
        circle = mpath.Path.unit_circle()
        # concatenate the circle with an internal cutout of the star
        verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
        codes = np.concatenate([circle.codes, star.codes])
        cut_star = mpath.Path(verts, codes)

        plt.subplot(1, 2, 1)
        plt.plot(r[1:], x_neighbor_dist, '--k', marker=cut_star, markersize=10)
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('Distance between consecutive x\'s')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(r, y_max_watermark, '--c', marker=cut_star, markersize=10)
        plt.xlabel('Iteration')
        plt.ylabel('Best Y')
        plt.title('Value of best selected sample')
        plt.grid()

