###### DO NOT CHANGE ######

from abc import ABC, abstractmethod

import torch

class Model(ABC):

    @abstractmethod
    def make_random_parameter(self) -> torch.Tensor:
        """ Returns a 1D vector of randomized parameters.

            Expected output size: (| Theta |,) where | Theta | is the
            number of model parameters
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """ Predicted output of the model. Assumes that x is in a batch, i.e.,
            the initial dimension is the batch number

            Expected output size: (batch_size, m) where m is the exponential
            family parameter dimension size
        """
        pass

class ExponentialFamily(ABC):

    @abstractmethod
    def log_partition(self, eta: torch.Tensor) -> float:
        """ Log partition function of an exponential family.
            Broadcasts to batches.

            Expected output size: (1,)
        """
        pass

    @abstractmethod
    def sufficient_statistic(self, y: torch.Tensor) -> torch.Tensor:
        """ Sufficient statistics of an exponential family.
            Broadcasts to batches.

            Expected output size: (m,) where m is the exponential
            family parameter dimension size
        """
        pass