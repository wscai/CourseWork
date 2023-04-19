from framework.exp_fam_model import Model, ExponentialFamily

import torch
from torch.autograd.functional import jacobian, vjp


# This is a copy of the class structure found in framework/exp_fam_model.py
# class Model(ABC):
#
#     @abstractmethod
#     def make_random_parameter(self) -> torch.Tensor:
#         """ Returns a 1D vector of randomized parameters.
#
#             Expected output size: (| Theta |,) where | Theta | is the
#             number of model parameters
#         """
#         pass
#
#     @abstractmethod
#     def predict(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
#         """ Predicted output of the model. Assumes that x is in a batch, i.e.,
#             the initial dimension is the batch number
#
#             Expected output size: (batch_size, m) where m is the exponential
#             family parameter dimension size
#         """
#         pass
#
# class ExponentialFamily(ABC):
#
#     @abstractmethod
#     def log_partition(self, eta: torch.Tensor) -> float:
#         """ Log partition function of an exponential family.
#             Broadcasts to batches.
#
#             Expected output size: (1,)
#         """
#         pass
#
#     @abstractmethod
#     def sufficient_statistic(self, y: torch.Tensor) -> torch.Tensor:
#         """ Sufficient statistics of an exponential family.
#             Broadcasts to batches.
#
#             Expected output size: (m,) where m is the exponential
#             family parameter dimension size
#         """
#         pass

###################################################

def grad_psi_wrt_eta(exp_fam: ExponentialFamily, eta: torch.Tensor,
                     batch_size: int) -> torch.Tensor:
    """ Calculate the gradient of the log partition function wrt parameters eta
        (2.8).
        Ensure that this will work for any sized batch

        expected output size: (batch_size, m)
    """
    ans = torch.zeros(batch_size, eta.shape[1])
    for i in range(batch_size):
        ans[i, :] += jacobian(exp_fam.log_partition, eta[i])
    return ans


def log_likelihood_grad(exp_fam: ExponentialFamily, model: Model,
                        xs: torch.Tensor, ys: torch.Tensor,
                        theta: torch.Tensor, batch_size: int) -> torch.Tensor:
    """ Calculate the gradient of the log likelihood function wrt model
        parameters (2.9).
        Ensure that this will work for any sized batch

        expected output size: (batch_size, | Theta |) where | Theta | is the
        number of model parameters

        You will need to use vjp -- vector-jacobian-product -- to make this
        efficient.
    """
    dphi_deta = grad_psi_wrt_eta(exp_fam, model.predict(xs, theta), batch_size)

    gradient = torch.zeros(batch_size, len(theta))
    for i in range(batch_size):
        # calculate the gradient w.r.t. theta
        gradient[i, :] += \
        vjp(model.predict, (xs[i], theta), (exp_fam.sufficient_statistic(ys[i]) - dphi_deta[i]).unsqueeze(0))[1][1]
    return gradient


def update(exp_fam: ExponentialFamily, model: Model,
           xs: torch.Tensor, ys: torch.Tensor, theta: torch.Tensor,
           batch_size: int, lr: float = 1e-1) -> torch.Tensor:
    """ Calculate the next set of parameters from a gradient accent update
        step.
        Ensure that this will work for any sized batch

        expected output size: (| Theta |,) where | Theta | is the
        number of model parameters
    """
    update_gradient = lr * torch.sum(log_likelihood_grad(exp_fam, model, xs, ys, theta, batch_size), 0) / batch_size
    theta += update_gradient
    return theta
