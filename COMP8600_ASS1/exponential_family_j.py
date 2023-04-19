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

    grad_psi = torch.zeros((batch_size, eta.shape[1]))
    for i in range(batch_size):
        grad_psi[i] = jacobian(exp_fam.log_partition, eta[i])
    # print(grad_psi.shape)
    return grad_psi

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
    grad_psi = grad_psi_wrt_eta(exp_fam, model.predict(xs, theta), batch_size)
    # grad_ll = torch.zeros((batch_size, 1))
    # print(grad_psi.shape)
    # for i in range(batch_size):

    #     t_v = exp_fam.sufficient_statistic(ys[i]) - grad_psi[i]
    #     grad_ll[i] = vjp(model.predict, (xs[i], theta), t_v.unsqueeze(0))[1][1]

    grad_psi_wrt_etas = grad_psi_wrt_eta(exp_fam, model.predict(xs, theta), batch_size)
    grad_ll = torch.zeros((batch_size, theta.shape[-1]))
    for i in range(batch_size):
        grad_ll[i] = vjp(model.predict, (xs[i],theta), (exp_fam.sufficient_statistic(ys[i]) - grad_psi_wrt_etas[i]).unsqueeze(0))[1][1]
    return grad_ll

def update(exp_fam: ExponentialFamily, model: Model,
           xs: torch.Tensor, ys: torch.Tensor, theta: torch.Tensor,
           batch_size: int, lr: float = 1e-1) -> torch.Tensor:
    """ Calculate the next set of parameters from a gradient accent update
        step.
        Ensure that this will work for any sized batch

        expected output size: (| Theta |,) where | Theta | is the
        number of model parameters 
    """
    grad_ll = log_likelihood_grad(exp_fam, model, xs, ys, theta, batch_size)
    theta = theta + lr * torch.sum(grad_ll, 0)/batch_size
    return theta #TODO
