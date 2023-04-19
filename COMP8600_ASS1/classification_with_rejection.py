import torch

from framework.exp_fam_model import Model, ExponentialFamily


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

def h_classifier_pred(exp_fam: ExponentialFamily, model: Model,
                      parameters: torch.Tensor, xs: torch.Tensor,
                      num_classses: int, batch_size: int) -> torch.Tensor:
    """ Calculate the optimal h* classifier.
        Make sure this works for any batch size

        expected output size: (batch_size,)
    """
    h_star = torch.zeros(batch_size)
    eta = model.predict(xs, parameters)
    class_probability = [0 for _ in range(num_classses)]
    for i in range(batch_size):
        for j in range(num_classses):
            u = exp_fam.sufficient_statistic(torch.Tensor([j]).to(torch.int64))
            class_probability[j] = torch.exp(torch.matmul(eta[i].T, u.T) - exp_fam.log_partition(eta[i]))
        h_star[i] = class_probability.index(max(class_probability))
    return h_star


def r_reject_pred(exp_fam: ExponentialFamily, model: Model,
                  parameters: torch.Tensor, xs: torch.Tensor,
                  num_classses: int, rejection_cost: float,
                  batch_size: int) -> torch.Tensor:
    """ Calculate the optimal r* classifier.
        Make sure this works for any batch size

        expected output size: (batch_size,)
    """
    r_star = torch.zeros(batch_size)
    eta = model.predict(xs, parameters)
    class_probability = [0 for _ in range(num_classses)]
    for i in range(batch_size):
        for j in range(num_classses):
            u = exp_fam.sufficient_statistic(torch.Tensor([j]).to(torch.int64))
            class_probability[j] = torch.exp(torch.matmul(eta[i].T, u.T) - exp_fam.log_partition(eta[i]))
        r_star[i] = max(class_probability) - 1 + rejection_cost
    return r_star


def cwr_pred(exp_fam: ExponentialFamily, model: Model,
             parameters: torch.Tensor, xs: torch.Tensor,
             num_classses: int, rejection_cost: float,
             batch_size: int) -> torch.Tensor:
    """ Calculate the CwR classifier with h* and r*.
        Make sure this works for any batch size

        expected output size: (batch_size,)
        You will want to use ".int()" to make the output tensor output integers
    """
    h_pred = h_classifier_pred(exp_fam, model, parameters, xs, num_classses, batch_size)
    r_pred = r_reject_pred(exp_fam, model, parameters, xs, num_classses, rejection_cost, batch_size)
    cwr_pred = torch.zeros(batch_size)
    for i in range(batch_size):
        if r_pred[i] > 0:
            cwr_pred[i] = h_pred[i]
        else:
            cwr_pred[i] = -1
    cwr_pred = cwr_pred.int()
    return cwr_pred
