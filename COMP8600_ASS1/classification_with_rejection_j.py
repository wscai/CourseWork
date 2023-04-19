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
    h_res = torch.zeros(batch_size)
    eta = model.predict(xs, parameters)
    p = torch.zeros(num_classses)
    for i in range(batch_size):
        for c_idx in range(num_classses):
            t_eta = eta[i].unsqueeze(0)
            t_sufi_stat = exp_fam.sufficient_statistic(torch.tensor(c_idx)).unsqueeze(1)
            p[c_idx] = torch.exp(torch.matmul(t_eta, t_sufi_stat) - exp_fam.log_partition(eta[i]))
        h_res[i] = torch.argmax(p)
    return h_res

def r_reject_pred(exp_fam: ExponentialFamily, model: Model,
                  parameters: torch.Tensor, xs: torch.Tensor,
                  num_classses: int, rejection_cost: float,
                  batch_size: int) -> torch.Tensor:
    """ Calculate the optimal r* classifier.
        Make sure this works for any batch size

        expected output size: (batch_size,)
    """
    r_res = torch.zeros(batch_size)
    eta = model.predict(xs, parameters)
    eta = model.predict(xs, parameters)
    p = torch.zeros(num_classses)
    for i in range(batch_size):
        for c_idx in range(num_classses):
            t_eta = eta[i].unsqueeze(0)
            t_sufi_stat = exp_fam.sufficient_statistic(torch.tensor(c_idx)).unsqueeze(1)
            p[c_idx] = torch.exp(torch.matmul(t_eta, t_sufi_stat) - exp_fam.log_partition(eta[i]))
        r_res[i] = torch.max(p) - (1 - rejection_cost)
    return r_res

def cwr_pred(exp_fam: ExponentialFamily, model: Model,
             parameters: torch.Tensor, xs: torch.Tensor,
             num_classses: int, rejection_cost: float,
             batch_size: int) -> torch.Tensor:
    """ Calculate the CwR classifier with h* and r*.
        Make sure this works for any batch size

        expected output size: (batch_size,)
        You will want to use ".int()" to make the output tensor output integers
    """
    h_res = h_classifier_pred(exp_fam, model, parameters, xs, num_classses, batch_size)
    r_res = r_reject_pred(exp_fam, model, parameters, xs, num_classses, rejection_cost, batch_size)
    cwr_res = torch.zeros(batch_size)
    for i in range(batch_size):
        if r_res[i] <=0 :
            cwr_res[i] = r_res[i]
        else:
            cwr_res[i] = h_res[i]
    return cwr_res.int()