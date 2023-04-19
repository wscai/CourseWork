###### DO NOT CHANGE ######

import torch
import torch.nn.functional as F

import framework.exp_fam_model as efm

from typing import List, Tuple

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
})

class Multinomial(efm.ExponentialFamily):

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes

    def log_partition(self, eta: torch.Tensor) -> float:
        """ psi: Rm x X -> R
        """
        return torch.logsumexp(eta, axis=-1)
        #return torch.log(torch.sum(torch.exp(eta), axis=-1))

    def sufficient_statistic(self, y: torch.Tensor) -> torch.Tensor:
        """ u: Rn x X -> Rm
        """
        return F.one_hot(y, num_classes=self.n_classes).float()

def calc_fc_size(in_features: int,
                 out_features: int) -> Tuple[torch.Size, torch.Size]:
    return torch.Size((out_features, in_features)), torch.Size((out_features,))

def calc_fc(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
            in_features: int, out_features: int) -> torch.Tensor:
    return F.linear(x, weight, bias)

class FC_Model(efm.Model):

    def __init__(self, n_classes: int) -> None:
        self.fc1_args = (784, 128)
        self.fc2_args = (128, n_classes)

        self.fc_1_shape = calc_fc_size(*self.fc1_args)
        self.fc_2_shape = calc_fc_size(*self.fc2_args)

    def make_random_parameter(self) -> torch.Tensor:
        total = 0
        total += sum(map(lambda x: x.numel(), self.fc_1_shape))
        total += sum(map(lambda x: x.numel(), self.fc_2_shape))
        return torch.randn(total)

    def predict(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        self.idx = 0

        def to_w_b(shape: Tuple[torch.Size, torch.Size]
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
            w = theta[self.idx:self.idx + shape[0].numel()].view(shape[0])
            self.idx += shape[0].numel() 
            b = theta[self.idx:self.idx + shape[1].numel()].view(shape[1])
            self.idx += shape[1].numel() 
            return w, b

        x = torch.flatten(x, 1)
        x = calc_fc(x, *to_w_b(self.fc_1_shape), *self.fc1_args)
        x = torch.sigmoid(x)
        x = calc_fc(x, *to_w_b(self.fc_2_shape), *self.fc2_args)

        return F.log_softmax(x, dim=1)

def view_image(indices: List[int], dataset: torch.Tensor,
               per_row: int = 5) -> None:

    fig, ax = plt.subplots(1 + (len(indices) // per_row),
                           min(len(indices), per_row))

    for i, idx in enumerate(indices):
        data, label = dataset[idx]
        pixels = data.reshape((28, 28))

        if (len(indices) // per_row) == 0:
            ax[i].imshow(pixels, cmap='gray')
            ax[i].set_title(r'$Y$=' + f'{label}')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        else:
            ax[i // per_row, i % per_row].imshow(pixels, cmap='gray')
            ax[i // per_row, i % per_row].set_title(r'$Y$=' + f'{label}')
            ax[i // per_row, i % per_row].set_xticks([])
            ax[i // per_row, i % per_row].set_yticks([])

    for i in range(i+1, per_row * (1+(len(indices) // per_row))):
        ax[i // per_row, i % per_row].set_xticks([])
        ax[i // per_row, i % per_row].set_yticks([])
        ax[i // per_row, i % per_row].spines['top'].set_visible(False)
        ax[i // per_row, i % per_row].spines['right'].set_visible(False)
        ax[i // per_row, i % per_row].spines['bottom'].set_visible(False)
        ax[i // per_row, i % per_row].spines['left'].set_visible(False)
    
    plt.subplots_adjust(hspace=0.75, wspace=0.25)
    plt.show()

def view_image_pred(indices: List[int], dataset: torch.Tensor, model: efm.Model,
                    parameters: torch.Tensor, per_row: int = 5) -> None:
    fig, ax = plt.subplots(1 + (len(indices) // per_row), min(len(indices), per_row))

    for i, idx in enumerate(indices):
        data, label = dataset[idx]
        pred = torch.argmax(model.predict(data.unsqueeze(0), parameters))
        pixels = data.reshape((28, 28))

        if (len(indices) // per_row) == 0:
            ax[i].imshow(pixels, cmap='gray')
            ax[i].set_title(r'$Y$=' + f'{label}' + r'; $\hat{Y}$=' + f'{pred}')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        else:
            ax[i // per_row, i % per_row].imshow(pixels, cmap='gray')
            ax[i // per_row, i % per_row].set_title(r'$Y$=' + f'{label}' + r'; $\hat{Y}$=' + f'{pred}')
            ax[i // per_row, i % per_row].set_xticks([])
            ax[i // per_row, i % per_row].set_yticks([])

    if (len(indices) // per_row) != 0:
        for i in range(i+1, per_row * (1+(len(indices) // per_row))):
            ax[i // per_row, i % per_row].set_xticks([])
            ax[i // per_row, i % per_row].set_yticks([])
            ax[i // per_row, i % per_row].spines['top'].set_visible(False)
            ax[i // per_row, i % per_row].spines['right'].set_visible(False)
            ax[i // per_row, i % per_row].spines['bottom'].set_visible(False)
            ax[i // per_row, i % per_row].spines['left'].set_visible(False)
    
    plt.subplots_adjust(hspace=0.75, wspace=0.25)
    plt.show()

def view_image_cwr(indices: List[int], dataset: torch.Tensor, model: efm.Model,
                   parameters: torch.Tensor, cwr_preds: torch.Tensor,
                   per_row: int = 5) -> None:
    fig, ax = plt.subplots(1 + (len(indices) // per_row), min(len(indices), per_row))

    for i, idx in enumerate(indices):
        data, label = dataset[idx]
        pred = torch.argmax(model.predict(data.unsqueeze(0), parameters))
        cwr_pred = cwr_preds[idx] if cwr_preds[idx] > -1 else r'R'
        pixels = data.reshape((28, 28))

        if (len(indices) // per_row) == 0:
            ax[i].imshow(pixels, cmap='gray')
            ax[i].set_title(r'$\hat{Y}_{R}$=' + f'{cwr_pred}' + r'; $\hat{Y}$=' + f'{pred}')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        else:
            ax[i // per_row, i % per_row].imshow(pixels, cmap='gray')
            ax[i // per_row, i % per_row].set_title(r'$\hat{Y}_{R}$=' + f'{cwr_pred}' + r'; $\hat{Y}$=' + f'{pred}')
            ax[i // per_row, i % per_row].set_xticks([])
            ax[i // per_row, i % per_row].set_yticks([])

    if (len(indices) // per_row) != 0:
        for i in range(i+1, per_row * (1+(len(indices) // per_row))):
            ax[i // per_row, i % per_row].set_xticks([])
            ax[i // per_row, i % per_row].set_yticks([])
            ax[i // per_row, i % per_row].spines['top'].set_visible(False)
            ax[i // per_row, i % per_row].spines['right'].set_visible(False)
            ax[i // per_row, i % per_row].spines['bottom'].set_visible(False)
            ax[i // per_row, i % per_row].spines['left'].set_visible(False)
    
    plt.subplots_adjust(hspace=0.75, wspace=0.25)
    plt.show()

def view_image_rej(indices: List[int], dataset: torch.Tensor, model: efm.Model,
                   parameters: torch.Tensor, rej_score: torch.Tensor,
                   per_row: int = 5) -> None:

    fig, ax = plt.subplots(1 + (len(indices) // per_row), min(len(indices), per_row))

    for i, idx in enumerate(indices):
        data, label = dataset[idx]
        pred = torch.argmax(model.predict(data.unsqueeze(0), parameters))
        rej = rej_score[idx]
        pixels = data.reshape((28, 28))

        if (len(indices) // per_row) == 0:
            ax[i].imshow(pixels, cmap='gray')
            ax[i].set_title(r'$r(x)$=' + f'{rej:.2f}' + r'; $\hat{Y}$=' + f'{pred}')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].title.set_size(7)
        else:
            ax[i // per_row, i % per_row].imshow(pixels, cmap='gray')
            ax[i // per_row, i % per_row].set_title(r'$r(x)$=' + f'{rej:.2f}' + r'; $\hat{Y}$=' + f'{pred}')
            ax[i // per_row, i % per_row].set_xticks([])
            ax[i // per_row, i % per_row].set_yticks([])
            ax[i // per_row, i % per_row].title.set_size(7)

    if (len(indices) // per_row) != 0:
        for i in range(i+1, per_row * (1+(len(indices) // per_row))):
            ax[i // per_row, i % per_row].set_xticks([])
            ax[i // per_row, i % per_row].set_yticks([])
            ax[i // per_row, i % per_row].spines['top'].set_visible(False)
            ax[i // per_row, i % per_row].spines['right'].set_visible(False)
            ax[i // per_row, i % per_row].spines['bottom'].set_visible(False)
            ax[i // per_row, i % per_row].spines['left'].set_visible(False)
    
    plt.subplots_adjust(hspace=0.75, wspace=0.25)
    plt.show()

def zero_one_classification_loss(model: efm.Model, xs: torch.Tensor,
                                 ys: torch.Tensor, parameters: torch.Tensor) -> float:
    """ Calculate 0-1-loss by taking largest score prediction
    """
    class_pred = torch.argmax(model.predict(xs, parameters), axis=-1)
    return torch.mean((ys != class_pred).float())