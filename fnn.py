import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn: torch.nn.Module = None,
                sequential: bool = False):
    """
    Utility function that adds batch norm, dropout and linear layer

    Arguments :
        n_in : Number of input neurons
        n_out : Number of output neurons
        bn : If there is a batch norm layer
        p : Bathc norm dropout rate
        act : Activation for the linear layer

    Returns :
        List of batch norm, dropout and linear layer

    """
    layers = [torch.nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(torch.nn.Dropout(p))
    layers.append(torch.nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    if sequential:
        return torch.nn.Sequential(layers)
    else:
        return layers


class fully_nn(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, bn: bool, drop: float, actn):
        """
        Simple Feedforward Net that accpest variable amount of layers

        Arguments :
            input_dim : Size of the input dimension
            hidden_dims : List containing size of all hidden layers
            num_classes : Number of classes
            bn : If there is a batch norm layer
            drop : dropout rate
        """
        super(fully_nn, self).__init__()

        layer_size = [input_dim] + hidden_dims + [num_classes]

        layers = []
        for n_in, n_out in zip(layer_size[:-1], layer_size[1:]):
            if n_out != hidden_dims:
                # add ReLU for every layer except last one
                layers += bn_drop_lin(n_in, n_out, bn, drop, actn)
            else:
                # don't add ReLU to last layer
                layers += bn_drop_lin(n_in, n_out, bn, drop, None)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Linear function 1
        out = self.layers(x)
        return out