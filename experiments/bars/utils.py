# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import sys
import torch as to
from typing import Dict
from tvo import get_device
from tvo.models import GaussianTVAE
from tvo.variational import FullEM
from tvo.utils.model_protocols import Trainable


class stdout_logger(object):
    """Redirect print statements both to console and file

    Source: https://stackoverflow.com/a/14906787
    """

    def __init__(self, txt_file):
        self.terminal = sys.stdout
        self.log = open(txt_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def compute_full_log_marginals(model: Trainable, data: to.Tensor) -> to.Tensor:
    """Compute log(p_Theta(xVec^n)) summing over full latent space

    :param model: Generative model
    :param data: Data points, is (N, D)
    :return: Log marginals, is (N,)
    """
    return (
        to.logsumexp(
            model.log_joint(
                data=data,
                states=FullEM(
                    N=data.shape[0],
                    H=model.shape[1],
                    precision=model.config["precision"],
                ).K,
            ),
            dim=1,
        )
    ).detach()


def get_singleton_means(theta: Dict[str, to.Tensor]) -> to.Tensor:
    """Initialize TVAE model with parameters `theta` and compute NN output for NN input vectors
       corresponding to singleton states (only one active unit per unit vector).

    :param theta: Dictionary with TVAE model parameters
    :return: Decoded means
    """
    n_layers = len(tuple(k for k in theta.keys() if k.startswith("W_")))
    W = tuple(theta[f"W_{ind_layer}"].clone().detach() for ind_layer in range(n_layers))
    b = tuple(theta[f"b_{ind_layer}"].clone().detach() for ind_layer in range(n_layers))
    sigma2 = float(theta["sigma2"])
    H0 = W[0].shape[0]
    m = GaussianTVAE(W_init=W, b_init=b, sigma2_init=sigma2)
    singletons = to.eye(H0).to(get_device())
    means = m.forward(singletons).detach().cpu()
    D = W[-1].shape[-1]
    assert means.shape == (H0, D)
    return means
