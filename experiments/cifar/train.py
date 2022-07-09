# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import h5py
import numpy as np
import torch as to
from tvo import get_device
from tvo.exp import EVOConfig, ExpConfig, Training
from tvo.models import GaussianTVAE
from tvo.utils.param_init import init_sigma2_default
from params import get_args, DATA_FILE
from utils import stdout_logger, get_timestamp
from decoder import FCNet

DEVICE = get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}


def cifar():

    # get hyperparameters
    args = get_args()
    print("Argument list:")
    for k in sorted(vars(args), key=lambda s: s.lower()):
        print("{: <25} : {}".format(k, vars(args)[k]))

    # determine directories to save output
    output_directory = (
        f"./out/{get_timestamp()}"
        if args.output_directory is None
        else args.output_directory
    )
    os.makedirs(output_directory, exist_ok=True)
    training_file = output_directory + "/training.h5"
    txt_file = output_directory + "/terminal.txt"
    sys.stdout = stdout_logger(txt_file)  # type: ignore
    print("Will write training output to {}".format(training_file))
    print("Will write terminal output to {}".format(txt_file))

    # read data set
    with h5py.File(DATA_FILE, "r") as f:
        data = to.tensor(f["train_data"][...])
    N, D = data.shape
    sigma2_init = init_sigma2_default(data, **dtype_device_kwargs)
    del data

    # initialize model
    decoder = FCNet(
        shape=args.hidden_shape
        + [
            D,
        ],
        activations=[to.nn.LeakyReLU for _ in range(len(args.hidden_shape) - 1)]
        + [
            to.nn.Sigmoid,
        ],
    ).to(**dtype_device_kwargs)

    model = GaussianTVAE(
        external_model=decoder,
        sigma2_init=sigma2_init,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        cycliclr_step_size_up=np.ceil(N / args.batch_size) * args.epochs_per_half_cycle,
        optimizer=to.optim.Adam(decoder.parameters(), lr=args.min_lr),
        precision=PRECISION,
    )

    # define EVO hyperparameters
    estep_conf = EVOConfig(
        n_states=args.Ksize,
        n_parents=args.no_parents,
        n_children=args.no_children,
        n_generations=args.no_generations,
        parent_selection=args.selection,
        crossover=args.crossover,
    )

    # setup experiment
    exp_config = ExpConfig(
        batch_size=args.batch_size,
        output=training_file,
        log_blacklist=["train_lpj", "train_states", "valid_lpj", "valid_states"],
        log_only_latest_theta=True,
    )
    exp = Training(
        conf=exp_config,
        estep_conf=estep_conf,
        model=model,
        train_data_file=DATA_FILE,
        val_data_file=DATA_FILE,
    )

    # run epochs
    for summary in exp.run(args.no_epochs):
        summary.print()

    print("Finished")


if __name__ == "__main__":
    cifar()
