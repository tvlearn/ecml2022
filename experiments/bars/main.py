# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import time
import datetime
import numpy as np
import torch as to

from tvo import get_device
from tvo.exp import FullEMConfig, EVOConfig, ExpConfig, Training
from tvo.models import BSC, GaussianTVAE

from params import get_args
from utils import stdout_logger, get_singleton_means
from data import get_bars_gfs, get_W0, generate_data_and_write_to_h5
from viz import Visualizer

DEVICE = get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}


def bars_test():

    # get hyperparameters
    args = get_args()
    args_dict = vars(args)

    H_train = args.H_gen if args.H_train is None else args.H_train
    D = int((args.H_gen / 2) ** 2)
    args_dict["D"] = D

    assert (
        args.Ksize <= 2**H_train
    ), f"Ksize must be smaller/equal {2**H_train} ({args.Ksize} provided)"
    full_posteriors = args.Ksize == 2**H_train
    args_dict["full_posteriors"] = full_posteriors
    if full_posteriors:
        for k in (
            "no_parents",
            "no_children",
            "no_generations",
            "selection",
            "crossover",
        ):
            args_dict[k] = None

    print("Argument list:")
    for k in sorted(args_dict, key=lambda s: s.lower()):
        print("{: <25} : {}".format(k, args_dict[k]))

    # determine directories to save output
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        "%y-%m-%d-%H-%M-%S"
    )
    output_directory = (
        f"./out/{timestamp}" if args.output_directory is None else args.output_directory
    )
    os.makedirs(output_directory, exist_ok=True)
    data_file, training_file = (
        output_directory + "/data.h5",
        output_directory + "/training.h5",
    )
    txt_file = output_directory + "/terminal.txt"
    sys.stdout = stdout_logger(txt_file)  # type: ignore
    print("Will write training output to {}.".format(training_file))
    print("Will write terminal output to {}".format(txt_file))

    # generate data set
    gfs = get_bars_gfs(no_bars=args.H_gen, bar_amp=args.bar_amp, **dtype_device_kwargs)
    assert gfs.shape == (D, args.H_gen)
    pi_gen = args.pi_gen if args.pi_gen is not None else 2 / args.H_gen
    gen_model = (
        GaussianTVAE(
            W_init=[get_W0(args.H_gen, **dtype_device_kwargs), gfs.t()],
            b_init=[to.zeros(s, **dtype_device_kwargs) for s in [args.H_gen, D]],
            sigma2_init=to.tensor([args.sigma2_gen], **dtype_device_kwargs),
            pi_init=to.full((args.H_gen,), pi_gen, **dtype_device_kwargs),
            precision=PRECISION,
        )
        if args.correlated
        else BSC(
            H=args.H_gen,
            D=D,
            W_init=gfs,
            sigma2_init=to.tensor([args.sigma2_gen], **dtype_device_kwargs),
            pies_init=to.full((args.H_gen,), pi_gen, **dtype_device_kwargs),
            precision=PRECISION,
        )
    )
    compute_ll = args.H_gen <= 10  # too slow otherwise
    print(
        "Generating training dataset"
        + (
            "\nComputing log-likelihood of generative parameters given the data"
            if compute_ll
            else ""
        )
    )
    data, theta_gen, ll_gen = generate_data_and_write_to_h5(
        gen_model, data_file, args.no_data_points, compute_ll
    )

    # initialize model
    model = GaussianTVAE(
        shape=[D, H_train, H_train],
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        cycliclr_step_size_up=np.ceil(args.no_data_points / args.batch_size)
        * args.epochs_per_half_cycle,
        precision=PRECISION,
    )

    # define hyperparameters of the variational optimization
    estep_conf = (
        EVOConfig(
            n_states=args.Ksize,
            n_parents=args.no_parents,
            n_children=args.no_children,
            n_generations=args.no_generations,
            parent_selection=args.selection,
            crossover=args.crossover,
        )
        if not full_posteriors
        else FullEMConfig(n_latents=H_train)
    )

    # define general hyperparameters of the experiment
    exp_config = ExpConfig(batch_size=args.batch_size, output=training_file)
    print("Initializing experiment")
    exp = Training(
        conf=exp_config, estep_conf=estep_conf, model=model, train_data_file=data_file
    )

    # initialize visualizer
    print("Initializing visualizer")
    visualizer = Visualizer(
        viz_every=args.viz_every if args.viz_every is not None else args.no_epochs,
        output_directory=output_directory,
        datapoints=data[:15].cpu(),
        theta_gen={
            "pies": theta_gen["pies"].detach().cpu(),
            "sigma2": theta_gen["sigma2"].detach().cpu(),
            "W": gfs.cpu(),
        },
        L_gen=ll_gen,
        gif_framerate=args.gif_framerate,
    )

    # run epochs
    for epoch, summary in enumerate(exp.run(args.no_epochs)):
        summary.print()
        visualizer.process_epoch(
            epoch=epoch,
            F=summary._results["train_F"],
            theta={
                "pies": model.theta["pies"].clone().detach().cpu(),
                "sigma2": model.theta["sigma2"].clone().detach().cpu(),
                "W": get_singleton_means(model.theta).T,
            },
        )

    visualizer.finalize()
    print("Finished")


if __name__ == "__main__":
    bars_test()
