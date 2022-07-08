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
from tvo.models import BSC, GaussianTVAE
from tvutil.viz import save_grid
from params import get_train_args
from utils import stdout_logger, get_timestamp, get_singleton_means
from viz import viz_bound, viz_pies

DEVICE = get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}


def natural_image_patches():

    # get hyperparameters
    args = get_train_args()
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
    print("Will write training output to {}.".format(training_file))
    print("Will write terminal output to {}".format(txt_file))

    # read data set
    with h5py.File(args.data_file, "r") as f:
        N, D = f["data"].shape
    assert np.prod(args.patch_size) == D

    # initialize model
    W_init, pies_init, sigma2_init, K_init = None, None, None, None
    if "from-init" in args.program:
        with h5py.File(args.init_file, "r") as f:
            D_init = f["theta/W"].shape[1]
            N_init = f["train_states"].shape[0]
            assert (
                D == D_init
            ), f"No. observables mismatch.\n{args.data_file} has {D}\n{args.init_file} has {D_init}"
            assert (
                N == N_init
            ), f"No. data points mismatch.\n{args.data_file} has {N}\n{args.init_file} has {N_init}"
            W_init = to.tensor(f["theta/W"][-1]).to(**dtype_device_kwargs)
            pies_init = to.tensor(f["theta/pies"][-1]).to(**dtype_device_kwargs)
            sigma2_init = to.tensor(f["theta/sigma2"][-1]).to(**dtype_device_kwargs)
            K_init = to.tensor(f["train_states"][...]).to(device=DEVICE)
        print(f"Loaded K and Theta from {args.init_file}")

    if args.program == "bsc-from-scratch":
        model = BSC(H=args.H, D=D, precision=PRECISION)
    elif args.program == "bsc-from-init":
        assert (
            W_init is not None and pies_init is not None and sigma2_init is not None
        ), "Theta not initialized"
        model = BSC(
            H=W_init.shape[1],
            D=D,
            precision=PRECISION,
            W_init=W_init,
            pies_init=pies_init,
            sigma2_init=sigma2_init,
        )
    elif args.program == "tvae-from-init":
        assert (
            W_init is not None and pies_init is not None and sigma2_init is not None
        ), "Theta not initialized"
        model = GaussianTVAE(
            W_init=[to.eye(W_init.shape[1]), W_init.t()],
            b_init=[to.zeros(s, **dtype_device_kwargs) for s in [W_init.shape[1], D]],
            pi_init=pies_init,
            sigma2_init=sigma2_init,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            cycliclr_step_size_up=np.ceil(N / args.batch_size)
            * args.epochs_per_half_cycle,
        )

    # define EVO hyperparameters
    estep_conf = EVOConfig(
        n_states=K_init.shape[1] if "from-init" in args.program else args.Ksize,
        n_parents=args.no_parents,
        n_children=args.no_children,
        n_generations=args.no_generations,
        parent_selection=args.selection,
        crossover=args.crossover,
    )

    # setup experiment
    exp_config = ExpConfig(batch_size=args.batch_size, output=training_file)
    exp = Training(
        conf=exp_config,
        estep_conf=estep_conf,
        model=model,
        train_data_file=args.data_file,
    )
    if "from-init" in args.program:
        assert K_init is not None, "K not initialized"
        exp.train_states.K[:] = K_init

    # run epochs
    bound_history = []
    for summary in exp.run(args.no_epochs):
        summary.print()
        inds_pies_sort = to.argsort(model.theta["pies"].detach())
        pies_sorted = model.theta["pies"].detach().cpu()[inds_pies_sort]
        gfs_sorted = (
            get_singleton_means(model.theta)[inds_pies_sort]
            if "tvae" in args.program
            else model.theta["W"].detach().cpu().t()[inds_pies_sort]
        )
        save_grid(
            png_file=f"{output_directory}/gfs-sorted.png",
            images=gfs_sorted.reshape(-1, 1, *args.patch_size).numpy(),
            nrow=32,
            repeat=8,
            global_clim=False,
            sym_clim=True,
        )

        bound_history.append(summary._results["train_F"])
        viz_bound(f"{output_directory}/free_energy.png", bound_history)
        viz_pies(f"{output_directory}/priors-sorted.png", pies_sorted)

    print("Finished")


if __name__ == "__main__":
    natural_image_patches()
