# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import h5py
import imageio
import numpy as np
import torch as to
import matplotlib.pyplot as plt

import tvo
from tvo.exp import EVOConfig, ExpConfig, Training
from tvo.models import GaussianTVAE

from tvutil.prepost import (
    OverlappingPatches,
    MultiDimOverlappingPatches,
)

from params import get_args
from utils import (
    stdout_logger,
    set_pixels_to_nan,
    store_as_h5,
    get_epochs_from_every,
    eval_fn,
)

DEVICE = tvo.get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}


def inpainting():

    # get hyperparameters
    args = get_args()
    print("Argument list:")
    for k in sorted(vars(args), key=lambda s: s.lower()):
        print("{: <25} : {}".format(k, vars(args)[k]))

    # determine directories to save output
    os.makedirs(args.output_directory, exist_ok=True)
    data_file, training_file = (
        args.output_directory + "/image_patches.h5",
        args.output_directory + "/training.h5",
    )
    txt_file = args.output_directory + "/terminal.txt"
    sys.stdout = stdout_logger(txt_file)  # type: ignore
    print("Will write training output to {}.".format(training_file))
    print("Will write terminal output to {}".format(txt_file))

    # generate incomplete image and extract image patches
    clean = to.tensor(imageio.imread(args.clean_image_file), **dtype_device_kwargs)
    isrgb = clean.dim() == 3 and clean.shape[2] == 3
    incomplete = set_pixels_to_nan(clean, args.percentage)
    png_file = f"{args.output_directory}/incomplete-{args.percentage}missing.png"
    plt.imsave(png_file, incomplete)
    print(f"Wrote {png_file}")
    OVP = MultiDimOverlappingPatches if isrgb else OverlappingPatches
    ovp = OVP(incomplete, args.patch_height, args.patch_width, patch_shift=1)
    train_data = ovp.get().t()
    store_as_h5({"data": train_data}, data_file)

    isrgb = len(clean) == 3
    D = args.patch_height * args.patch_width * (3 if isrgb else 1)

    with h5py.File(data_file, "r") as f:
        N, D_read = f["data"].shape
        assert D == D_read

    print("Initializing model")

    # initialize model
    model = GaussianTVAE(
        shape=[
            D,
        ]
        + args.inner_net_shape,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        cycliclr_step_size_up=np.ceil(N / args.batch_size) * args.epochs_per_half_cycle,
        precision=PRECISION,
    )

    print("Initializing experiment")

    # define hyperparameters of the variational optimization
    estep_conf = EVOConfig(
        n_states=args.Ksize,
        n_parents=5,
        n_children=4,
        n_generations=1,
        parent_selection="fitness",
        crossover=False,
    )

    # setup the experiment
    reco_epochs = get_epochs_from_every(every=args.merge_every, total=args.no_epochs)
    exp_config = ExpConfig(
        batch_size=32,
        output=training_file,
        reco_epochs=reco_epochs,
        log_blacklist=[
            "train_lpj",
            "train_states",
            "train_subs",
            "train_reconstruction",
        ],
        log_only_latest_theta=True,
    )
    exp = Training(
        conf=exp_config, estep_conf=estep_conf, model=model, train_data_file=data_file
    )
    logger, trainer = exp.logger, exp.trainer
    # append the noisy image to the data logger
    logger.set_and_write(incomplete_image=incomplete)

    # run epochs
    for epoch, summary in enumerate(exp.run(args.no_epochs)):
        summary.print()

        # merge reconstructed image patches and generate reconstructed image
        merge = epoch in (reco_epochs + 1)
        assert hasattr(trainer, "train_reconstruction")
        reco = ovp.set_and_merge(trainer.train_reconstruction.t()) if merge else None

        # assess reconstruction quality in terms of PSNR
        psnr = eval_fn(clean, reco) if merge else None

        to_log = {"reco_image": reco, "psnr": psnr} if merge else None

        if to_log is not None:
            logger.append_and_write(**to_log)

        # visualize epoch
        if merge:
            psnr_str = f"{psnr:.2f}".replace(".", "_")
            png_file = f"{args.output_directory}/reco-epoch{epoch}-psnr{psnr_str}.png"
            plt.imsave(png_file, reco)
            print(f"Wrote {png_file}")

    print("Finished")


if __name__ == "__main__":
    inpainting()
