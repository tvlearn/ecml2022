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
from tvo.utils.param_init import init_sigma2_default

from tvutil.prepost import (
    OverlappingPatches,
    MultiDimOverlappingPatches,
)

from params import get_args
from utils import stdout_logger, store_as_h5, eval_fn
from decoder import FCNet

DEVICE = tvo.get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}


def denoising():

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

    # generate noisy image and extract image patches
    clean = to.tensor(imageio.imread(args.clean_image_file)).to(**dtype_device_kwargs)
    isrgb = clean.dim() == 3 and clean.shape[2] == 3
    clean = clean / args.norm if args.norm is not None else clean
    sigma = args.sigma / args.norm if args.norm is not None else args.sigma
    noisy = clean + sigma * to.randn_like(clean)
    psnr = eval_fn(clean, noisy, data_range=args.norm if args.norm is not None else 255)
    psnr_str = f"{psnr:.2f}".replace(".", "_")
    png_file = f"{args.output_directory}/noisy-psnr{psnr_str}.png"
    plt.imsave(png_file, noisy.detach().cpu().numpy())
    print(f"Wrote {png_file}")
    OVP = MultiDimOverlappingPatches if isrgb else OverlappingPatches
    ovp = OVP(noisy, args.patch_height, args.patch_width, patch_shift=1)
    train_data = ovp.get().t()
    store_as_h5({"data": train_data}, data_file)

    D = args.patch_height * args.patch_width * (3 if isrgb else 1)
    with h5py.File(data_file, "r") as f:
        data = to.tensor(f["data"][...])
    N, D_read = data.shape
    assert D == D_read
    sigma2_init = (
        (to.sqrt(init_sigma2_default(data, **dtype_device_kwargs)) / args.norm).pow(2)
        if args.norm is not None
        else init_sigma2_default(data, **dtype_device_kwargs)
    )
    del data

    print("Initializing model")

    # initialize model
    tvae_decoder_arg = {
        "house": {
            "shape": [
                D,
            ]
            + args.inner_net_shape
        },
        "barbara": {
            "external_model": FCNet(
                shape=args.inner_net_shape[::-1]
                + [
                    D,
                ],
                activations=[
                    to.nn.LeakyReLU for _ in range(len(args.inner_net_shape) - 1)
                ]
                + [
                    to.nn.Identity,
                ],
            ).to(**dtype_device_kwargs)
        },
    }[args.image_name]

    model = GaussianTVAE(
        **tvae_decoder_arg,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        cycliclr_step_size_up=np.ceil(N / args.batch_size) * args.epochs_per_half_cycle,
        sigma2_init=sigma2_init,
        precision=PRECISION,
    )

    print("Initializing experiment")

    # define hyperparameters of the variational optimization
    estep_conf = EVOConfig(
        n_states=args.Ksize,
        n_parents=args.n_parents,
        n_children=args.n_children,
        n_generations=args.n_generations,
        parent_selection="fitness",
        crossover=False,
    )

    # setup the experiment
    exp_config = ExpConfig(
        batch_size=32,
        output=training_file,
        reco_epochs=to.arange(args.no_epochs),
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
    logger.set_and_write(noisy_image=noisy)

    # run epochs
    for epoch, summary in enumerate(exp.run(args.no_epochs)):
        summary.print()

        # merge reconstructed image patches and generate reconstructed image
        merge = ((epoch - 1) % args.merge_every) == 0
        assert hasattr(trainer, "train_reconstruction")
        reco = ovp.set_and_merge(trainer.train_reconstruction.t()) if merge else None

        # assess reconstruction quality in terms of PSNR
        psnr = (
            eval_fn(clean, reco, data_range=args.norm if args.norm is not None else 255)
            if merge
            else None
        )

        to_log = {"reco_image": reco, "psnr": psnr} if merge else None

        if to_log is not None:
            logger.append_and_write(**to_log)

        # visualize epoch
        if merge:
            psnr_str = f"{psnr:.2f}".replace(".", "_")
            png_file = f"{args.output_directory}/reco-epoch{epoch-1}-psnr{psnr_str}.png"
            plt.imsave(png_file, reco.detach().cpu().numpy())
            print(f"Wrote {png_file}")

    print("Finished")


if __name__ == "__main__":
    denoising()
