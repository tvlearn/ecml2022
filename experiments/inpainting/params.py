# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse
import time
import datetime
import os


def get_args():
    p = argparse.ArgumentParser(
        description="Image Inpainting with Missing Completely at Random Pixels"
    )
    p.add_argument(
        "image",
        type=str,
        help="Benchmark image",
        choices=["house-50", "castle-50", "castle-80"],
    )
    args = p.parse_args()

    image_name, percentage = args.image.split("-")
    args.clean_image_file = "./img/{}.{}".format(
        image_name, {"house": "png", "castle": "jpg"}[image_name]
    )
    args.percentage = int(percentage)
    args.patch_height = args.patch_width = {
        "house-50": 12,
        "castle-50": 5,
        "castle-80": 12,
    }[args.image]
    args.Ksize = {"house-50": 64, "castle-50": 32, "castle-80": 64}[args.image]
    args.Ksize = {"house-50": 64, "castle-50": 32, "castle-80": 64}[args.image]
    args.inner_net_shape = [512, 512]
    args.min_lr = 0.0001
    args.max_lr = {"house-50": 0.01, "castle-50": 0.00125, "castle-80": 0.001}[
        args.image
    ]
    args.batch_size = 32
    args.no_epochs = 500
    args.epochs_per_half_cycle = 20
    args.merge_every = 20
    args.output_directory = "./out/{}".format(
        os.environ["SLURM_JOBID"]
        if "SLURM_JOBID" in os.environ
        else datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d-%H-%M-%S")
    )

    return args
