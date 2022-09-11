# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse
import time
import datetime
import os


def get_args():
    p = argparse.ArgumentParser(description="Gaussian Denoising")
    p.add_argument(
        "image",
        type=str,
        help="Benchmark",
        choices=["house-15", "house-25", "house-50", "barbara-25"],
    )
    args = p.parse_args()

    args.image_name, sigma = args.image.split("-")
    args.norm = {
        "house-15": None,
        "house-25": None,
        "house-50": None,
        "barbara-25": 255,
    }[args.image]
    args.clean_image_file, args.sigma = f"./img/{args.image_name}.png", int(sigma)
    args.patch_height = args.patch_width = {
        "house-15": 8,
        "house-25": 8,
        "house-50": 12,
        "barbara-25": 8,
    }[args.image]
    args.Ksize = {
        "house-15": 200,
        "house-25": 200,
        "house-50": 64,
        "barbara-25": 100,
    }[args.image]
    args.n_parents = {
        "house-15": 10,
        "house-25": 10,
        "house-50": 5,
        "barbara-25": 5,
    }[args.image]
    args.n_children = {
        "house-15": 9,
        "house-25": 9,
        "house-50": 4,
        "barbara-25": 4,
    }[args.image]
    args.n_generations = {
        "house-15": 4,
        "house-25": 4,
        "house-50": 1,
        "barbara-25": 1,
    }[args.image]
    args.inner_net_shape = {
        "house-15": [64, 64],
        "house-25": [64, 64],
        "house-50": [512, 512],
        "barbara-25": [500, 500, 50],
    }[args.image]
    args.min_lr = 0.0001
    args.max_lr = {
        "house-15": 0.01,
        "house-25": 0.01,
        "house-50": 0.05,
        "barbara-25": 0.001,
    }[args.image]
    args.batch_size = {
        "house-15": 32,
        "house-25": 32,
        "house-50": 32,
        "barbara-25": 512,
    }[args.image]
    args.no_epochs = 500
    args.epochs_per_half_cycle = {
        "house-15": 10,
        "house-25": 10,
        "house-50": 10,
        "barbara-25": 80,
    }[args.image]
    args.merge_every = 20
    args.output_directory = "./out/{}".format(
        os.environ["SLURM_JOBID"]
        if "SLURM_JOBID" in os.environ
        else datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d-%H-%M-%S")
    )

    return args
