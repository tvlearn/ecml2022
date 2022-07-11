# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse

DATA_FILE = "./data/cifar.h5"


def get_args():
    parser = argparse.ArgumentParser(
        description="Training and testing on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        help="Directory to write H5 training output and visualizations to"
        "(will be output/<TIMESTAMP> if not specified)",
        default=None,
    )

    parser.add_argument(
        "--hidden_shape",
        type=int,
        nargs="+",
        help="Decoder shape excluding no. obsersables, H0-H1-...",
        default=[32, 512],
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0004,
        help="minimal learning rate",
    )

    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.009,
        help="maximal learning rate",
    )

    parser.add_argument(
        "--epochs_per_half_cycle",
        type=int,
        default=80,
        help="epochs per half cycle of learning rate scheduler",
    )

    parser.add_argument(
        "--Ksize",
        type=int,
        help="Size of the K sets (i.e., S=|K|)",
        default=20,
    )

    parser.add_argument(
        "--selection",
        type=str,
        help="Selection operator",
        choices=["fitness", "uniform"],
        default="fitness",
    )

    parser.add_argument(
        "--crossover",
        action="store_true",
        help="Whether to apply crossover. Must be False if no_children is specified",
        default=False,
    )

    parser.add_argument(
        "--no_parents",
        type=int,
        help="Number of parental states to select per generation",
        default=15,
    )

    parser.add_argument(
        "--no_children",
        type=int,
        help="Number of children to evolve per generation",
        default=10,
    )

    parser.add_argument(
        "--no_generations",
        type=int,
        help="Number of generations to evolve",
        default=1,
    )

    parser.add_argument(
        "--no_epochs",
        type=int,
        help="Number of epochs to train",
        default=500,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size",
    )

    return parser.parse_args()
