# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Bars Test for TVAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output_directory",
        type=str,
        help="Directory to write training output and visualizations to (will be output/<TIMESTAMP> "
        "if not specified)",
        default=None,
    )

    parser.add_argument(
        "--H_gen",
        type=int,
        help="Number of bars used to generate data",
        default=8,
    )

    parser.add_argument(
        "--bar_amp",
        type=float,
        help="Bar amplitude",
        default=1.0,
    )

    parser.add_argument(
        "--no_data_points",
        type=int,
        help="Number of datapoints",
        default=500,
    )

    parser.add_argument(
        "--correlated",
        action="store_true",
        help="Whether to introduce correlations in generated bars combinations",
        default=False,
    )

    parser.add_argument(
        "--pi_gen",
        type=float,
        help="Sparsity used for data generation (defaults to 2/H if not specified)",
        default=None,
    )

    parser.add_argument(
        "--sigma2_gen",
        type=float,
        help="Noise level used for data generation",
        default=0.01,
    )

    parser.add_argument(
        "--H_train",
        type=int,
        help="Number of generative fields to learn (set to H_gen if not specified)",
        default=None,
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0001,
        help="minimal learning rate",
    )

    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.05,
        help="maximal learning rate",
    )

    parser.add_argument(
        "--epochs_per_half_cycle",
        type=int,
        default=10,
        help="epochs per half cycle of learning rate scheduler",
    )

    parser.add_argument(
        "--Ksize",
        type=int,
        help="Size of the K sets (i.e., S=|K|)",
        default=64,
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
        default=3,
    )

    parser.add_argument(
        "--no_children",
        type=int,
        help="Number of children to evolve per generation",
        default=2,
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
        default=400,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )

    return parser.parse_args()
