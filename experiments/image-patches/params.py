# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse


data_parser = argparse.ArgumentParser(add_help=False)
data_parser.add_argument(
    "--image_file",
    type=str,
    help="Full path to image to extract patches from (png, jpg, ... file)",
    default="./data/image.tiff",
)

data_parser.add_argument(
    "-N",
    "--no_patches",
    dest="N",
    type=int,
    help="Number of patches to extract",
    default=1000,
)

data_parser.add_argument(
    "--timestamp",
    action="store_true",
    help="Whether to append timestamp to file name of written out training data set",
    default=False,
)

patch_size_parser = argparse.ArgumentParser(add_help=False)
patch_size_parser.add_argument(
    "--patch_size",
    type=int,
    nargs=2,
    help="Patch (height, width)",
    default=[8, 8],
)

train_io_parser = argparse.ArgumentParser(add_help=False)
train_io_parser.add_argument(
    "--data_file",
    type=str,
    help="HDF5 file with image patches",
    default="./data/image-P8x8-N1000.h5",
)
train_io_parser.add_argument(
    "--output_directory",
    type=str,
    help="Directory to write H5 training output and visualizations to (will be output/<TIMESTAMP> "
    "if not specified)",
    default=None,
)

param_init_parser = argparse.ArgumentParser(add_help=False)
param_init_parser.add_argument(
    "--init_file",
    type=str,
    help="Full path to HDF5 file with K and Theta to use for initializing model",
    required=True,
)

Ksize_parser = argparse.ArgumentParser(add_help=False)
Ksize_parser.add_argument(
    "--Ksize",
    type=int,
    help="Size of the K sets (i.e., S=|K|)",
    default=30,
)
Ksize_parser.add_argument(
    "-H",
    type=int,
    help="Latent vector size H",
    default=10,
)

tvae_parser = argparse.ArgumentParser(add_help=False)
tvae_parser.add_argument(
    "--min_lr",
    type=float,
    default=0.0001,
    help="minimal learning rate",
)
tvae_parser.add_argument(
    "--max_lr",
    type=float,
    default=0.1,
    help="maximal learning rate",
)

tvae_parser.add_argument(
    "--epochs_per_half_cycle",
    type=int,
    default=10,
    help="epochs per half cycle of learning rate scheduler",
)


variational_parser = argparse.ArgumentParser(add_help=False)
variational_parser.add_argument(
    "--selection",
    type=str,
    help="Selection operator",
    choices=["fitness", "uniform"],
    default="fitness",
)

variational_parser.add_argument(
    "--crossover",
    action="store_true",
    help="Whether to apply crossover. Must be False if no_children is specified",
    default=False,
)

variational_parser.add_argument(
    "--no_parents",
    type=int,
    help="Number of parental states to select per generation",
    default=20,
)

variational_parser.add_argument(
    "--no_children",
    type=int,
    help="Number of children to evolve per generation",
    default=2,
)

variational_parser.add_argument(
    "--no_generations",
    type=int,
    help="Number of generations to evolve",
    default=1,
)


experiment_parser = argparse.ArgumentParser(add_help=False)
experiment_parser.add_argument(
    "--no_epochs",
    type=int,
    help="Number of epochs to train",
    default=200,
)

experiment_parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="batch size",
)


def get_data_args():
    parser = argparse.ArgumentParser(
        prog="Build data set of whitened image patches extracted from image file",
        parents=[data_parser, patch_size_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args()


def get_train_args():
    parser = argparse.ArgumentParser(prog="Training on whitened image patches")
    parsers = parser.add_subparsers(
        help="Select program", dest="program", required=True
    )
    parsers.add_parser(
        "bsc-from-scratch",
        help="Train randomly initialized BSC model",
        parents=[
            train_io_parser,
            patch_size_parser,
            Ksize_parser,
            variational_parser,
            experiment_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parsers.add_parser(
        "bsc-from-init",
        help="Initialize BSC model from file and train",
        parents=[
            param_init_parser,
            train_io_parser,
            patch_size_parser,
            variational_parser,
            experiment_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parsers.add_parser(
        "tvae-from-init",
        help="Initialize TVAE model from file and train",
        parents=[
            param_init_parser,
            train_io_parser,
            patch_size_parser,
            variational_parser,
            tvae_parser,
            experiment_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return parser.parse_args()
