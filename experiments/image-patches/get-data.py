# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import imageio
import tifffile
import numpy as np
import torch as to
from typing import Iterable, Union
from tvutil.prepost import apply_zca_whitening, extract_random_patches
from utils import store_as_h5, get_timestamp
from params import get_data_args


def prepare_training_dataset(
    image_file: str,
    patch_size: Union[int, Iterable[int]],
    no_patches: int,
    perc_highest_amps: float = 0.02,
    perc_lowest_vars: float = None,
) -> to.Tensor:
    """Read image from file, optionally rescale image size and return as to.Tensor

    :param image_file: Full path to image file (.png, .jpg, ...)
    :param patch_size: Patch size
    :param no_patches: Number of patches to extract
    :param perc_highest_amps: Percentage of highest image amplitudes to clamp
    :param perc_lowest_vars: Percentage of patches with lowest variance to clamp
    :return: Whitened image patches as torch tensor
    """
    imread = (
        tifffile.imread
        if os.path.splitext(image_file)[1] == ".tiff"
        else imageio.imread
    )
    img = imread(image_file)
    print("Read {}".format(image_file))
    isrgb = np.ndim(img) == 3 and img.shape[2] == 3
    isgrey = np.ndim(img) == 2
    assert isrgb or isgrey, "Expect img image to be either RGB or grey"

    # Clamp highest amplitudes
    if perc_highest_amps is not None:
        img = np.clip(
            img,
            np.min(img),
            np.sort(img.flatten())[::-1][int(perc_highest_amps * img.size)],
        )

    # Extract image patches and whiten
    patches = extract_random_patches(
        images=img[None, :, :, :] if isrgb else img[None, :, :, None],
        patch_size=patch_size,
        no_patches=no_patches,
    )
    whitened = apply_zca_whitening(patches)

    # Discard patches with lowest variance (assuming these do not contain significant structure)
    if perc_lowest_vars is not None:
        whitened = whitened[
            np.argsort(np.var(whitened, axis=1))[
                int(perc_lowest_vars * no_patches) :  # noqa
            ]
        ]

    whitened_to = to.from_numpy(whitened)

    return whitened_to


if __name__ == "__main__":
    args = get_data_args()
    data = prepare_training_dataset(
        image_file=args.image_file, patch_size=args.patch_size, no_patches=args.N
    )
    N, D = data.shape
    patch_size = "x".join(map(str, args.patch_size))
    print(f"Extract N={N} patches of shape {patch_size}")
    image_name = os.path.split(args.image_file)[1].split(".")[0]
    timestamp_str = f"-{get_timestamp()}" if args.timestamp else ""
    store_as_h5(
        output_name=f"./data/{image_name}-P{patch_size}-N{args.N}{timestamp_str}.h5",
        data=data,
    )
