# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import h5py
import numpy as np
import torch as to
from skimage.metrics import peak_signal_noise_ratio
from typing import Dict, Union


class stdout_logger(object):
    """Redirect print statements both to console and file

    Source: https://stackoverflow.com/a/14906787
    """

    def __init__(self, txt_file):
        self.terminal = sys.stdout
        self.log = open(txt_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def set_pixels_to_nan(image: to.Tensor, percentage: int) -> to.Tensor:
    set_to_nan = to.rand_like(image) < percentage / 100.0
    image_with_nans = image.clone().detach()
    image_with_nans[set_to_nan] = float("nan")
    print(f"Randomly set {percentage} % of pixels to nan")
    return image_with_nans


def store_as_h5(to_store_dict: Dict[str, to.Tensor], output_name: str) -> None:
    """Takes dictionary of tensors and writes to H5 file

    :param to_store_dict: Dictionary of torch Tensors
    :param output_name: Full path of H5 file to write data to
    """
    os.makedirs(os.path.split(output_name)[0], exist_ok=True)
    with h5py.File(output_name, "w") as f:
        for key, val in to_store_dict.items():
            f.create_dataset(
                key, data=val if isinstance(val, float) else val.detach().cpu()
            )
    print(f"Wrote {output_name}")


def get_epochs_from_every(every: int, total: int) -> to.Tensor:
    """Return indices corresponding to every Xth. Sequence starts at (every - 1) and always
    includes (total - 1) as last step.

    :param every: Step interval
    :param total: Total number of steps
    :return: Step indices

    Example:
    >>> print(get_epochs_from_every(2, 9))
    >>>
    """
    return to.unique(
        to.cat(
            (to.arange(start=every - 1, end=total, step=every), to.tensor([total - 1]))
        )
    )


def eval_fn(
    target: Union[np.ndarray, to.Tensor],
    reco: Union[np.ndarray, to.Tensor],
    data_range: int = 255,
) -> to.Tensor:
    return to.tensor(
        peak_signal_noise_ratio(
            target.detach().cpu().numpy() if isinstance(target, to.Tensor) else target,
            reco.detach().cpu().numpy() if isinstance(reco, to.Tensor) else reco,
            data_range=data_range,
        )
    )
