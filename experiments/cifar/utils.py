# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import time
import h5py
import datetime
import torch as to
from typing import Dict


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


def get_timestamp() -> str:
    """Return current time as YYYY-MM-DD-HH-MM-SS"""
    return datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d-%H-%M-%S")


def store_as_h5(output_name: str, **to_store: Dict[str, to.Tensor]) -> None:
    """Takes dictionary of tensors and writes to H5 file

    :param output_name: Full path of H5 file to write data to
    :param to_store: Dictionary of torch Tensors to write out
    """
    os.makedirs(os.path.split(output_name)[0], exist_ok=True)
    with h5py.File(output_name, "w") as f:
        for key, val in to_store.items():
            f.create_dataset(
                key, data=val if isinstance(val, float) else val.detach().cpu()
            )
    print(f"Wrote {output_name}")
