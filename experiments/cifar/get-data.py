# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import shutil
import torch as to
from torchvision import datasets, transforms
from utils import store_as_h5
from params import DATA_FILE


if __name__ == "__main__":

    data_directory = os.path.split(DATA_FILE)[0]

    if os.path.exists(DATA_FILE):
        sys.exit(f"Found {DATA_FILE}")

    train_dataset = datasets.CIFAR10(
        data_directory, train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.CIFAR10(
        data_directory, train=False, download=True, transform=transforms.ToTensor()
    )

    D = 3 * 32 * 32
    train_data = to.empty((len(train_dataset), D), dtype=to.float32)
    test_data = to.empty((len(test_dataset), D), dtype=to.float32)

    for i, d in enumerate(train_dataset):
        train_data[i] = d[0].reshape(D)
    for i, d in enumerate(test_dataset):
        test_data[i] = d[0].reshape(D)

    store_as_h5(DATA_FILE, train_data=train_data, val_data=test_data)

    try:
        os.remove(f"{data_directory}/cifar-10-python.tar.gz")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(f"{data_directory}/cifar-10-batches-py")
    except FileNotFoundError:
        pass
