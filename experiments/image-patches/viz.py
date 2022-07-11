# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import numpy as np
import torch as to
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List


LABELSIZE = 14


def viz_bound(png_file: str, bounds: List[float], labelsize: int = LABELSIZE):
    _, ax = plt.subplots(1)
    ax.plot(bounds, "b")
    ax.set_xlabel("Epoch", fontsize=labelsize)
    ax.set_ylabel(r"$\mathcal{F}(\mathcal{K},\Theta)$", fontsize=labelsize)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    os.makedirs(os.path.split(png_file)[0], exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_file)
    print(f"Wrote {png_file}")
    plt.close()


def viz_pies(png_file: str, pies: to.Tensor, labelsize: int = LABELSIZE):
    _, ax = plt.subplots(1)
    ax.plot(
        np.arange(1, len(pies) + 1),
        pies,
        "b",
        linestyle="none",
        marker=".",
        markersize=4,
    )
    ax.set_ylabel(r"$\pi_h$", fontsize=labelsize)
    ax.set_xlabel(r"$h$", fontsize=labelsize)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.text(
        0.9,
        0.9,
        r"$\sum_h \pi_h$ = " + "{:.2f}".format(pies.sum()),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    os.makedirs(os.path.split(png_file)[0], exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_file)
    print(f"Wrote {png_file}")
    plt.close()
