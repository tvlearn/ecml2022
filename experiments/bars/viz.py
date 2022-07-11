# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import math
import numpy as np
import torch as to
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from tvutil.viz import make_grid_with_black_boxes_and_white_background


class Visualizer(object):
    def __init__(
        self,
        output_directory,
        train_samples,
        theta_gen,
        L_gen=None,
        test_samples=None,
        test_marginals_gen=None,
        sort_acc_to_desc_priors=True,
    ):
        self._output_directory = output_directory
        self._train_samples = train_samples
        self._theta_gen = theta_gen
        self._L_gen = L_gen
        self._test_samples = test_samples
        self._test_marginals_gen = test_marginals_gen
        self._sort_acc_to_desc_priors = sort_acc_to_desc_priors
        self._cmap_weights = plt.cm.jet
        self._cmap_train_samples = (
            plt.cm.gray if train_samples.dtype == to.uint8 else plt.cm.jet
        )
        self._labelsize = 10
        self._legendfontsize = 8

        self._memory = {k: [] for k in ["F", "sigma2"]}

        positions = (
            {
                "train_samples": [0.0, 0.0, 0.07, 0.94],
                "W_gen": [0.08, 0.0, 0.1, 0.94],
                "W": [0.2, 0.0, 0.1, 0.94],
                "F": [0.4, 0.81, 0.58, 0.17],
                "sigma2": [0.4, 0.62, 0.58, 0.17],
                "pies": [0.4, 0.33, 0.58, 0.17],
                "test_samples": [0.34, 0.07, 0.75, 0.11],
            }
            if test_samples is not None
            else {
                "train_samples": [0.0, 0.0, 0.07, 0.94],
                "W_gen": [0.08, 0.0, 0.1, 0.94],
                "W": [0.2, 0.0, 0.1, 0.94],
                "F": [0.4, 0.76, 0.58, 0.23],
                "sigma2": [0.4, 0.43, 0.58, 0.23],
                "pies": [0.4, 0.1, 0.58, 0.23],
            }
        )

        self._fig = plt.figure()
        self._axes = {k: self._fig.add_axes(v) for k, v in positions.items()}
        self._handles = {k: None for k in positions}
        for k in theta_gen.keys():
            self._handles["{}_gen".format(k)] = None
        self._handles["L_gen"] = None
        self._viz_train_samples()
        self._viz_gen_weights()
        if test_samples is not None:
            self._viz_test_samples()

    def _viz_train_samples(self):
        assert "train_samples" in self._axes
        ax = self._axes["train_samples"]
        train_samples = self._train_samples
        N, D = train_samples.shape
        (
            grid,
            cmap,
            vmin,
            vmax,
            scale_suff,
        ) = make_grid_with_black_boxes_and_white_background(
            images=train_samples.numpy().reshape(
                N, 1, int(math.sqrt(D)), int(math.sqrt(D))
            ),
            nrow=int(math.ceil(N / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=self._cmap_train_samples == plt.cm.jet,
            cmap=self._cmap_train_samples,
            eps=0.02,
        )

        self._handles["train_samples"] = ax.imshow(
            np.squeeze(grid), interpolation="none"
        )
        ax.axis("off")

        self._handles["train_samples"].set_cmap(cmap)
        self._handles["train_samples"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(r"$\vec{y}^{\,(n)}$", fontsize=self._labelsize)

    def _viz_gen_weights(self):
        assert "W_gen" in self._axes
        ax = self._axes["W_gen"]
        W_gen = self._theta_gen["W"]
        D, H = W_gen.shape
        (
            grid,
            cmap,
            vmin,
            vmax,
            scale_suff,
        ) = make_grid_with_black_boxes_and_white_background(
            images=W_gen.numpy()
            .copy()
            .T.reshape(H, 1, int(math.sqrt(D)), int(math.sqrt(D))),
            nrow=int(math.ceil(H / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=True,
            cmap=self._cmap_weights,
            eps=0.02,
        )

        if self._handles["W_gen"] is None:
            self._handles["W_gen"] = ax.imshow(np.squeeze(grid), interpolation="none")
            ax.axis("off")
        else:
            self._handles["W_gen"].set_data(np.squeeze(grid))
        self._handles["W_gen"].set_cmap(cmap)
        self._handles["W_gen"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(r"$\vec{W}_{h}^{\mathrm{gen}}$", fontsize=self._labelsize)

    def _viz_weights(self, epoch, W, inds_sort=None):
        assert "W" in self._axes
        ax = self._axes["W"]
        D, H = W.shape
        W = W.numpy()[:, inds_sort] if inds_sort is not None else W.numpy().copy()
        (
            grid,
            cmap,
            vmin,
            vmax,
            scale_suff,
        ) = make_grid_with_black_boxes_and_white_background(
            images=W.T.reshape(H, 1, int(math.sqrt(D)), int(math.sqrt(D))),
            nrow=int(math.ceil(H / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=True,
            cmap=self._cmap_weights,
            eps=0.02,
        )

        if self._handles["W"] is None:
            self._handles["W"] = ax.imshow(np.squeeze(grid), interpolation="none")
            ax.axis("off")
        else:
            self._handles["W"].set_data(np.squeeze(grid))
        self._handles["W"].set_cmap(cmap)
        self._handles["W"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(
            r"$\vec{\mu}(\vec{e}_h;W)$" + "@{}".format(epoch), fontsize=self._labelsize
        )

    def _viz_free_energy(self):
        memory = self._memory
        assert "F" in memory
        assert "F" in self._axes
        ax = self._axes["F"]
        xdata = to.arange(1, len(memory["F"]) + 1)
        ydata_learned = memory["F"]
        if self._handles["F"] is None:
            (self._handles["F"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                label=r"$\mathcal{F}(\Phi,\Theta)$",
            )
            # ax.set_xlabel("Epoch", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xticklabels(())
            add_legend = True
        else:
            self._handles["F"].set_xdata(xdata)
            self._handles["F"].set_ydata(ydata_learned)
            ax.relim()
            ax.autoscale_view()
            ax.set_ylim(
                [ydata_learned[int(0.1 * len(ydata_learned))], ax.get_ylim()[1]]
            )
            add_legend = False

        if self._L_gen is not None:
            ydata_gen = self._L_gen * np.ones_like(ydata_learned)
            if self._handles["L_gen"] is None:
                (self._handles["L_gen"],) = ax.plot(
                    xdata,
                    ydata_gen,
                    "b--",
                    label=r"$\mathcal{L}(\Theta^{\mathrm{gen}})$",
                )
            else:
                self._handles["L_gen"].set_xdata(xdata)
                self._handles["L_gen"].set_ydata(ydata_gen)

        if add_legend:
            ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_pies(self, epoch, pies, inds_sort=None):
        assert "pies" in self._axes
        ax = self._axes["pies"]
        xdata = to.arange(1, len(pies) + 1)
        ydata_learned = pies[inds_sort] if inds_sort is not None else pies
        if self._handles["pies"] is None:
            (self._handles["pies"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                linestyle="none",
                marker=".",
                markersize=4,
                label=r"$\pi_h$ @ {}".format(epoch),
            )
            ax.set_xlabel(r"$h$", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            self._handles["pies"].set_xdata(xdata)
            self._handles["pies"].set_ydata(ydata_learned)
            self._handles["pies"].set_label(r"$\pi_h$ @ {}".format(epoch))
            ax.relim()
            ax.autoscale_view()

        ydata_gen = self._theta_gen["pies"]
        xdata = to.arange(1, len(ydata_gen) + 1)
        if self._handles["pies_gen"] is None:
            (self._handles["pies_gen"],) = ax.plot(
                xdata,
                ydata_gen,
                "b",
                linestyle="none",
                marker="o",
                fillstyle=Line2D.fillStyles[-1],
                markersize=4,
                label=r"$\pi_h^{\mathrm{gen}}$",
            )
        else:
            self._handles["pies_gen"].set_xdata(xdata)
            self._handles["pies_gen"].set_ydata(ydata_gen)

        ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_sigma2(self):
        memory = self._memory
        assert "sigma2" in memory
        assert "sigma2" in self._axes
        ax = self._axes["sigma2"]
        xdata = to.arange(1, len(memory["sigma2"]) + 1)
        ydata_learned = np.squeeze(to.tensor(memory["sigma2"]))
        if self._handles["sigma2"] is None:
            (self._handles["sigma2"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                label=r"$\sigma^2$",
            )
            ax.set_xlabel("Epoch", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            add_legend = True
        else:
            self._handles["sigma2"].set_xdata(xdata)
            self._handles["sigma2"].set_ydata(ydata_learned)
            ax.relim()
            ax.autoscale_view()
            add_legend = True

        ydata_gen = self._theta_gen["sigma2"] * np.ones_like(ydata_learned)
        if self._handles["sigma2_gen"] is None:
            (self._handles["sigma2_gen"],) = ax.plot(
                xdata,
                ydata_gen,
                "b--",
                label=r"$(\sigma^{\mathrm{gen}})^2$",
            )
        else:
            self._handles["sigma2_gen"].set_xdata(xdata)
            self._handles["sigma2_gen"].set_ydata(ydata_gen)

        if add_legend:
            ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_test_samples(self):
        assert "test_samples" in self._axes
        assert self._test_samples is not None
        ax = self._axes["test_samples"]
        test_samples = self._test_samples
        N, D = test_samples.shape

        (
            grid,
            cmap,
            vmin,
            vmax,
            scale_suff,
        ) = make_grid_with_black_boxes_and_white_background(
            images=test_samples.numpy().reshape(
                N, 1, int(math.sqrt(D)), int(math.sqrt(D))
            ),
            nrow=N,
            surrounding=4,
            padding=20,
            repeat=20,
            global_clim=True,
            sym_clim=self._cmap_train_samples == plt.cm.jet,
            cmap=self._cmap_train_samples,
            eps=0.02,
        )

        self._handles["test_samples"] = ax.imshow(
            np.squeeze(grid), interpolation="none"
        )
        ax.axis("off")

        self._handles["test_samples"].set_cmap(cmap)
        self._handles["test_samples"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title("Test samples", fontsize=self._labelsize)

        text_kwargs = {
            "horizontalalignment": "center",
            "verticalalignment": "center",
            "transform": ax.transAxes,
            "fontsize": 9,
        }
        ax.text(
            -0.11,
            -0.10,
            r"$\log p_{\Theta}(\vec{x})$",
            **text_kwargs,
        )
        ax.text(
            -0.11,
            -0.45,
            r"$\log p_{\Theta^{\mathrm{gen}}}(\vec{x})$",
            **text_kwargs,
        )

        for n in range(N):
            xpos = 0.98 * n / N + 1 / 2 / N
            self._handles[f"marginals-{n}"] = ax.text(
                xpos,
                -0.10,
                "",
                **text_kwargs,
            )
            self._handles[f"marginals-{n}-gen"] = ax.text(
                xpos,
                -0.45,
                f"{self._test_marginals_gen[n]:.1f}",
                **text_kwargs,
            )

    def _viz_marginals(self, marginals):
        assert self._test_samples is not None
        N = self._test_samples.shape[0]
        for n in range(N):
            assert f"marginals-{n}" in self._handles
            self._handles[f"marginals-{n}"].set_text(f"{marginals[n]:.1f}")

    def _viz_epoch(self, epoch, F, theta, marginals=None):
        pies = theta["pies"]
        inds_sort = (
            to.argsort(pies, descending=True) if self._sort_acc_to_desc_priors else None
        )
        self._viz_weights(epoch, theta["W"])
        self._viz_pies(epoch, pies, inds_sort=inds_sort)
        self._viz_free_energy()
        self._viz_sigma2()
        if marginals is not None:
            self._viz_marginals(marginals)

    def process_epoch(self, epoch, F, theta, marginals=None):
        memory = self._memory
        [
            v.append(
                np.squeeze({"F": F, **{k_: v_.clone() for k_, v_ in theta.items()}}[k])
            )
            for k, v in memory.items()
        ]
        self._viz_epoch(epoch, F, theta, marginals)
        self._save_epoch(epoch)

    def _save_epoch(self, epoch):
        output_directory = self._output_directory
        png_file = "{}/training.png".format(output_directory)
        plt.savefig(png_file)
        print("\tWrote " + png_file)

    def finalize(self):
        plt.close()
