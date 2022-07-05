#   Overlap-based blockprocessing for visual data
#
#   Created on July 20, 2017
#   By Jakob Drefs <jakob.heinrich.drefs AT uni-oldenburg.de>
#   Adapted for Python3 by E. Guiraud on 11/10/2019

from __future__ import print_function
from pdb import set_trace as BP
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import MinMaxScaler


def col2im_sliding(B, m, n, mm, nn):
    return B.reshape(nn - n + 1, mm - m + 1).T


def gaussian2d(Dw, Dh, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], mu=0.0, sigma=1.0):
    """
    Returns the pdf of a two-dimensional multivariate gaussian distribution.
    """
    xstep = np.diff(xlim) / Dw
    ystep = np.diff(ylim) / Dh

    grd = np.empty((Dw, Dh, 2))
    grd[:, :, 0], grd[:, :, 1] = np.mgrid[xlim[0] : xlim[1] : xstep, ylim[0] : ylim[1] : ystep]

    return multivariate_normal.pdf(grd, mu * np.ones(2), sigma * np.eye(2)).flatten()


class ZCA(BaseEstimator, TransformerMixin):
    # Zero-phase whitening as proposed by [Olshausen et al]
    # Implemented by Georgios Exarchakis

    def __init__(self, n_components=None, bias=0.1, copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy

    def fit(self, X, var=0.95, y=None):
        # X = array2d(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        eigs, eigv = eigh(np.dot(X.T, X) / n_samples + self.bias * np.identity(n_features))
        inds = np.argsort(eigs)[::-1]

        eigs = eigs[inds]
        eigv = eigv[:, inds]
        neigs = eigs / np.sum(eigs)
        nc = np.arange(eigs.shape[0])[np.cumsum(neigs) >= var][0]
        eigs = eigs[:nc]
        eigv = eigv[:, :nc]

        W_zca = np.dot(eigv * np.sqrt(1.0 / eigs), eigv.T)
        W_zca_inv = np.dot(eigv * np.sqrt(eigs), eigv.T)
        self.W_zca = W_zca
        self.W_zca_inv = W_zca_inv

        # Order the explained variance from greatest to least
        self.explained_variance_ = eigs  # [inds]
        return self

    def transform(self, X):
        # X = array2d(X)
        if self.mean_ is not None:
            X -= self.mean_
        X_transformed = np.dot(X, self.W_zca)
        return X_transformed

    def inverse_transform(self, X):
        # X = array2d(X)
        X_inverse_transformed = np.dot(X, self.W_zca_inv)
        if self.mean_ is not None:
            X_inverse_transformed += self.mean_
        return X_inverse_transformed


class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True):
        self.copy = copy
        self.scaler = MinMaxScaler(copy=self.copy)

    def fit(self, X, feature_range=[0.0, 10.0], globally=True):
        self.feature_range = feature_range
        self.globally = globally
        self.scaler.set_params(feature_range=feature_range)

        # X = array2d(X)
        X = as_float_array(X, copy=self.copy)
        X_shape = X.shape

        if self.globally:
            X = X.reshape(-1, 1)

        self.scaler.fit(X)
        self.X_shape = X_shape

    def scale(self, X):
        # X = array2d(X)
        X = as_float_array(X, copy=self.copy)
        if self.globally:
            X = X.reshape(-1, 1)
        X = self.scaler.transform(X)
        if self.globally:
            X = X.reshape(self.X_shape)
        return X

    def rescale(self, X):
        # X = array2d(X)
        X = as_float_array(X, copy=self.copy)
        if self.globally:
            X = X.reshape(-1, 1)
        X = self.scaler.inverse_transform(X)
        if self.globally:
            X = X.reshape(self.X_shape)
        return X


class BlockProcessing(BaseEstimator, TransformerMixin):
    def __init__(
        self, data, mask, patchheight, patchwidth, mainprocessing=None, pp_params={}, patchstep=1
    ):
        self.I, self.M = data, mask
        self.Ih, self.Iw = data.shape
        self.Dh, self.Dw, self.step = patchheight, patchwidth, patchstep
        self.D = self.Dh * self.Dw
        self.Nh = int(np.ceil(float(self.Ih - self.Dh) / self.step) + 1)  # no patches in vert dir
        self.Nw = int(np.ceil(float(self.Iw - self.Dw) / self.step) + 1)  # no patches in hor dir
        # assert self.Dh-self.step > 0
        # assert self.Dw-self.step > 0
        self.N = self.Nh * self.Nw
        self.Nh_ = int(
            np.ceil(float(self.Ih - self.Dh) / 1) + 1
        )  # no patches in vert dir for patchstep=1
        self.Nw_ = int(
            np.ceil(float(self.Iw - self.Dw) / 1) + 1
        )  # no patches in hor dir for patchstep=1
        self.N_ = self.Nh_ * self.Nw_
        self.mainprocessing = mainprocessing
        self.zca = ZCA()
        self.scale = Scaling()
        self.pp_params = pp_params
        self.pp_params_req_keys = (
            "sf_type",
            "sf_xlim",
            "sf_ylim",
            "sf_mu",
            "sf_sigma",
            "pp_type",
            "clamp_perc",
            "zca_var",
            "feature_range",
            "bp_verbose",
            "cvalue",
        )

    def check_params(self):
        pp_params = self.pp_params
        pp_params_req_keys = self.pp_params_req_keys
        for k in pp_params_req_keys:
            if k not in pp_params:
                pp_params[k] = None

        if pp_params["sf_xlim"] is None:
            pp_params["sf_xlim"] = [-1.0, 1.0]
        if pp_params["sf_ylim"] is None:
            pp_params["sf_ylim"] = [-1.0, 1.0]

        if pp_params["sf_mu"] is None:
            pp_params["sf_mu"] = 0.0
        if pp_params["sf_sigma"] is None:
            pp_params["sf_sigma"] = 1.0

        if pp_params["bp_verbose"] is None:
            pp_params["bp_verbose"] = False

    def segmentation(self):

        I = self.I
        Ih, Iw = self.Ih, self.Iw
        D, Dh, Dw, step = self.D, self.Dh, self.Dw, self.step
        Nh, Nw, N = self.Nh, self.Nw, self.N
        Nh_, Nw_, N_ = self.Nh_, self.Nw_, self.N_

        # sliding window moves l->r and then top->bottom
        # temporarily use step=1
        # is (Ih-Dh+1, Iw-Dw+1, Dh, Dw)
        Y = view_as_windows(I, window_shape=[Dh, Dw], step=1)
        # is (D,N_)
        Y = Y.reshape(self.Dh, self.Dw, self.N_).reshape(self.N_, self.D).T
        X = view_as_windows(self.M, window_shape=[self.Dh, self.Dw], step=1)
        X = X.reshape(self.Dh, self.Dw, self.N_).reshape(self.N_, self.D).T

        # lower right window locations in original
        ninds_ = np.arange(N_).reshape(Nh_, Nw_)  # for patchstep = 1
        ninds = np.arange(N).reshape(Nh, Nw)  # for patchstep = step

        # relevant patches for patchstep=step
        hinds = (np.unique(np.append(np.arange(1, Nh_, step), [Nh_])) - 1).flatten()
        winds = (np.unique(np.append(np.arange(1, Nw_, step), [Nw_])) - 1).flatten()
        ninds = (ninds_[hinds, :][:, winds]).flatten()  # ind of rel patches

        assert len(ninds) == N

        Y = Y[:, ninds]  # is (D,N)
        X = X[:, ninds]

        # spatial order of pixel indices, is (Dh, Dw)
        # indexed l->r and then top->bottom
        dinds = np.arange(D).reshape(Dh, Dw)

        self.Y = Y
        self.X = X
        self.ninds_, self.ninds, self.dinds = ninds_, ninds, dinds

    def preprocessing(self):

        pp_type = self.pp_params["pp_type"]
        if pp_type is None:
            return

        Y = self.Y.T  # is (N,D)
        clamp_perc = self.pp_params["clamp_perc"]
        zca_var = self.pp_params["zca_var"]
        feature_range = self.pp_params["feature_range"]

        if "w" in pp_type:
            print("Performing ZCA")
            self.zca.fit(Y, var=zca_var)
            Y = self.zca.transform(Y)

        if "s" in pp_type:
            print("Performing scaling")
            self.scale.fit(Y, feature_range=feature_range, globally=True)
            Y = self.scale.scale(Y)

        self.Y = Y.T

    def postprocessing(self):

        pp_type = self.pp_params["pp_type"]
        if pp_type is None:
            return

        Y = self.Y.T

        if "s" in pp_type:
            print("Performing re-scaling")
            Y = self.scale.rescale(Y)

        if "w" in pp_type:
            print("Performing inverse ZCA")
            Y = self.zca.inverse_transform(Y)

        self.Y = Y.T

    def synthesis(self):

        I, M, Y = self.I, self.M, self.Y
        Ih, Iw = self.Ih, self.Iw
        Dw, Dh, step = self.Dw, self.Dh, self.step
        Nh_, Nw_, Nh, Nw = self.Nh_, self.Nw_, self.Nh, self.Nw
        ninds_, ninds, dinds = self.ninds_, self.ninds, self.dinds
        D, N = self.D, self.N
        sf_type = self.pp_params["sf_type"]
        verbose = self.pp_params["bp_verbose"]
        verbosep = 0.1
        cvalue = self.pp_params["cvalue"]

        if sf_type == "gauss_":
            sf = gaussian2d(
                Dw,
                Dh,
                xlim=self.pp_params["sf_xlim"],
                ylim=self.pp_params["sf_ylim"],
                mu=self.pp_params["sf_mu"],
                sigma=self.pp_params["sf_sigma"],
            )
        elif sf_type == "mean_":
            sf = np.ones((Dw * Dh,))

        # index tuples of missing values, is (total # missing vals)
        dmr, dmc = np.where(np.logical_not(M))

        # no missing values
        Xm = dmr.size

        for xm in range(Xm):

            # location of missing value in original image
            r = dmr[xm]
            c = dmc[xm]

            if verbose:
                print("Restoring pixel at (%i,%i) \n=============================" % (r, c))

            # location of relevant patches for patchstep = 1(rows and columns of ninds_)
            r_ = np.arange(max(r - Dh + 2, 1), min(r + 1, Nh_) + 1) - 1
            c_ = np.arange(max(c - Dw + 2, 1), min(c + 1, Nw_) + 1) - 1

            if verbose:
                print("Row ind of relevant patches are \n  %s\n" % (r_))
                print("Col ind of relevant patches are \n  %s\n" % (c_))

            ns_ = ninds_[r_, :][
                :, c_
            ].flatten()  # is no relevant patches for given pixel in original
            ds_ = np.sort(dinds[r - r_, :][:, c - c_].flatten())[::-1]

            if verbose:
                print("Patch ind for step=1 are \n  %s\n" % (ns_))
                print("Pixel ind for step=1 are \n  %s\n" % (ds_))

            # only use patches compatible with given patchstep
            if self.step > 1:
                # FIXME the np.isin calls are very slow, one could probably
                # only produce the right indexes in the first place and skip this step
                nsinds = np.isin(ns_, ninds)
                ns = ns_[nsinds]
                ds = ds_[nsinds]
            else:
                ns = ns_
                ds = ds_

            if verbose:
                print("Patch ind for step=%i considering all patches are \n  %s\n" % (step, ns))

            # indices considering remaining patches
            if self.step > 1:
                # FIXME the np.isin calls are very slow, one could probably
                # only produce the right indexes in the first place and skip this step
                ns = np.where(np.isin(ninds, ns))[0]

            if verbose:
                print(
                    "Patch ind for step=%i considering remaining patches are \n  %s\n" % (step, ns)
                )
                print("Pixel ind for step=%i are \n  %s\n" % (step, ds))

            # gather estimates and apply synthesis filter
            restored = Y[ds, ns]

            if verbose:
                print("Reconstructions from all patches \n  %s\n" % (restored))

            if sf_type == "median_":
                estimate = np.median(restored[restored != cvalue])
            elif sf_type == "max_":
                estimate = np.max(restored[restored != cvalue])
            elif sf_type in ("gauss_", "mean_"):
                sf_ = sf[ds] / sf[ds].sum()
                estimate = ((restored * sf_)[restored != cvalue]).sum()

            if verbose:
                print("Estimate is \n  %s\n" % (estimate))
                print()

            I[r, c] = estimate

            if verbose and (xm % np.floor(verbosep * Xm)) == 0.0:
                print("%d %%" % (float(xm) / Xm * 100))

    def process(self):

        self.check_params()

        self.segmentation()

        self.preprocessing()

        self.mainprocessing(self.Y, self.X)

        self.postprocessing()

        self.synthesis()

    def im2bl(self):

        self.check_params()

        self.segmentation()

        self.preprocessing()

    def bl2im(self):

        self.postprocessing()

        self.synthesis()


def mainprocessing_dummy(Y, X):
    # pass
    D, N = Y.shape
    a = {"y": Y.T, "x": X.T}
    y, x = a["y"], a["x"]
    # print y.T

    if np.logical_not(x).all():
        x = np.logical_not(x)

    for n in range(N):

        this_y = y[n, :]
        this_x = x[n, :]

        # this_y[np.logical_not(this_x)] = this_y[this_x].mean()
        # this_y[np.logical_not(this_x)] = this_y.mean()
        # this_y[np.logical_not(this_x)] = n
        # this_y[np.logical_not(this_x)] = np.random.random(np.logical_not(this_x).sum())
        # np.random.seed(100+n)
        # this_y[:] = np.random.random(len(this_y))
        this_y[:] += n * np.random.random()

    # Y[:] = y[:,:].copy().T


def plot_filter():

    # Plot Gaussian filter for weighted averaging
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Make data.
    X = np.arange(-5, 5, 1.0)
    Y = np.arange(-5, 5, 1.0)
    X, Y = np.meshgrid(X, Y)
    f = gaussian2d(Dw=10, Dh=10, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], mu=0.0, sigma=0.1)
    f = f / f.sum()
    print(f.sum())
    f = f.reshape(10, 10)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, f, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.01, 0.11)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def test():

    print("Testing blockprocessing\n=====================================")

    Dh, Dw, patchstep = 2, 2, 2

    # pp_params = {}
    pp_params = {"sf_type": "max_", "bp_verbose": True}

    # create image and artificial mask matrix
    Ih, Iw = 5, 5
    np.random.seed(0)
    I = np.random.random((Ih, Iw))
    I_copy = I.copy()
    # I = np.arange(Ih*Iw).reshape(Ih,Iw).astype(float)

    # Ih, Iw = 4, 3
    # I = np.array([[0.23362892, 0.4054638, 0.2431283, 0.34739911], \
    #  [0.8939867, 0.63614618, 0.69853074, 0.85552265], \
    #  [0.48580606, 0.93264408, 0.38811137, 0.84876699], \
    #  [0.74255703, 0.2544165, 0.08428175, 0.1612792 ]])
    # I = np.array(I[:Ih,:Iw])

    # I2 = I.copy()

    if pp_params["bp_verbose"]:
        print("Using parameters as follows:")
        print("Image size\t %i x %i" % (Ih, Iw))
        print("Patch size\t %i x %i" % (Dh, Dw))
        print("Patch step\t %i" % (patchstep))
        print("Filter type\t %s" % pp_params["sf_type"])
        print()
        print("Image")
        print(I)
        print()

    M = np.ones_like(I, dtype=bool)  # inpainting mask
    # M[0,0] = False
    # M[1,0] = False
    # M[0,1] = False
    M[2, 2] = False
    # M[3,3] = False
    # M[3,2] = False
    # M[4,4] = False

    # M = np.zeros_like(I,dtype=bool) # denoising mask
    # M = np.random.random(I.shape)<0.5

    if pp_params["bp_verbose"]:
        print("Mask")
        print(M)
        print()

    # # run block-processing
    # bp = BlockProcessing(I, M, Dh, Dw, mainprocessing_dummy, pp_params=pp_params)
    # bp.process()

    # same processing but different function calls
    bp2 = BlockProcessing(I, M, Dh, Dw, pp_params=pp_params, patchstep=patchstep)
    bp2.im2bl()

    if pp_params["bp_verbose"]:
        print("Blocks")
        print(bp2.Y.T)
        print()

    mainprocessing_dummy(bp2.Y, bp2.X)

    if pp_params["bp_verbose"]:
        print("Blocks processed")
        print(bp2.Y.T)
        print()

    bp2.bl2im()

    if pp_params["bp_verbose"]:
        print("Image processed")
        print(I)

    assert (I[M] == I_copy[M]).all()


if __name__ == "__main__":

    test()
