# Code in support of "Direct Evolutionary Optimization of Variational Autoencoders With Binary Latents"

## Overview

The [experiments](./experiments) directory contains implementations of the experiments described in the paper. Execution requires an installation of the [Truncated Variational Optimization](https://github.com/tvlearn/tvo) (TVO) framework, which implements the Truncated Variational Autoencoder. Experiments furthermore leverage pre-/postprocessing and visualization utilities provided by [tvutil](https://github.com/tvlearn/tvutil).

After following the [Setup](#setup) instructions described below, you will be able to turn to running the experiments. Please consult the READMEs in the experiments' sub-directories for further instructions.

The code has only been tested on Linux systems.


## Setup
We recommend [Anaconda](https://www.anaconda.com/) to manage the installation, and to create a new environment for hosting installed packages:

```bash
$ conda create -n tvo python==3.8
$ conda activate tvo
```

Next, make sure to install the packages specified in `requirements.txt `:

```bash
$ pip install -r requirements.txt
```

Finally, `tvo` can be set up:

```bash
$ git clone https://github.com/tvlearn/tvo.git
$ cd tvo
$ python setup.py build_ext
$ python setup.py install
$ cd ..
```

Likewise, install `tvutil` as follows:

```bash
$ git clone https://github.com/tvlearn/tvutil.git
$ cd tvutil
$ python setup.py install
$ cd ..
```
