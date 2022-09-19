# Code in support of "Direct Evolutionary Optimization of Variational Autoencoders With Binary Latents"

## Overview

The [experiments](./experiments) directory contains implementations of the experiments described in the [paper](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_1229.pdf). Execution requires an installation of the [Truncated Variational Optimization](https://github.com/tvlearn/tvo) (TVO) framework, which implements the Truncated Variational Autoencoder. Experiments furthermore leverage pre-/postprocessing and visualization utilities provided by [tvutil](https://github.com/tvlearn/tvutil).

After following the [Setup](#setup) instructions described below, you will be able to turn to running the experiments. Please consult the READMEs in the experiments' sub-directories for further instructions.

The code has only been tested on Linux systems.


## Setup
We recommend [Anaconda](https://www.anaconda.com/) to manage the installation, and to create a new environment for hosting the installed packages:

```bash
$ conda env create
$ conda activate ecml2022
```

The `tvo` package can be installed via:

```bash
$ git clone https://github.com/tvlearn/tvo.git
$ cd tvo
$ python setup.py build_ext
$ python setup.py install
$ cd ..
```

To install `tvutil`, run:

```bash
$ git clone https://github.com/tvlearn/tvutil.git
$ cd tvutil
$ python setup.py install
$ cd ..
```

## Reference

```bibtex
@inproceedings{DrefsEtAl2022,
  title={{Direct Evolutionary Optimization of Variational Autoencoders With Binary Latents}},
  author={Drefs, Jakob and Guiraud, Enrico and Panagiotou Filippos and J{\"o}rg L{\"u}cke},
  booktitle={Machine Learning and Knowledge Discovery in Databases. Research Track, accepted},
  year={2022}
}
```
