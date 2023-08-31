# Code in support of "Direct Evolutionary Optimization of Variational Autoencoders With Binary Latents"

## Overview

The [experiments](./experiments) directory contains implementations of the experiments described in the [paper](https://link.springer.com/chapter/10.1007/978-3-031-26409-2_22). Execution requires an installation of the [Truncated Variational Optimization](https://github.com/tvlearn/tvo) (TVO) framework, which implements the Truncated Variational Autoencoder. Experiments furthermore leverage pre-/postprocessing and visualization utilities provided by [tvutil](https://github.com/tvlearn/tvutil).

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
@InProceedings{DrefsGuiraudEtAl2022,
  author="Drefs, Jakob
  and Guiraud, Enrico
  and Panagiotou, Filippos
  and L{\"u}cke, J{\"o}rg",
  editor="Amini, Massih-Reza
  and Canu, St{\'e}phane
  and Fischer, Asja
  and Guns, Tias
  and Kralj Novak, Petra
  and Tsoumakas, Grigorios",
  title="Direct Evolutionary Optimization of Variational Autoencoders with Binary Latents",
  booktitle="Machine Learning and Knowledge Discovery in Databases",
  year="2023",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="357--372",
}
```
