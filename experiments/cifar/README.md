# CIFAR-10

Train TVAE on CIFAR-10.


## Requirements
To run the experiment, make sure to have completed the installation instructions [described here](../../README.md) and to have the `tvo` environment activated.

```bash
conda activate tvo
```


## Get started
First, you need to download the CIFAR-10 training and test data set, by running: 

```bash
python get-data.py
```

The experiment can be started via:

```bash
python train.py
```

To list all possible options, run:

```bash
python train.py -h
```


## GPU execution

To exploit GPU parallelization, run `env TVO_GPU=0 python train.py ...`. GPU execution requires a cudatoolkit installation.
