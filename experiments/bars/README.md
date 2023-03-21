# Bars Test

Standard and correlated bars tests.


## Requirements
To run the experiment, make sure to have completed the installation instructions [described here](../../README.md) and to have the `ecml2022` environment activated.

```bash
conda activate ecml2022
```


## Get started
For a standard bars test, run:

```bash
python main.py
```

To run the experiment using correlated bars, execute:

```bash
python main.py --correlated
```

To see all possible options, run:

```bash
python main.py -h
```


## GPU execution

To exploit GPU parallelization, run `env TVO_GPU=0 python main.py ...`. GPU execution requires a cudatoolkit installation.
