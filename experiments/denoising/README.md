# Gaussian Denoising

Gaussian Denoising with TVAE


## Requirements
To run the experiment, make sure to have completed the installation instructions [described here](../../README.md) and to have the `ecml2022` environment activated.

```bash
conda activate ecml2022
```


## Get started
The experiment can be started via:

```bash
python main.py <BENCHMARK>
```

with `<BENCHMARK>` being one of `{house-15,house-25,house-50,barbara-25}`. Run `python main.py -h` for help.



## GPU execution

To exploit GPU parallelization, run `env TVO_GPU=0 python main.py ...`. GPU execution requires a cudatoolkit installation.
