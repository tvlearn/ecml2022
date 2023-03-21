# Image Inpainting

Apply TVAE to Image Inpainting with Missing Completely at Random Pixels


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

with `<BENCHMARK>` being one of `{house-50,castle-50,castle-80}`. Run `python main.py -h` for help.



## GPU execution

To exploit GPU parallelization, run `env TVO_GPU=0 python main.py ...`. GPU execution requires a cudatoolkit installation.
