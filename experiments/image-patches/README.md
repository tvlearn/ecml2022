# Natural Image Patches

Training TVAE on van Hateren image patches. Initialize with BSC encoding.


## Requirements
To run the experiment, make sure to have completed the installation instructions [described here](../../README.md) and to have the `tvo` environment activated.

```bash
conda activate tvo
```


## Data
First, create a training data set by running

```bash
python get-data.py
```

The results reported in the paper were obtained using images patches randomly sampled from the full data set of van Hateren et al. [1]. For storage reasons, we here provide one exemplary image from the full data set. 
Possible options to sample patches are:

```bash
usage: Build data set of whitened image patches extracted from image file [-h] [--image_file IMAGE_FILE] [-N N] [--timestamp]
                                                                          [--patch_size PATCH_SIZE PATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --image_file IMAGE_FILE
                        Full path to image to extract patches from (png, jpg, ... file) (default: ./data/image.tiff)
  -N N, --no_patches N  Number of patches to extract (default: 1000)
  --timestamp           Whether to append timestamp to file name of written out training data set (default: False)
  --patch_size PATCH_SIZE PATCH_SIZE
                        Patch (height, width) (default: [8, 8])
```

## Train

To first learn initial parameters using BSC, run:

```bash
python train.py bsc-from-scratch
```

By default, the training output will be written to `output/<TIMESTAMP>.h5` with `<TIMESTAMP>` corresponding to the date and time at execution. To initialize and train TVAE with these parameters, run:

```bash
python train.py tvae-from-init --init_file output/<TIMESTAMP>.h5
```

Likewise, another BSC model can be initialized and trained (for comparison) via:

```bash
python train.py bsc-from-init --init_file output/<TIMESTAMP>.h5
```

Options can be displayed via `python train.py <PROGRAM> -h` with `<PROGRAM>` being one of `bsc-from-scratch`, `tvae-from-init`, `bsc-from-init`.


## GPU execution

To exploit GPU parallelization, run `env TVO_GPU=0 python train.py ...`. GPU execution requires a cudatoolkit installation.


## References

[1] van Hateren, J.H., van der Schaaf, A.: Independent component filters of natural images compared with simple cells in primary visual cortex. Proceedings of the Royal Society of London. Series B: Biological Sciences 265, 359–66 (1998)
