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


## Options

Possible options of the experiment are:

```bash
usage: train.py [-h] [--output_directory OUTPUT_DIRECTORY] [--hidden_shape HIDDEN_SHAPE [HIDDEN_SHAPE ...]] [--min_lr MIN_LR] [--max_lr MAX_LR]
                [--epochs_per_half_cycle EPOCHS_PER_HALF_CYCLE] [--Ksize KSIZE] [--selection {fitness,uniform}] [--crossover] [--no_parents NO_PARENTS]
                [--no_children NO_CHILDREN] [--no_generations NO_GENERATIONS] [--no_epochs NO_EPOCHS] [--batch_size BATCH_SIZE]

Training and testing on CIFAR-10

optional arguments:
  -h, --help            show this help message and exit
  --output_directory OUTPUT_DIRECTORY
                        Directory to write H5 training output and visualizations to(will be output/<TIMESTAMP> if not specified) (default: None)
  --hidden_shape HIDDEN_SHAPE [HIDDEN_SHAPE ...]
                        Decoder shape excluding no. obsersables, H0-H1-... (default: [10, 10])
  --min_lr MIN_LR       minimal learning rate (default: 0.0001)
  --max_lr MAX_LR       maximal learning rate (default: 0.1)
  --epochs_per_half_cycle EPOCHS_PER_HALF_CYCLE
                        epochs per half cycle of learning rate scheduler (default: 10)
  --Ksize KSIZE         Size of the K sets (i.e., S=|K|) (default: 30)
  --selection {fitness,uniform}
                        Selection operator (default: fitness)
  --crossover           Whether to apply crossover. Must be False if no_children is specified (default: False)
  --no_parents NO_PARENTS
                        Number of parental states to select per generation (default: 20)
  --no_children NO_CHILDREN
                        Number of children to evolve per generation (default: 2)
  --no_generations NO_GENERATIONS
                        Number of generations to evolve (default: 1)
  --no_epochs NO_EPOCHS
                        Number of epochs to train (default: 200)
  --batch_size BATCH_SIZE
                        batch size (default: 32)
```


## GPU execution

To exploit GPU parallelization, run `env TVO_GPU=0 python train.py ...`. GPU execution requires a cudatoolkit installation.
