# Bars Test

Standard and correlated bars tests.


## Requirements
To run the experiment, make sure to have completed the installation instructions [described here](../../README.md) and to have the `tvo` environment activated.

```bash
conda activate tvo
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


## Options

Possible options are:

```bash
usage: Bars Test for TVAE [-h] [--output_directory OUTPUT_DIRECTORY] [--H_gen H_GEN] [--bar_amp BAR_AMP] [--no_data_points NO_DATA_POINTS] [--correlated]
                          [--Ksize KSIZE] [--selection {fitness,uniform}] [--crossover] [--no_parents NO_PARENTS] [--no_children NO_CHILDREN]
                          [--no_generations NO_GENERATIONS] [--no_epochs NO_EPOCHS] [--batch_size BATCH_SIZE] [--viz_every VIZ_EVERY]
                          [--gif_framerate GIF_FRAMERATE] [--pi_gen PI_GEN] [--sigma2_gen SIGMA2_GEN] [--H_train H_TRAIN] [--min_lr MIN_LR] [--max_lr MAX_LR]
                          [--epochs_per_half_cycle EPOCHS_PER_HALF_CYCLE]

optional arguments:
  -h, --help            show this help message and exit
  --output_directory OUTPUT_DIRECTORY
                        Directory to write training output and visualizations to (will be output/<TIMESTAMP> if not specified) (default: None)
  --H_gen H_GEN         Number of bars used to generate data (default: 8)
  --bar_amp BAR_AMP     Bar amplitude (default: 1.0)
  --no_data_points NO_DATA_POINTS
                        Number of datapoints (default: 500)
  --correlated          Whether to introduce correlations in generated bars combinations (default: False)
  --Ksize KSIZE         Size of the K sets (i.e., S=|K|) (default: 32)
  --selection {fitness,uniform}
                        Selection operator (default: fitness)
  --crossover           Whether to apply crossover. Must be False if no_children is specified (default: False)
  --no_parents NO_PARENTS
                        Number of parental states to select per generation (default: 5)
  --no_children NO_CHILDREN
                        Number of children to evolve per generation (default: 3)
  --no_generations NO_GENERATIONS
                        Number of generations to evolve (default: 2)
  --no_epochs NO_EPOCHS
                        Number of epochs to train (default: 200)
  --batch_size BATCH_SIZE
                        batch size (default: 32)
  --viz_every VIZ_EVERY
                        Create visualizations every Xth epoch. Set to no_epochs if not specified. (default: 1)
  --gif_framerate GIF_FRAMERATE
                        Frames per second for gif animation (e.g., 2/1 for 2 fps). If not specified, no gif will be produced. (default: None)
  --pi_gen PI_GEN       Sparsity used for data generation (defaults to 2/H if not specified) (default: None)
  --sigma2_gen SIGMA2_GEN
                        Noise level used for data generation (default: 0.01)
  --H_train H_TRAIN     Number of generative fields to learn (set to H_gen if not specified) (default: None)
  --min_lr MIN_LR       minimal learning rate (default: 0.0001)
  --max_lr MAX_LR       maximal learning rate (default: 0.1)
  --epochs_per_half_cycle EPOCHS_PER_HALF_CYCLE
                        epochs per half cycle of learning rate scheduler (default: 10)
```


## GPU execution

To exploit GPU parallelization, run `env TVO_GPU=0 python main.py ...`. GPU execution requires a cudatoolkit installation.
