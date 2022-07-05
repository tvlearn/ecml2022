# -*- coding: utf-8 -*-
import numpy as np
import argparse
import torch as to
from typing import Tuple
import h5py

from tvo.exp import EVOConfig, ExpConfig, Training
from tvo.models import GaussianTVAE as TVAE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="HD5 file as expected in input by tvo.Training")
    parser.add_argument("--Ksize", type=int, help="size of each K^n set")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument(
        "--net-shape",
        required=True,
        type=parse_net_shape,
        help="column-separated list of layer sizes",
    )
    parser.add_argument("--min-lr", type=float, help="MLP min learning rate", required=True)
    parser.add_argument("--max-lr", type=float, help="MLP max learning rate", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", help="output file for train log", required=True)
    parser.add_argument(
        "--seed", type=int, help="seed value for random number generators. default is a random seed"
    )
    return parser.parse_args()


def parse_net_shape(net_shape: str) -> Tuple[int, ...]:
    """
    Parse string with TVAE shape into a tuple.

    :param net_shape: column-separated list of integers, e.g. `"10:10:2"`
    :returns: a tuple with the shape as integers, e.g. `(10,10,2)`
    """
    return tuple(map(int, net_shape.split(":")))


def train():
    args = parse_args()

    # parameter initialization
    print("parameter initialization...", end="")
    if args.seed:
        to.manual_seed(args.seed)
        np.random.seed(args.seed)
    S = args.Ksize
    net_shape = args.net_shape
    H = net_shape[-1]
    data_fname = args.dataset
    data_file = h5py.File(data_fname, "r")
    N, D = data_file["data"].shape
    assert net_shape[0] == D, 'net shape 0 is {}, D is {}'.format(net_shape[0], D)

    print(f"\ninput file: {args.dataset}")
    print(f"net layers: {net_shape}")
    for var in "S", "H", "N", "D":
        print(f"{var} = {eval(var)}")

    epochs_per_half_cycle = 10
    cycliclr_half_step_size = np.ceil(N / args.batch_size) * epochs_per_half_cycle
    assert args.epochs % (cycliclr_half_step_size * 2 // np.ceil(N / args.batch_size)) == 0,\
           "training end doesn't coincide with end of learning rate cycle(lrc={},epochs={})".format((cycliclr_half_step_size * 2 // np.ceil(N / args.batch_size)), args.epochs)
    m = TVAE(net_shape, min_lr=args.min_lr, max_lr=args.max_lr, cycliclr_step_size_up=cycliclr_half_step_size, precision=to.float32)
    conf = ExpConfig(batch_size=args.batch_size, output=args.output, reco_epochs=list(range(5, args.epochs, 5)))
    estep_conf = EVOConfig(n_states=args.Ksize, n_parents=10, n_children=9, n_generations=4, crossover=False, )
    t = Training(conf, estep_conf, m, data_fname)
    print("\nlearning...")
    for e_log in t.run(args.epochs):
        e_log.print()


if __name__ == "__main__":
    train()

