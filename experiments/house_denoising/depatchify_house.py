from blockprocessing import BlockProcessing
from utils import save, eval_psnr
import PIL.Image
import matplotlib.pyplot as plt
from contextlib import closing
import numpy as np
import h5py
from argparse import ArgumentParser

plt.switch_backend("agg")

def parse_args():
    p = ArgumentParser()
    p.add_argument("--train-out", required=True, help="output of TVAE training")
    p.add_argument("--image", required=True, help="original PNG house image")
    p.add_argument("--out-file-stem", required=True, help="stem for output files (must include parent directories if any)")
    return p.parse_args()


def depatchify_house(train_out, image, out_file_stem):
    """Take output of TVEM training, depatchify it, evaluate PSNR, produce HDF5 and PNG results."""
    with closing(h5py.File(train_out, "r")) as train_out:
       patches = train_out["train_reconstruction"][...]
    house = np.array(PIL.Image.open(image), dtype=np.float32)
    mask = np.zeros_like(house)  # 0 -> to reconstruct
    house_copy = house.copy()  # bp modifies the input
    bp = BlockProcessing(house_copy, mask=mask, patchheight=12, patchwidth=12,\
                         pp_params={"pp_type": None, "sf_type": "gauss_"})
    bp.im2bl()
    bp.Y[:] = patches.T
    bp.bl2im()
    psnr = eval_psnr(house, bp.I)

    save(f"{out_file_stem}.h5", {"data": bp.I, "psnr": psnr})
    plt.imshow(bp.I, cmap="gray")
    plt.axis("off")
    plt.title(f"Denoised house (PSNR={psnr:.2f})")
    plt.savefig(f"{out_file_stem}.png")


if __name__ == "__main__":
    args = parse_args()
    depatchify_house(args.train_out, args.image, args.out_file_stem)
