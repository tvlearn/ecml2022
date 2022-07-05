import numpy as np
from PIL import Image
from blockprocessing import BlockProcessing
from argparse import ArgumentParser
from utils import save

'''
--img ../data/house.png --out-fname ../data/house_p8.h5 --patch-size 8 --sigma 0.25
'''
def parse_args():
    p = ArgumentParser()
    p.add_argument("--img", required=True, help="house PNG image path")
    p.add_argument("--out-fname", required=True, help="HDF5 output path")
    p.add_argument("--patch-size", required=True, type=int, help="height/width of square image patches")
    p.add_argument("--sigma", required=True, type=float, help="scale of image noise")
    return p.parse_args()
    

def patchify_house(img, out_fname, patch_size, sigma):
    """Take png image, produce HDF5 file with patchified version."""

    img = np.array(Image.open(img))
    img = img + np.random.normal(scale=sigma, size=img.shape)
    bp = BlockProcessing(img, mask=np.ones_like(img), patchheight=patch_size, patchwidth=patch_size)
    bp.im2bl()
    data = bp.Y.T.astype(np.float32)
    save(out_fname, {"data": data})


if __name__ == "__main__":
    args = parse_args()
    patchify_house(args.img, args.out_fname, args.patch_size, args.sigma)
