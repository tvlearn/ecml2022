import numpy as np
import h5py
from contextlib import closing


def save(fname, dict_to_save):
   with closing(h5py.File(fname, "w")) as f:
      for key, data in dict_to_save.items():
         f.create_dataset(key, data=data)


def eval_psnr(fig1, fig2):
   def my_rms(x):
      return np.sqrt(np.mean((x.flatten())**2))

   pamp = 255 # was: np.amax(fig1)
   rms = my_rms(fig1 - fig2)
   if rms == 0.:
      psnr_ = np.inf
   else:
      psnr_ = 20.*np.log10(pamp / rms)
   return psnr_



