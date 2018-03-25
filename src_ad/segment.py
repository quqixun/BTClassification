from __future__ import print_function

import os
import glob
import shutil
import subprocess
import numpy as np
from scipy.ndimage.interpolation import zoom
from multiprocessing import Pool, cpu_count


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data")
data_src_dir = os.path.join(data_dir, "adni_subj")
data_dst_dir = os.path.join(data_dir, "adni_subj_seg")
