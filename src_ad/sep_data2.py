from __future__ import print_function

import os
import shutil
from tqdm import *


def txt2list(path, sep="\n"):
    return open(path, "r").read().split(sep)


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def move_files(mode, idx_list):
    ad_out_dir = os.path.join(output_dir, mode, "AD")
    nc_out_dir = os.path.join(output_dir, mode, "NC")
    create_dir(ad_out_dir)
    create_dir(nc_out_dir)

    for idx in tqdm(idx_list):
        idx_ad_path = os.path.join(input_dir, "AD", idx)
        idx_nc_path = os.path.join(input_dir, "NC", idx)

        if os.path.isfile(idx_ad_path):
            idx_in_path = idx_ad_path
            idx_out_path = os.path.join(ad_out_dir, idx)
        elif os.path.isfile(idx_nc_path):
            idx_in_path = idx_nc_path
            idx_out_path = os.path.join(nc_out_dir, idx)
        else:
            continue

        shutil.copyfile(idx_in_path, idx_out_path)

    return


parent_dir = os.path.dirname(os.getcwd())
input_dir = os.path.join(parent_dir, "data", "trimmed_112")
output_dir = os.path.join(parent_dir, "data", "new_sepsubj")

train_idx = txt2list("train.txt")
valid_idx = txt2list("val.txt")
test_idx = txt2list("test.txt")

move_files("train", train_idx)
move_files("valid", valid_idx)
move_files("test", test_idx)
