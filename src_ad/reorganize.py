from __future__ import print_function

import os
import glob
import shutil
from tqdm import *


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


parent_dir = os.path.dirname(os.getcwd())
data_src_dir = os.path.join(parent_dir, "data", "Original")
data_dst_dir = os.path.join(parent_dir, "data", "ADNI")
create_dir(data_dst_dir)
labels = ["AD", "NC"]


for label in tqdm(labels):
    label_src_dir = os.path.join(data_src_dir, label)
    label_dst_dir = os.path.join(data_dst_dir, label)
    create_dir(label_dst_dir)

    subjects = os.listdir(label_src_dir)
    for subject in tqdm(subjects):
        subj_src_dir = os.path.join(label_src_dir, subject)
        subj_dst_dir = os.path.join(label_dst_dir, subject)
        create_dir(subj_dst_dir)

        scan_src_paths = glob.glob(subj_src_dir + "/*/*/*/*.nii")
        scan_src_paths.sort()

        for i in range(len(scan_src_paths)):
            scan_src_path = scan_src_paths[i]
            scan_dst_path = os.path.join(subj_dst_dir, str(i) + ".nii")
            shutil.copyfile(scan_src_path, scan_dst_path)
