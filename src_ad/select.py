from __future__ import print_function

import os
import shutil


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data", "Processed")
base_dir = os.path.join(parent_dir, "data", "adni_subj")
out_dir = os.path.join(parent_dir, "data", "Selected")
create_dir(out_dir)
label_dirs = ["AD", "NC"]


for label in label_dirs:
    data_label_dir = os.path.join(data_dir, label)
    base_label_dir = os.path.join(base_dir, label)
    out_label_dir = os.path.join(out_dir, label)
    create_dir(out_label_dir)

    for subject in os.listdir(base_label_dir):
        data_subj_dir = os.path.join(data_label_dir, subject)
        out_subj_dir = os.path.join(out_label_dir, subject)
        shutil.copytree(data_subj_dir, out_subj_dir)
