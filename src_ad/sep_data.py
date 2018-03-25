from __future__ import print_function


import os
import shutil


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data", "adni_all")
out_dir = os.path.join(parent_dir, "data", "adni_subj")
create_dir(out_dir)
label_dirs = ["AD", "NC"]

for label_dir in label_dirs:
    data_lable_dir = os.path.join(data_dir, label_dir)
    out_label_dir = os.path.join(out_dir, label_dir)
    create_dir(out_label_dir)
    for scan in os.listdir(data_lable_dir):
        src_path = os.path.join(data_lable_dir, scan)

        scan_temp = scan
        if scan_temp[0] == "_":
            scan_temp = scan_temp[1:]
        if label_dir == "AD":
            ID = scan_temp[:10]
        else:
            ID = scan_temp[5:15]

        dst_dir = os.path.join(out_label_dir, ID)
        create_dir(dst_dir)
        dst_path = os.path.join(dst_dir, scan)
        shutil.copyfile(src_path, dst_path)
