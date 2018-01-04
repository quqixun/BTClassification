from __future__ import print_function

import os
import subprocess
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


TEMPLATE_PATH = "Template/MNI152_T1_1mm.nii.gz"
# TEMPLATE_PATH = "MNI152_T1_1mm_brain.nii.gz"


def plt_middle(volume, slice_no=None):
    if not slice_no:
        slice_no = volume.shape[-1] // 2
    plt.figure()
    plt.imshow(volume[..., slice_no], cmap="gray")
    plt.show()
    return


def registration(in_path, out_path, ref_path):
    command = ["flirt", "-in", in_path, "-ref", ref_path, "-out", out_path,
               "-bins", "256", "-cost", "corratio", "-searchrx", "0", "0",
               "-searchry", "0", "0", "-searchrz", "0", "0", "-dof", "12",
               "-interp", "spline"]
    subprocess.call(command)
    return


def orient2std(in_path, out_path):
    command = ["fslreorient2std", in_path, out_path]
    subprocess.call(command)
    return


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def unwarp_main(arg, **kwarg):
    return main(*arg, **kwarg)


def main(input_data_dir, output_data_dir, subject):
    subject_in_dir = os.path.join(input_data_dir, subject)
    print("Working on: ", subject_in_dir)
    subject_out_dir = os.path.join(output_data_dir, subject)
    create_dir(subject_out_dir)

    try:
        for scan in os.listdir(subject_in_dir):
            if "t1ce" in scan:
                t1ce_in_path = os.path.join(subject_in_dir, scan)
                t1ce_out_path = os.path.join(subject_out_dir, scan)
                orient2std(t1ce_in_path, t1ce_out_path)
                registration(t1ce_out_path, t1ce_out_path, TEMPLATE_PATH)

        for scan in os.listdir(subject_in_dir):
            if "flair" in scan:
                scan_in_path = os.path.join(subject_in_dir, scan)
                scan_out_path = os.path.join(subject_out_dir, scan)
                orient2std(scan_in_path, scan_out_path)
                registration(scan_out_path, scan_out_path, t1ce_out_path)
    except RuntimeError:
        print("\tFalied on: ", subject_in_dir)

    return


cwd = os.getcwd()
input_data_dir = os.path.join(cwd, "FlairT1ce")
output_data_dir = os.path.join(cwd, "FlairT1ceReg")
subjects = os.listdir(input_data_dir)

# Test
# main(input_data_dir, output_data_dir, subjects[0])

# Multi-processing
paras = zip([input_data_dir] * len(subjects),
            [output_data_dir] * len(subjects),
            subjects)
pool = Pool(processes=cpu_count())
pool.map(unwarp_main, paras)
