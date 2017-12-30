import os
import subprocess
import nibabel as nib
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt


# TEMPLATE_PATH = "MNI152_T1_1mm_brain.nii.gz"
TEMPLATE_PATH = "MNI152_T1_1mm.nii.gz"
MASK_PATH = "MNI152_T1_1mm_brain_mask.nii.gz"


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


input_data_dir = "/home/qixun/Desktop/TCGA/TCGA-LGG"
output_data_dir = "/home/qixun/Desktop/TCGA-SS/TCGA-LGG"

subjects = os.listdir(input_data_dir)
for subject in subjects[5:]:
    subject_in_dir = os.path.join(input_data_dir, subject)
    print("Working on: ", subject_in_dir)
    subject_out_dir = os.path.join(output_data_dir, subject)
    create_dir(subject_out_dir)

    try:
        for scan in os.listdir(subject_in_dir):
            if "T1Gd" in scan:
                t1gd_in_path = os.path.join(subject_in_dir, scan)
                t1gd_out_path = os.path.join(subject_out_dir, scan)
                orient2std(t1gd_in_path, t1gd_out_path)
                registration(t1gd_out_path, t1gd_out_path, TEMPLATE_PATH)

        for scan in os.listdir(subject_in_dir):
            if "T1Gd" not in scan:
                scan_in_path = os.path.join(subject_in_dir, scan)
                scan_out_path = os.path.join(subject_out_dir, scan)
                orient2std(scan_in_path, scan_out_path)
                registration(scan_out_path, scan_out_path, t1gd_out_path)
    except:
        print("    Falied on: ", subject_in_dir)
        continue

# input_nii_path = "T1Gd.nii.gz"
# output_path = "T1Gdo.nii.gz"

# input_nii = np.fliplr(np.rot90(nib.load(input_nii_path).get_data(), 1))
# # plt_middle(input_nii)

# mask = np.fliplr(np.rot90(nib.load(MASK_PATH).get_data(), 1))
# mask = binary_fill_holes(mask)

# ss = np.multiply(input_nii, mask).astype(input_nii.dtype)
# ss = np.rot90(ss, 3)
# ss = nib.Nifti1Image(ss, np.eye(4))
# nib.save(ss, output_path)
