from __future__ import print_function

import os
import subprocess
import numpy as np
import nibabel as nib
from multiprocessing import Pool, cpu_count
from scipy.ndimage.morphology import binary_fill_holes


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def bet(in_path, out_path):
    command = ["bet", in_path, out_path, "-R", "-f", "0.5", "-g", "0.0", "-m"]
    subprocess.call(command)
    return


def mask(in_path, out_path, mask_path):
    in_volume = np.flipud(nib.load(in_path).get_data())
    mask_volume = binary_fill_holes(np.flipud(nib.load(mask_path).get_data()))
    out_volume = np.multiply(in_volume, mask_volume).astype(in_volume.dtype)
    nib.save(nib.Nifti1Image(out_volume, np.eye(4)), out_path)
    return


def unwarp_strip_skull(arg, **kwarg):
    return strip_skull(*arg, **kwarg)


def strip_skull(input_subj_dir, output_subj_dir):
    print("Working on :", input_subj_dir)
    create_dir(output_subj_dir)

    flair_in_path = os.path.join(input_subj_dir, "flair.nii.gz")
    t1ce_in_path = os.path.join(input_subj_dir, "t1ce.nii.gz")

    flair_out_path = os.path.join(output_subj_dir, "flair.nii.gz")
    flair_mask_out_path = os.path.join(output_subj_dir, "flair_mask.nii.gz")
    # flair_skull_out_path = os.path.join(output_subj_dir, "flair_skull.nii.gz")
    t1ce_out_path = os.path.join(output_subj_dir, "t1ce.nii.gz")

    try:
        bet(flair_in_path, flair_out_path)
        mask(t1ce_in_path, t1ce_out_path, flair_mask_out_path)
        # mask(flair_in_path, flair_out_path, flair_mask_out_path)
        os.remove(flair_mask_out_path)
        # os.remove(flair_skull_out_path)
    except RuntimeError:
        print("\tFailed on: ", input_subj_dir)

    return


input_dir = os.path.join(os.getcwd(), "FlairT1ceReg")
output_dir = os.path.join(os.getcwd(), "FlairT1ceSS")
subjects = os.listdir(input_dir)

input_subj_dirs = [os.path.join(input_dir, subj) for subj in subjects]
output_subj_dirs = [os.path.join(output_dir, subj) for subj in subjects]

# Test
# strip_skull(input_subj_dirs[0], output_subj_dirs[0])

# Multi-processing
paras = zip(input_subj_dirs, output_subj_dirs)
pool = Pool(processes=cpu_count())
pool.map(unwarp_strip_skull, paras)


# Function of Method 1 - Apply Brain Mask
# def ss1(in_dir, out_dir, mask_path):
#     mask = np.fliplr(np.rot90(nib.load(mask_path).get_data(), 1))
#     mask = binary_fill_holes(mask)

#     subjects = os.listdir(input_dir)

#     for subject in tqdm(subjects):
#         input_subj_dir = os.path.join(input_dir, subject)
#         output_subj_dir = os.path.join(output_dir1, subject)
#         create_dir(output_subj_dir)

#         for scan in os.listdir(input_subj_dir):
#             input_nii_path = os.path.join(input_subj_dir, scan)
#             output_nii_path = os.path.join(output_subj_dir, scan)

#             input_nii = np.fliplr(np.rot90(nib.load(input_nii_path).get_data(), 1))
#             ss = np.multiply(input_nii, mask).astype(input_nii.dtype)
#             ss = np.rot90(ss, 3)
#             ss = nib.Nifti1Image(ss, np.eye(4))
#             nib.save(ss, output_nii_path)


# # Implementation of Method 1 - Apply Brain Mask
# output_dir1 = os.path.join(os.getcwd(), "FlairT1ceSSMask")
# mask_path = "Template/MNI152_T1_1mm_brain_mask.nii.gz"
# ss1(input_dir, output_dir1, mask_path)
