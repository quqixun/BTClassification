from __future__ import print_function

import os
import numpy as np
import nibabel as nib
from scipy.signal import medfilt
from multiprocessing import Pool, cpu_count
from scipy.ndimage.morphology import (binary_erosion, generate_binary_structure)
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection


# ---------------- #
# Helper Functions #
# ---------------- #

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_nii(path):
    return nib.load(path).get_data()


def save_nii(data, path):
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return


# ------------------------------------------- #
# Processing Step 1: N4 Bias Field Correction #
# ------------------------------------------- #

def unwarp_bias_field_correction(arg, **kwarg):
    return bias_field_correction(*arg, **kwarg)


def bias_field_correction(in_subj_dir, out_subj_dir):
    print("N4ITK on: ", in_subj_dir)
    create_dir(out_subj_dir)

    for scan_name in os.listdir(in_subj_dir):

        if "mask" in scan_name:
            continue

        in_path = os.path.join(in_subj_dir, scan_name)
        out_path = os.path.join(out_subj_dir, scan_name)
        try:
            n4 = N4BiasFieldCorrection()
            n4.inputs.input_image = in_path
            n4.inputs.output_image = out_path

            n4.inputs.dimension = 3
            n4.inputs.n_iterations = [100, 100, 60, 40]
            n4.inputs.shrink_factor = 3
            n4.inputs.convergence_threshold = 1e-4
            n4.inputs.bspline_fitting_distance = 300
            n4.run()
        except RuntimeError:
            print("\tFailed on: ", in_path)

    return


# -------------------------------------- #
# Processing Step 2: Denoise and Enahnce #
#   [1] Apply Median Filter              #
#   [2] Rescale Intensities              #
#   [3] Apply Histogram Equalization     #
# -------------------------------------- #

def denoise(volume, kernel_size=3):
    return medfilt(volume, kernel_size)


def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])

    obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
    obj_volume[np.where(obj_volume < 1)] = 1
    obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume[np.where(volume > 0)] = obj_volume
    return volume


def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num, normed=True)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


def unwarp_preprocess(arg, **kwarg):
    return preprocess(*arg, **kwarg)


def preprocess(in_subj_dir, out_subj_dir, kernel_size=3,
               percentils=[0.5, 99.5], bins_num=256):
    print("Preprocess on: ", in_subj_dir)
    create_dir(out_subj_dir)

    for scan_name in os.listdir(in_subj_dir):
        in_path = os.path.join(in_subj_dir, scan_name)
        out_path = os.path.join(out_subj_dir, scan_name)

        try:
            volume = load_nii(in_path)
            volume = denoise(volume, 3)
            volume = rescale_intensity(volume, percentils, bins_num)
            volume = equalize_hist(volume, bins_num)
            save_nii(volume, out_path)
        except RuntimeError:
            print("\tFailed on: ", in_path)


# --------------- #
# Implementations #
# --------------- #

# Current working directory
cwd = os.getcwd()


# --------------------------------------------------- #
# Step 1: Implementations of N4 Bias Field Correction #
# --------------------------------------------------- #

print("\nStep 1: N4 Bias Field Correction\n")

n4bfc_input_dir = os.path.join(cwd, "FlairT1ceBrain")
n4bfc_output_dir = os.path.join(cwd, "FlairT1ceN4BFC")

subjects = os.listdir(n4bfc_input_dir)
subj_num = len(subjects)

input_subj_dirs = [os.path.join(n4bfc_input_dir, subj) for subj in subjects]
output_subj_dirs = [os.path.join(n4bfc_output_dir, subj) for subj in subjects]

# Test
# bias_field_correction(input_subj_dirs[0], output_subj_dirs[0])

# Multi-processing
paras = zip(input_subj_dirs, output_subj_dirs)
pool = Pool(processes=cpu_count())
# pool.map(unwarp_bias_field_correction, paras)


# ---------------------------------------------- #
# Step 2: Implementations of Denoise and Enahnce #
# ---------------------------------------------- #

print("\nStep 1: Denoise and Enahnce\n")

prep_input_dir = n4bfc_output_dir
prep_output_dir = os.path.join(cwd, "FlairT1cePrep")

subjects = os.listdir(prep_input_dir)
subj_num = len(subjects)

input_subj_dirs = [os.path.join(prep_input_dir, subj) for subj in subjects]
output_subj_dirs = [os.path.join(prep_output_dir, subj) for subj in subjects]

kernel_size = 5
percentils = [0.5, 99.5]
bins_num = 1024

# Test
# preprocess(input_subj_dirs[0], output_subj_dirs[0],
#            kernel_size, percentils, bins_num)

# Multi-processing
paras = zip(input_subj_dirs, output_subj_dirs,
            [kernel_size] * subj_num,
            [percentils] * subj_num,
            [bins_num] * subj_num)
pool = Pool(processes=cpu_count())
pool.map(unwarp_preprocess, paras)
