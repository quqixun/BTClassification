from __future__ import print_function

import os
import shutil
import subprocess
import numpy as np
import nibabel as nib
from multiprocessing import Pool, cpu_count
from scipy.ndimage.morphology import binary_fill_holes


# ---------------- #
# Helper Functions #
# ---------------- #

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_nii(path):
    return np.flipud(nib.load(path).get_data())


def save_nii(data, path):
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return


def mask(in_path, out_path, mask_path):
    in_volume = load_nii(in_path)
    mask_volume = binary_fill_holes(load_nii(mask_path))
    out_volume = np.multiply(in_volume, mask_volume).astype(in_volume.dtype)
    save_nii(out_volume, out_path)
    return


# --------------------------------------- #
# Function of Method 1 - Apply Brain Mask #
# --------------------------------------- #

def unwarp_strip_skull_mask(arg, **kwarg):
    return strip_skull_mask(*arg, **kwarg)


def strip_skull_mask(input_subj_dir, output_subj_dir, mask_path):
    print("Working on :", input_subj_dir)
    create_dir(output_subj_dir)

    for scan in os.listdir(input_subj_dir):
        input_nii_path = os.path.join(input_subj_dir, scan)
        output_nii_path = os.path.join(output_subj_dir, scan)
        mask(input_nii_path, output_nii_path, mask_path)
        new_mask_path = os.path.join(output_subj_dir, "mask.nii.gz")
        shutil.copyfile(mask_path, new_mask_path)
    return


# --------------------------------- #
# Functions of Method 2 - Apply BET #
# --------------------------------- #

def bet(in_path, out_path, frac="0.5", grad="0.0"):
    command = ["bet", in_path, out_path, "-R", "-f", frac, "-g", grad, "-m"]
    subprocess.call(command)
    return


def unwarp_strip_skull_bet(arg, **kwarg):
    return strip_skull_bet(*arg, **kwarg)


def strip_skull_bet(input_subj_dir, output_subj_dir,
                    mode="flair", frac="0.5", grad="0.0"):
    print("Working on :", input_subj_dir)
    create_dir(output_subj_dir)

    flair_in_path = os.path.join(input_subj_dir, "flair.nii.gz")
    t1ce_in_path = os.path.join(input_subj_dir, "t1ce.nii.gz")

    flair_out_path = os.path.join(output_subj_dir, "flair.nii.gz")
    t1ce_out_path = os.path.join(output_subj_dir, "t1ce.nii.gz")

    flair_mask_out_path = os.path.join(output_subj_dir, "flair_mask.nii.gz")
    t1ce_mask_out_path = os.path.join(output_subj_dir, "t1ce_mask.nii.gz")
    bet_mask_out_path = os.path.join(output_subj_dir, "bet_mask.nii.gz")

    try:
        if mode == "t1ce":
            bet(t1ce_in_path, t1ce_out_path, frac, grad)
            mask(flair_in_path, flair_out_path, t1ce_mask_out_path)
            os.rename(t1ce_mask_out_path, bet_mask_out_path)
        else:
            bet(flair_in_path, flair_out_path, frac, grad)
            mask(t1ce_in_path, t1ce_out_path, flair_mask_out_path)
            os.rename(flair_mask_out_path, bet_mask_out_path)
    except RuntimeError:
        print("\tFailed on: ", input_subj_dir)

    return


# ---------------------------------- #
# Functions of Method 3 - Apply ANTs #
# ---------------------------------- #

def ants(in_path, out_prefix, templates):
    command = ["antsBrainExtraction.sh", "-d 3", "-a", in_path,
               "-e", templates[0], "-m", templates[1], "-f", templates[2],
               "-o", out_prefix]
    cnull = open(os.devnull, 'w')
    subprocess.call(command, stdout=cnull, stderr=subprocess.STDOUT)
    brain_name = out_prefix + "BrainExtractionBrain.nii.gz"
    os.rename(brain_name, out_prefix + ".nii.gz")
    os.remove(out_prefix + "BrainExtractionPrior0GenericAffine.mat")
    return


def unwarp_strip_skull_ants(arg, **kwarg):
    return strip_skull_ants(*arg, **kwarg)


def strip_skull_ants(input_subj_dir, output_subj_dir, templates, mode="flair"):
    print("Working on :", input_subj_dir)
    create_dir(output_subj_dir)

    flair_in_path = os.path.join(input_subj_dir, "flair.nii.gz")
    t1ce_in_path = os.path.join(input_subj_dir, "t1ce.nii.gz")

    flair_out_path = os.path.join(output_subj_dir, "flair.nii.gz")
    t1ce_out_path = os.path.join(output_subj_dir, "t1ce.nii.gz")

    flair_out_prefix = os.path.join(output_subj_dir, "flair")
    t1ce_out_prefix = os.path.join(output_subj_dir, "t1ce")

    flair_mask_out_path = flair_out_prefix + "BrainExtractionMask.nii.gz"
    t1ce_mask_out_path = t1ce_out_prefix + "BrainExtractionMask.nii.gz"
    bet_mask_out_path = os.path.join(output_subj_dir, "ants_mask.nii.gz")

    try:
        if mode == "t1ce":
            ants(t1ce_in_path, t1ce_out_prefix, templates)
            mask(flair_in_path, flair_out_path, t1ce_mask_out_path)
            os.rename(t1ce_mask_out_path, bet_mask_out_path)
        else:
            ants(flair_in_path, flair_out_prefix, templates)
            mask(t1ce_in_path, t1ce_out_path, flair_mask_out_path)
            os.rename(flair_mask_out_path, bet_mask_out_path)
    except RuntimeError:
        print("\tFailed on: ", input_subj_dir)

    return


# ------------------------ #
# Functions of Mask Fusion #
# ------------------------ #

def unwarp_mask_fusion(arg, **kwarg):
    return mask_fusion(*arg, **kwarg)


def mask_fusion(input_subj_dir, output_subj_dir,
                temp_mask_path, bet_mask_path, ants_mask_path,
                weights=[0.1, 0.4, 0.5], threshold=None):
    print("Working on :", input_subj_dir)
    create_dir(output_subj_dir)

    temp_mask = load_nii(temp_mask_path)
    bet_mask = load_nii(bet_mask_path)
    ants_mask = load_nii(ants_mask_path)

    mask = temp_mask * weights[0] + bet_mask * weights[1] + ants_mask * weights[2]

    if threshold is not None:
        new_mask = np.zeros(mask.shape)
        new_mask[np.where(mask >= threshold)] = 1
        mask = new_mask

    mask = binary_fill_holes(mask) * 1.0
    mask_path = os.path.join(output_subj_dir, "brain_mask.nii.gz")
    save_nii(mask, mask_path)

    for scan in os.listdir(input_subj_dir):
        in_path = os.path.join(input_subj_dir, scan)
        out_path = os.path.join(output_subj_dir, scan)

        in_volume = load_nii(in_path)
        out_volume = np.multiply(in_volume, mask).astype(in_volume.dtype)
        save_nii(out_volume, out_path)

    return


# --------------- #
# Implementations #
# --------------- #

# Input Directory
cwd = os.getcwd()
input_dir = os.path.join(cwd, "FlairT1ceReg")
templates_dir = os.path.join(cwd, "Template")

# Obtain all subjects' ID
subjects = os.listdir(input_dir)
subj_num = len(subjects)

# Generate the paths of input directory
input_subj_dirs = [os.path.join(input_dir, subj) for subj in subjects]

# --------------------------------------------- #
# Implementation of Method 1 - Apply Brain Mask #
# --------------------------------------------- #

print("\nImplementation of Method 1 - Apply Brain Mask\n")

# Path of template's mask
mask_path = os.path.join(templates_dir, "MNI152_T1_1mm_brain_mask.nii.gz")

# Generate the paths of output directory
temp_output_dir = os.path.join(cwd, "FlairT1ceTempMask")
temp_output_subj_dirs = [os.path.join(temp_output_dir, subj) for subj in subjects]

# Test
# strip_skull_mask(input_subj_dirs[0], temp_output_subj_dirs[0], mask_path)

# Multi-processing
paras = zip(input_subj_dirs, temp_output_subj_dirs, [mask_path] * subj_num)
pool = Pool(processes=cpu_count())
# pool.map(unwarp_strip_skull_mask, paras)

# -------------------------------------- #
# Implementation of Method 2 - Apply BET #
# -------------------------------------- #

print("\nImplementation of Method 2 - Apply BET\n")

# Generate the paths of output directory
bet_output_dir = os.path.join(cwd, "FlairT1ceBetMask")
bet_output_subj_dirs = [os.path.join(bet_output_dir, subj) for subj in subjects]

# Test
# strip_skull_bet(input_subj_dirs[0], bet_output_subj_dirs[0])

# Multi-processing
paras = zip(input_subj_dirs, bet_output_subj_dirs)
pool = Pool(processes=cpu_count())
# pool.map(unwarp_strip_skull_bet, paras)

# --------------------------------------- #
# Implementation of Method 3 - Apply ANTs #
# --------------------------------------- #

print("\nImplementation of Method 2 - Apply ANTs\n")

# Path of templates
templates = [os.path.join(templates_dir, "MNI152_T1_1mm.nii.gz"),
             os.path.join(templates_dir, "MNI152_T1_1mm_brain_mask.nii.gz"),
             os.path.join(templates_dir, "MNI152_T1_1mm_first_brain_mask.nii.gz")]

# Generate the paths of output directory
ants_output_dir = os.path.join(cwd, "FlairT1ceTempMask")
ants_output_subj_dirs = [os.path.join(ants_output_dir, subj) for subj in subjects]

# Test
# strip_skull_ants(input_subj_dirs[0], ants_output_subj_dirs[0], templates, "flair")

# Multi-processing
paras = zip(input_subj_dirs, ants_output_subj_dirs, [templates] * subj_num)
pool = Pool(processes=cpu_count())
# pool.map(unwarp_strip_skull_ants, paras)

# ----------------------------- #
# Implementation of Mask Fusion #
# ----------------------------- #

print("\nImplementation of Mask Fusion to Extract Brain\n")

# Generate the paths of input masks
temp_mask_paths = [os.path.join(cwd, "FlairT1ceTempMask", subj, "mask.nii.gz") for subj in subjects]
bet_mask_paths = [os.path.join(cwd, "FlairT1ceBetMask", subj, "bet_mask.nii.gz") for subj in subjects]
ants_mask_paths = [os.path.join(cwd, "FlairT1ceAntsMask", subj, "ants_mask.nii.gz") for subj in subjects]

# Generate the paths of output directory
output_dir = os.path.join(cwd, "FlairT1ceBrain")
output_subj_dirs = [os.path.join(output_dir, subj) for subj in subjects]

# Set the weights for temp mask, bet mask and ants mask
weights = [1, 1, 1]
threshold = 3

# Test
# mask_fusion(input_subj_dirs[0], output_subj_dirs[0],
#             temp_mask_paths[0], bet_mask_paths[0], ants_mask_paths[0],
#             weights, threshold)

# Multi-processing
paras = zip(input_subj_dirs, output_subj_dirs,
            temp_mask_paths, bet_mask_paths, ants_mask_paths,
            [weights] * subj_num, [threshold] * subj_num)
pool = Pool(processes=cpu_count())
pool.map(unwarp_mask_fusion, paras)
