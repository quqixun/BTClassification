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


def mask(in_path, out_path, mask_path):
    in_volume = np.flipud(nib.load(in_path).get_data())
    mask_volume = binary_fill_holes(np.flipud(nib.load(mask_path).get_data()))
    out_volume = np.multiply(in_volume, mask_volume).astype(in_volume.dtype)
    nib.save(nib.Nifti1Image(out_volume, np.eye(4)), out_path)
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


# temp_mask = binary_fill_holes(np.flipud(nib.load(mask_path).get_data()))
# brain_mask = binary_fill_holes(np.flipud(nib.load(t1ce_mask_out_path).get_data()))
# brain_mask = binary_fill_holes(np.flipud(nib.load(flair_mask_out_path).get_data()))
# new_mask = np.zeros(temp_mask.shape)
# new_mask[np.where((brain_mask * 1 + temp_mask * 1) == 2)] = 1

# --------------- #
# Implementations #
# --------------- #

# Input Directory
input_dir = os.path.join(os.getcwd(), "FlairT1ceReg")
output_dir = os.path.join(os.getcwd(), "FlairT1ceSS")
templates_dir = os.path.join(os.getcwd(), "Template")

subjects = os.listdir(input_dir)
input_subj_dirs = [os.path.join(input_dir, subj) for subj in subjects]
output_subj_dirs = [os.path.join(output_dir, subj) for subj in subjects]

# --------------------------------------------- #
# Implementation of Method 1 - Apply Brain Mask #
# --------------------------------------------- #

print("\nImplementation of Method 1 - Apply Brain Mask\n")

# Path of template's mask
mask_path = os.path.join(templates_dir, "MNI152_T1_1mm_brain_mask.nii.gz")

# Test
# strip_skull_mask(input_subj_dirs[0], output_subj_dirs[0], mask_path)

# Multi-processing
paras = zip(input_subj_dirs, output_subj_dirs, [mask_path] * len(subjects))
pool = Pool(processes=cpu_count())
# pool.map(unwarp_strip_skull_mask, paras)

# -------------------------------------- #
# Implementation of Method 2 - Apply BET #
# -------------------------------------- #

print("\nImplementation of Method 2 - Apply BET\n")

# Test
# strip_skull_bet(input_subj_dirs[0], output_subj_dirs[0])

# Multi-processing
paras = zip(input_subj_dirs, output_subj_dirs)
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

# Test
# strip_skull_ants(input_subj_dirs[0], output_subj_dirs[0], templates, "flair")

# Multi-processing
paras = zip(input_subj_dirs, output_subj_dirs, [templates] * len(subjects))
pool = Pool(processes=6)
pool.map(unwarp_strip_skull_ants, paras)
