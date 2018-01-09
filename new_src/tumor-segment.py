from __future__ import print_function

import os
import numpy as np
from tqdm import *
import nibabel as nib
from pyobb.obb import OBB
from math import factorial
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage.measurements import label
from multiprocessing import Pool, cpu_count
from scipy.ndimage.morphology import (binary_closing, binary_fill_holes,
                                      binary_opening, binary_dilation,
                                      binary_erosion, generate_binary_structure)


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


# --------------------------------------------- #
# Tumor Segemtation Step 1: Compute Difference #
# --------------------------------------------- #

def rescale_intensity(volume, bins_num):
    volume[np.where(volume < 0)] = 0
    none_bg_volume = volume[np.where(volume > 0)]

    min_value = np.min(none_bg_volume)
    max_value = np.max(none_bg_volume)

    none_bg_volume = (none_bg_volume - min_value) / (max_value - min_value) * (bins_num - 1)
    none_bg_volume = np.round(none_bg_volume).astype(np.uint8)
    volume[np.where(volume > 0)] = none_bg_volume

    return volume


def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num, normed=True)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(np.uint8)
    volume[np.where(volume > 0)] = obj_volume
    return volume


def unwarp_difference(arg, **kwarg):
    return difference(*arg, **kwarg)


def difference(in_subj_dir, out_subj_dir, cb_mask_path, bins_num=256):
    print("Compute Difference of: ", in_subj_dir)
    create_dir(out_subj_dir)

    t1ce = load_nii(os.path.join(in_subj_dir, "t1ce.nii.gz"))
    flair = load_nii(os.path.join(in_subj_dir, "flair.nii.gz"))

    # cb_mask = load_nii(cb_mask_path)[..., 1]
    # cb_mask = 1 - cb_mask / np.max(cb_mask)

    cb_mask = load_nii(cb_mask_path)
    cb_mask[np.where(cb_mask != 2)] = 1
    cb_mask[np.where(cb_mask == 2)] = 0

    diff = flair - t1ce
    diff = np.multiply(diff, cb_mask)
    diff = rescale_intensity(diff, bins_num)
    diff = equalize_hist(diff, bins_num)

    save_nii(diff, os.path.join(out_subj_dir, "diff.nii.gz"))
    return


# ------------------------------------------- #
# Tumor Segemtation Step 2: Compute Histogram #
# ------------------------------------------- #

def compute_hist(volume, bins_num):
    bins = np.arange(0, bins_num)
    hist = np.histogram(volume, bins=bins, density=True)

    x = hist[1][2:]
    y = hist[0][1:]

    return x, y


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])

    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y, mode='valid')


def plot_hist(volume_paths, bins_num):
    plt.figure()
    plt.title("Histogram of All Difference Volumes", fontsize=12)
    for path in tqdm(volume_paths):
        volume = load_nii(path)
        x, y = compute_hist(volume, bins_num)
        non_zero_idx = np.where(y > 0)
        x = x[non_zero_idx]
        y = y[non_zero_idx]
        y = savitzky_golay(y, 9, 0)
        plt.plot(x, y, "k", lw=0.3, alpha=0.5)
    plt.xlabel("Intensity", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid("on", linestyle="--", linewidth=0.5)
    plt.show()

    return


#
#
#

def thresholding(volume, threshold):
    volume[np.where(volume < threshold)] = 0
    mask = (volume > 0) * 1
    return volume, mask


def extract_features(volume):
    x_idx, y_idx, z_idx = np.where(volume > 0)
    features = []
    for x, y, z in zip(x_idx, y_idx, z_idx):
        features.append([volume[x, y, z] * np.sqrt(3), x, y, z])
    return np.array(features).astype(int)


def kmeans_cluster(volume, label_out_path, n_clusters):
    features = extract_features(volume)
    # feat = features[:, 0].reshape((-1, 1))
    # print(features.shape)
    # print(min(feat), max(feat))
    kmeans_model = KMeans(n_clusters=n_clusters, init="k-means++",
                          precompute_distances=True, verbose=0,
                          random_state=7, n_jobs=1,
                          max_iter=1000, tol=1e-6).fit(features)

    label_volume = np.zeros(volume.shape)
    for l, f in zip(kmeans_model.labels_, features):
        label_volume[f[1], f[2], f[3]] = l + 1

    return label_volume


def segment(volume, labels, n_clusters):
    # mask = (volume > 0) * 1
    # mask = binary_fill_holes(mask)
    structure = generate_binary_structure(3, 1)
    # structure1 = sphere([3, 3, 3], 1, [1, 1, 1])
    # structure2 = sphere([5, 5, 5], 3, [2, 2, 2])
    # mask = binary_opening(mask, structure).astype(np.uint8)
    # volume = np.multiply(volume, mask)

    mean_intensities = []
    for i in range(1, n_clusters + 1):
        mean_intensities.append(np.mean(volume[np.where(labels == i)]))

    target_label = mean_intensities.index(max(mean_intensities)) + 1
    seg = np.zeros(volume.shape)
    seg[np.where(labels == target_label)] = 1

    seg = binary_fill_holes(seg)
    seg = binary_opening(seg, structure)
    # seg = binary_closing(seg, structure2)
    seg = binary_dilation(seg, structure)
    seg = binary_fill_holes(seg)
    seg = binary_opening(seg, structure)
    # seg = binary_erosion(seg, structure).astype(np.uint8)

    return seg.astype(np.uint8)


def min_area_mask(mask):
    none_zero_idx = np.where(mask == 1)
    poses = []
    for i in range(len(none_zero_idx[0])):
        poses.append([none_zero_idx[0][i],
                      none_zero_idx[1][i],
                      none_zero_idx[2][i]])
    # poses = np.array(poses)
    obb = OBB.build_from_points(poses)
    min_idx = np.round(obb.min).astype(int)
    max_idx = np.round(obb.max).astype(int)

    cube_mask = np.zeros(mask.shape)
    cube_mask[min_idx[0]:max_idx[0],
              min_idx[1]:max_idx[1],
              min_idx[2]:max_idx[2]] = 1
    return cube_mask


def postprocess(volume):
    structure = generate_binary_structure(3, 3)
    labeled_array, num_features = label(volume, structure)
    voxel_nums = []
    for i in range(1, num_features + 1):
        voxel_nums.append(len(np.where(labeled_array == i)[0]))
    target_label = voxel_nums.index(max(voxel_nums)) + 1
    labeled_array[np.where(labeled_array != target_label)] = 0
    labeled_array[np.where(labeled_array == target_label)] = 1

    cube_mask1 = min_area_mask(labeled_array)
    cube_mask2 = np.zeros(labeled_array.shape)
    obj_index = np.where(labeled_array == 1)
    cube_mask2[min(obj_index[0]):max(obj_index[0]),
               min(obj_index[1]):max(obj_index[1]),
               min(obj_index[2]):max(obj_index[2])] = 1

    return labeled_array, cube_mask1, cube_mask2


def unwarp_tumor_segment(arg, **kwarg):
    return tumor_segment(*arg, **kwarg)


def tumor_segment(in_subj_dir, out_subj_dir):
    print("Segment on: ", in_subj_dir)
    create_dir(out_subj_dir)

    in_path = os.path.join(in_subj_dir, "diff.nii.gz")
    # thresh_out_path = os.path.join(out_subj_dir, "thresh.nii.gz")
    # label_out_path = os.path.join(out_subj_dir, "label.nii.gz")
    seg_out_path = os.path.join(out_subj_dir, "seg_mask.nii.gz")
    cube_out_path = os.path.join(out_subj_dir, "cube_mask.nii.gz")

    volume = load_nii(in_path)
    volume, mask = thresholding(volume, 220)
    structure = generate_binary_structure(3, 3)
    mask = binary_opening(mask, structure).astype(np.uint8)
    volume = np.multiply(volume, mask)

    seg_mask, cube_mask1, cube_mask2 = postprocess(volume)
    save_nii(seg_mask, seg_out_path)
    # save_nii(cube_mask1, cube1_out_path)
    save_nii(cube_mask2, cube_out_path)

    # n_clusters = 3
    # labels = kmeans_cluster(volume, label_out_path, n_clusters)
    # save_nii(labels, label_out_path)

    # seg = segment(volume, labels, n_clusters)
    # save_nii(seg, seg_out_path)

    return


def sphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (np.abs(x_i / semisize) ** 2)
    # the inner part of the sphere will have distance below 1
    return arr <= 1.0


#
#
#


def unwarp_mark_tumor(arg, **kwarg):
    return mark_tumor(*arg, **kwarg)


def mark_tumor(in_subj_dir, seg_subj_dir):
    print("Mark tumor in: ", in_subj_dir)
    create_dir(seg_subj_dir)

    mask = load_nii(os.path.join(seg_subj_dir, "cube_mask.nii.gz"))
    mask[np.where(mask == 0)] = 0.333
    for scan_name in os.listdir(in_subj_dir):
        scan = load_nii(os.path.join(in_subj_dir, scan_name))
        scan = np.multiply(scan, mask)
        save_nii(scan, os.path.join(seg_subj_dir, scan_name))

    return


# --------------- #
# Implementations #
# --------------- #

# Current working directory
cwd = os.getcwd()

# b_mask_path = os.path.join(cwd, "Template", "MNI152_T1_1mm_brain_mask.nii.gz")
cb_mask_path = os.path.join(cwd, "Template", "MNI-maxprob-thr0-1mm.nii.gz")
# edge_mask_path = os.path.join(cwd, "Template", "MNI-maxprob-thr50-1mm.nii.gz")

# cb_mask = load_nii(cb_mask_path)
# cb_mask[np.where(cb_mask != 2)] = 1
# cb_mask[np.where(cb_mask == 2)] = 0

# structure1 = np.ones((3, 3, 3))
# structure2 = sphere([3, 3, 3], 1, [1, 1, 1])

# b_mask = binary_fill_holes(load_nii(b_mask_path)).astype(np.uint8)
# b_mask[:, :, 0] = np.zeros([182, 218])
# edge_mask = load_nii(edge_mask_path)
# edge_mask[np.where(edge_mask > 0)] = 1

# mask = np.multiply(edge_mask + b_mask, cb_mask)
# mask[np.where(mask == 2)] = 0
# mask = binary_opening(mask, structure1)
# mask = 1 - binary_fill_holes(binary_closing(mask, structure1))
# save_nii(mask.astype(np.uint8), "edge_mask.nii.gz")

# eroded_mask = binary_erosion(mask, structure2).astype(np.uint8)
# edge_mask = np.multiply(np.multiply(eroded_mask, edge_mask), cb_mask) + cb_mask
# edge_mask[np.where(edge_mask != 1)] = 0
# edge_mask = 1 - edge_mask.astype(bool) * 1
# eroded_edge = mask - eroded_mask
# new_edge_mask = edge_mask - eroded_mask
# save_nii(edge_mask, "new_edge_mask.nii.gz")
# cb_mask_path = "new_edge_mask.nii.gz"


# -------------------------------------------- #
# Implementation of Step 1: Compute Difference #
# -------------------------------------------- #

print("\nTumor Segmentation Step 1: Compte Difference\n")

cdiff_input_dir = os.path.join(cwd, "FlairT1cePrep")
cdiff_output_dir = os.path.join(cwd, "FlairT1ceDiff")

subjects = os.listdir(cdiff_input_dir)
subj_num = len(subjects)

input_subj_dirs = [os.path.join(cdiff_input_dir, subj) for subj in subjects]
output_subj_dirs = [os.path.join(cdiff_output_dir, subj) for subj in subjects]

bins_num = 256


# Test
# difference(input_subj_dirs[0], output_subj_dirs[0], cb_mask_path, bins_num)

# Multi-processing
paras = zip(input_subj_dirs, output_subj_dirs,
            [cb_mask_path] * subj_num, [bins_num] * subj_num)
pool = Pool(processes=cpu_count())
# pool.map(unwarp_difference, paras)


# ------------------------------------------- #
# Implementation of Step 2: Compute Histogram #
# ------------------------------------------- #

print("\nTumor Segmentation Step 2: Plot Histogram\n")

diff_volume_paths = [os.path.join(subj_dir, "diff.nii.gz") for subj_dir in output_subj_dirs]
# plot_hist(diff_volume_paths, bins_num)


# -------------------------------------- #
# Implementation of Step 3: Bounding Box #
# -------------------------------------- #

print("\nTumor Segmentation Step 3: Cluster and Segment\n")

input_subj_dirs = [os.path.join(cdiff_output_dir, subj) for subj in subjects]
segment_output_dir = os.path.join(cwd, "TumorSegment")
output_subj_dirs = [os.path.join(segment_output_dir, subj) for subj in subjects]

# Test
# tumor_segment(input_subj_dirs[0], output_subj_dirs[0])

# Multi-processing
paras = zip(input_subj_dirs, output_subj_dirs)
pool = Pool(processes=cpu_count())
pool.map(unwarp_tumor_segment, paras)


# --------------------------------------- #
# Implementation of Step 4: Emphsis Tumor #
# --------------------------------------- #

volume_dir = os.path.join(cwd, "FlairT1ceN4BFC")
input_subj_dirs = [os.path.join(volume_dir, subj) for subj in subjects]
seg_subj_dirs = [os.path.join(segment_output_dir, subj) for subj in subjects]

# Test
# mark_tumor(input_subj_dirs[0], seg_subj_dirs[0])

# Multi-processing
paras = zip(input_subj_dirs, seg_subj_dirs)
pool = Pool(processes=cpu_count())
pool.map(unwarp_mark_tumor, paras)
