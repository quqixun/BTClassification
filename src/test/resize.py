import os
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom


# Ignore the warning caused by SciPy
warnings.simplefilter("ignore", UserWarning)


def plot_middle_one(data):
    plt.figure()
    plt.axis("off")
    plt.imshow(data[..., data.shape[-1] // 2], cmap="gray")
    plt.show()
    return


def plot_middle_two(data1, data2):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(data1[..., data1.shape[-1] // 2], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(data2[..., data2.shape[-1] // 2], cmap="gray")
    plt.show()
    return


def trim(in_path):
    noskull = nib.load(in_path).get_data()
    noskull = np.transpose(noskull, [0, 2, 1])
    noskull = np.rot90(noskull, 1)
    # plot_middle_one(noskull)
    non_zero_slice_indices = [i for i in range(noskull.shape[-1]) if np.sum(noskull[..., i]) > 0]
    noskull = noskull[..., non_zero_slice_indices]

    row_begins, row_ends = [], []
    col_begins, col_ends = [], []
    for i in range(noskull.shape[-1]):
        non_zero_pixel_indices = np.where(noskull > 0)
        row_begins.append(np.min(non_zero_pixel_indices[0]))
        row_ends.append(np.max(non_zero_pixel_indices[0]))
        col_begins.append(np.min(non_zero_pixel_indices[1]))
        col_ends.append(np.max(non_zero_pixel_indices[1]))

    row_begin, row_end = min(row_begins), max(row_ends)
    col_begin, col_end = min(col_begins), max(col_ends)

    rows_num = row_end - row_begin
    cols_num = col_end - col_begin
    more_col_len = rows_num - cols_num
    more_col_len_left = more_col_len // 2
    more_col_len_right = more_col_len - more_col_len_left
    col_begin -= more_col_len_left
    col_end += more_col_len_right
    len_of_side = rows_num + 1

    trimmed = np.zeros([len_of_side, len_of_side, noskull.shape[-1]])
    for i in range(noskull.shape[-1]):
        trimmed[..., i] = noskull[row_begin:row_end + 1,
                                  col_begin:col_end + 1, i]
    return trimmed


def resize(trimmed):
    old_shape = list(trimmed.shape)
    factor = [n / float(o) for n, o in zip(target_shape, old_shape)]
    resized = zoom(trimmed, zoom=factor, order=1, prefilter=False)
    # plot_middle_two(trimmed, resized)

    return resized


def save2nii(to_path, resized):
    resized = resized.astype(np.int16)
    resized_nii = nib.Nifti1Image(resized, np.eye(4))
    nib.save(resized_nii, to_path)

    # new_data = nib.load(to_path).get_data()
    # plot_middle_one(new_data)
    # plt.show()
    return


def unwrap_rescale(arg, **kwarg):
    return rescale(*arg, **kwarg)


def rescale(target_shape, in_path, to_path):
    try:
        print("Rescaling on: " + in_path)
        trimmed = trim(in_path)
        resized = resize(trimmed)
        save2nii(to_path, resized)
    except:
        print("  Failed to rescal:" + in_path)
        return
    return


target_shape = [110, 110, 110]
# target_shape = [196, 196, 76]
target_shape_str = "_".join(map(str, target_shape))

parent_dir = os.path.dirname(os.getcwd())
resize_dir = os.path.join(parent_dir, "resize_ax", target_shape_str)
noskull_dir = os.path.join(parent_dir, "noskull_ax")
if not os.path.isdir(resize_dir):
    os.makedirs(resize_dir)

all_scan_file_path, all_scan_file2path = [], []
for type_dir in os.listdir(noskull_dir):
    type_dir_path = os.path.join(noskull_dir, type_dir)
    type_dir2path = os.path.join(resize_dir, type_dir)
    if not os.path.isdir(type_dir2path):
        os.makedirs(type_dir2path)
    for subj_dir in os.listdir(type_dir_path):
        subj_dir_path = os.path.join(type_dir_path, subj_dir)
        subj_dir2path = os.path.join(type_dir2path, subj_dir)
        if not os.path.isdir(subj_dir2path):
            os.makedirs(subj_dir2path)
        for scan_file in os.listdir(subj_dir_path):
            all_scan_file_path.append(os.path.join(subj_dir_path, scan_file))
            all_scan_file2path.append(os.path.join(subj_dir2path, scan_file))

# in_path = all_scan_file_path[21]
# to_path = all_scan_file2path[21]
# rescale(target_shape, in_path, to_path)

paras = zip([target_shape] * len(all_scan_file_path),
            all_scan_file_path, all_scan_file2path)
pool = Pool(processes=cpu_count())
pool.map(unwrap_rescale, paras)
