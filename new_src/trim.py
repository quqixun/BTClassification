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
    # plt.imshow(data[..., 100], cmap="gray")
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


def segment(in_path, seg_path):
    volume = np.rot90(nib.load(in_path).get_data(), 3)
    mask = np.rot90(nib.load(seg_path).get_data(), 3)
    # plot_middle_two(volume, mask)
    if np.min(volume) != 0:
        volume -= np.min(volume)

    non_mask_idx = np.where(mask == 0)
    segged = np.copy(volume)
    segged[non_mask_idx] = segged[non_mask_idx] * 0.2
    # plot_middle_two(volume, segged)

    return segged


def trim(volume):
    non_zero_slices = [i for i in range(volume.shape[-1]) if np.sum(volume[..., i]) > 0]
    volume = volume[..., non_zero_slices]

    row_begins, row_ends = [], []
    col_begins, col_ends = [], []
    for i in range(volume.shape[-1]):
        non_zero_pixels = np.where(volume > 0)
        row_begins.append(np.min(non_zero_pixels[0]))
        row_ends.append(np.max(non_zero_pixels[0]))
        col_begins.append(np.min(non_zero_pixels[1]))
        col_ends.append(np.max(non_zero_pixels[1]))

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

    trimmed = np.zeros([len_of_side, len_of_side, volume.shape[-1]])
    for i in range(volume.shape[-1]):
        trimmed[..., i] = volume[row_begin:row_end + 1,
                                 col_begin:col_end + 1, i]
    return trimmed


def resize(trimmed, target_shape):
    old_shape = list(trimmed.shape)
    factor = [n / float(o) for n, o in zip(target_shape, old_shape)]
    resized = zoom(trimmed, zoom=factor, order=1, prefilter=False)
    # plot_middle_two(trimmed, resized)
    return resized


def save2nii(to_path, resized):
    resized = resized.astype(np.int16)
    resized = np.rot90(resized, 3)
    resized_nii = nib.Nifti1Image(resized, np.eye(4))
    nib.save(resized_nii, to_path)
    return


def unwrap_rescale(arg, **kwarg):
    return rescale(*arg, **kwarg)


def rescale(target_shape, in_path, to_path, seg_path):
    try:
        print("Rescaling on: " + in_path)
        segged = segment(in_path, seg_path)
        trimmed = trim(segged)
        resized = resize(trimmed, target_shape)
        save2nii(to_path, resized)
    except:
        print("  Failed to rescal:" + in_path)
        return
    return


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def generate_paths(in_dir, out_dir):
    if not os.path.isdir(in_dir):
        raise IOError("Input folder is not exist.")

    create_dir(out_dir)

    scan_paths, scan2paths = [], []
    scan_seg_paths = []
    for subject in os.listdir(in_dir):
        subject_dir = os.path.join(in_dir, subject)
        subject2dir = os.path.join(out_dir, subject)
        create_dir(subject2dir)

        scan_names = os.listdir(subject_dir)

        for scan_name in scan_names:
            if "seg" in scan_name:
                scan_seg_path = os.path.join(subject_dir, scan_name)

        for scan_name in scan_names:
            if "seg" in scan_name:
                continue
            else:
                scan_paths.append(os.path.join(subject_dir, scan_name))
                scan2paths.append(os.path.join(subject2dir, scan_name))
                scan_seg_paths.append(scan_seg_path)

    return scan_paths, scan2paths, scan_seg_paths


if __name__ == "__main__":

    target_shape = [112, 112, 112]

    parent_dir = os.path.dirname(os.getcwd())
    hgg_input_dir = os.path.join(parent_dir, "data", "Original", "BraTS", "HGG")
    lgg_input_dir = os.path.join(parent_dir, "data", "Original", "BraTS", "LGG")
    hgg_output_dir = os.path.join(parent_dir, "data", "Original", "BraTS", "HGGTrimmed")
    lgg_output_dir = os.path.join(parent_dir, "data", "Original", "BraTS", "LGGTrimmed")

    hgg_paths, hgg2paths, hgg_seg_paths = generate_paths(hgg_input_dir, hgg_output_dir)
    lgg_paths, lgg2paths, lgg_seg_paths = generate_paths(lgg_input_dir, lgg_output_dir)

    in_paths = hgg_paths + lgg_paths
    out_paths = hgg2paths + lgg2paths
    seg_paths = hgg_seg_paths + lgg_seg_paths

    # rescale(target_shape, in_paths[0], out_paths[0], seg_paths[0])
    paras = zip([target_shape] * len(in_paths),
                in_paths, out_paths, seg_paths)
    pool = Pool(processes=cpu_count())
    pool.map(unwrap_rescale, paras)
