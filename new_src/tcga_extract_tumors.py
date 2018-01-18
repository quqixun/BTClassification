import os
import scipy.misc
import numpy as np
from tqdm import *
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_nii(path):
    return nib.load(path).get_data()


def plot_slices(slices):
    slice_num = len(slices)
    plt.figure()
    for i in range(slice_num):
        plt.subplot(1, slice_num, i + 1)
        plt.imshow(slices[i], cmap="gray")
        plt.axis("off")
    plt.show()


def trim(volume):
    none_zero_idx = np.where(volume > 0)

    min_i, max_i = np.min(none_zero_idx[0]), np.max(none_zero_idx[0])
    min_j, max_j = np.min(none_zero_idx[1]), np.max(none_zero_idx[1])

    diff_i = max_i - min_i
    diff_j = max_j - min_j

    if diff_i > diff_j:
        diff = diff_i - diff_j
        half_diff = diff // 2
        min_j -= half_diff
        max_j += (diff - half_diff)
    elif diff_i < diff_j:
        diff = diff_j - diff_i
        half_diff = diff // 2
        min_i -= half_diff
        max_i += (diff - half_diff)

    return min_i, max_i, min_j, max_j


def rescale(in_slice, target_shape=[224, 224]):
    factors = [t / s for s, t in zip(in_slice.shape, target_shape)]
    resized = zoom(in_slice, zoom=factors, order=1, prefilter=False)
    return resized


def norm(in_slice):
    return (in_slice - np.mean(in_slice)) / np.std(in_slice)


def save_to_dir(in_slice, mode, out_subj_dir, i):
    type_dir = os.path.join(out_subj_dir, mode)
    create_dir(type_dir)
    in_slice_path = os.path.join(type_dir, str(i) + ".npy")
    in_slice_png_path = os.path.join(type_dir, str(i) + ".png")
    np.save(in_slice_path, in_slice)
    scipy.misc.imsave(in_slice_png_path, in_slice)
    return


parent_dir = os.path.dirname(os.getcwd())
input_dir = os.path.join(parent_dir, "VolumeData", "TCGA")
output_dir1 = os.path.join(parent_dir, "SliceData", "Tumor")
output_dir2 = os.path.join(parent_dir, "SliceData", "EnahnceTumor")
output_dir3 = os.path.join(parent_dir, "SliceData", "SquareTumor")
output_dir4 = os.path.join(parent_dir, "SliceData", "SquareEnhanceTumor")
subjects = os.listdir(input_dir)

print("Extract tumor ...")
subjects_info = []
tumors_info = []
for subject in tqdm(subjects):
    out_subj_dir3 = os.path.join(output_dir3, subject)
    out_subj_dir4 = os.path.join(output_dir4, subject)
    create_dir(out_subj_dir3)
    create_dir(out_subj_dir4)
    in_subj_dir = os.path.join(input_dir, subject)

    mask_path = os.path.join(in_subj_dir, subject + "_mask.nii.gz")
    t1ce_path = os.path.join(in_subj_dir, subject + "_t1Gd.nii.gz")
    t2_path = os.path.join(in_subj_dir, subject + "_t2.nii.gz")

    mask = load_nii(mask_path)
    t1ce = load_nii(t1ce_path)
    t2 = load_nii(t2_path)

    min_i, max_i, min_j, max_j = trim(t1ce)
    mask_min_i, mask_max_i, mask_min_j, mask_max_j = trim(mask)
    t1ce = t1ce[min_i:max_i, min_j:max_j, :]
    t2 = t2[min_i:max_i, min_j:max_j, :]
    mask = mask[min_i:max_i, min_j:max_j, :].astype(np.float32)
    subjects_info.append([subject, t1ce, t2, mask])

    tumor_mask_idx = []
    for i in range(mask.shape[2]):
        if np.sum(mask[..., i]) > 0:
            tumor_mask_idx.append(i)

    idx_len = len(tumor_mask_idx)
    mid_idx = idx_len // 2
    dif_idx = int(idx_len * 0.1)

    extract_idx = [tumor_mask_idx[mid_idx - dif_idx],
                   tumor_mask_idx[mid_idx],
                   tumor_mask_idx[mid_idx + dif_idx]]

    slices_info = []
    for i in range(len(extract_idx)):
        idx = extract_idx[i]
        slice_mask = mask[..., idx]
        mask_min_i, mask_max_i, mask_min_j, mask_max_j = trim(slice_mask)
        mask_center = [(mask_min_i + mask_max_i) // 2,
                       (mask_min_j + mask_max_j) // 2]
        mask_length = np.max([mask_max_i - mask_min_i + 1,
                              mask_max_j - mask_min_j + 1])
        half_mask_length = mask_length // 2
        slices_info.append([idx, mask_center, mask_length])

        # min_i, max_i = mask_center[0] - half_mask_length, mask_center[0] + half_mask_length
        # min_j, max_j = mask_center[1] - half_mask_length, mask_center[1] + half_mask_length

        # print(subject, i, min_i, max_i, min_j, max_j)

        t1ce_slice = t1ce[..., idx]
        t2_slice = t2[..., idx]
        mask_slice = mask[..., idx]

        pad_t1ce_slice = np.pad(t1ce_slice, half_mask_length, "constant")
        pad_t2_slice = np.pad(t2_slice, half_mask_length, "constant")
        pad_mask_slice = np.pad(mask_slice, half_mask_length, "constant")

        min_i, max_i = mask_center[0], mask_center[0] + mask_length
        min_j, max_j = mask_center[1], mask_center[1] + mask_length

        tumor_t1ce_slice = pad_t1ce_slice[min_i:max_i, min_j:max_j]
        tumor_t2_slice = pad_t2_slice[min_i:max_i, min_j:max_j]
        tumor_mask_slice = pad_mask_slice[min_i:max_i, min_j:max_j]

        res_t1ce_slice = norm(rescale(np.rot90(tumor_t1ce_slice, 3)))
        res_t2_slice = norm(rescale(np.rot90(tumor_t2_slice, 3)))

        save_to_dir(res_t1ce_slice, "t1ce", out_subj_dir3, i)
        save_to_dir(res_t2_slice, "t2", out_subj_dir3, i)

        tumor_mask_slice[np.where(tumor_mask_slice > 0)] = 1
        tumor_mask_slice[np.where(tumor_mask_slice == 0)] = 0.333
        tumor_t1ce_slice = np.multiply(tumor_t1ce_slice, tumor_mask_slice)
        tumor_t2_slice = np.multiply(tumor_t2_slice, tumor_mask_slice)

        res_t1ce_slice = norm(rescale(np.rot90(tumor_t1ce_slice, 3)))
        res_t2_slice = norm(rescale(np.rot90(tumor_t2_slice, 3)))

        save_to_dir(res_t1ce_slice, "t1ce", out_subj_dir4, i)
        save_to_dir(res_t2_slice, "t2", out_subj_dir4, i)

    tumors_info.append([subject, slices_info])

print(len(subjects_info), len(tumors_info))

mask_lengths, ss = [], []
for case in tumors_info:
    for sub_case in case[1]:
        mask_lengths.append(sub_case[-1])
        ss.append(case[0])
max_length = max(mask_lengths)
max_idx = mask_lengths.index(max_length)
print(ss[max_idx])
half_max_length = max_length // 2
print(max_length, min(mask_lengths), np.mean(mask_length), np.median(mask_lengths))

for case in tqdm(tumors_info):
    out_subj_dir1 = os.path.join(output_dir1, case[0])
    out_subj_dir2 = os.path.join(output_dir2, case[0])
    create_dir(out_subj_dir1)
    create_dir(out_subj_dir2)

    subject_idx = [i for i, s in enumerate(subjects_info) if s[0] == case[0]][0]

    t1ce = subjects_info[subject_idx][1]
    t2 = subjects_info[subject_idx][2]
    mask = subjects_info[subject_idx][3]

    for i in range(len(case[1])):
        sub_case = case[1][i]
        t1ce_slice = t1ce[..., sub_case[0]]
        t2_slice = t2[..., sub_case[0]]
        mask_slice = mask[..., sub_case[0]]

        pad_t1ce_slice = np.pad(t1ce_slice, half_max_length, "constant")
        pad_t2_slice = np.pad(t2_slice, half_max_length, "constant")
        pad_mask_slice = np.pad(mask_slice, half_max_length, "constant")

        min_i, max_i = sub_case[1][0], sub_case[1][0] + max_length
        min_j, max_j = sub_case[1][1], sub_case[1][1] + max_length

        tumor_t1ce_slice = pad_t1ce_slice[min_i:max_i, min_j:max_j]
        tumor_t2_slice = pad_t2_slice[min_i:max_i, min_j:max_j]
        tumor_mask_slice = pad_mask_slice[min_i:max_i, min_j:max_j]

        res_t1ce_slice = norm(rescale(np.rot90(tumor_t1ce_slice, 3)))
        res_t2_slice = norm(rescale(np.rot90(tumor_t2_slice, 3)))

        save_to_dir(res_t1ce_slice, "t1ce", out_subj_dir1, i)
        save_to_dir(res_t2_slice, "t2", out_subj_dir1, i)

        tumor_mask_slice[np.where(tumor_mask_slice > 0)] = 1
        tumor_mask_slice[np.where(tumor_mask_slice == 0)] = 0.333
        tumor_t1ce_slice = np.multiply(tumor_t1ce_slice, tumor_mask_slice)
        tumor_t2_slice = np.multiply(tumor_t2_slice, tumor_mask_slice)

        res_t1ce_slice = norm(rescale(np.rot90(tumor_t1ce_slice, 3)))
        res_t2_slice = norm(rescale(np.rot90(tumor_t2_slice, 3)))

        save_to_dir(res_t1ce_slice, "t1ce", out_subj_dir2, i)
        save_to_dir(res_t2_slice, "t2", out_subj_dir2, i)
