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


def resize(in_slice):
    r, c = in_slice.shape
    diff = np.abs(r - c)
    if r > c:
        new_slice = np.zeros((r, r))
        start = diff // 2
        end = start + c
        new_slice[:, start:end] = in_slice
    return new_slice


def rescale(in_slice, target_shape=[224, 224]):
    factors = [t / s for s, t in zip(in_slice.shape, target_shape)]
    resized = zoom(in_slice, zoom=factors, order=1, prefilter=False)
    return resized


def norm(in_slice):
    return (in_slice - np.mean(in_slice)) / np.std(in_slice)


def save_to_dir(slices, mode, out_subj_dir):
    for i in range(len(slices)):
        type_dir = os.path.join(out_subj_dir, mode)
        create_dir(type_dir)
        in_slice = slices[i]
        in_slice_path = os.path.join(type_dir, str(i) + ".npy")
        in_slice_png_path = os.path.join(type_dir, str(i) + ".png")
        np.save(in_slice_path, in_slice)
        scipy.misc.imsave(in_slice_png_path, in_slice)
    return


input_dir = os.path.join(os.getcwd(), "TumorSegment")
output_dir = os.path.join(os.getcwd(), "TumorSegmentSlice")

subjects = os.listdir(input_dir)
for subject in tqdm(subjects):
    in_subj_dir = os.path.join(input_dir, subject)
    out_subj_dir = os.path.join(output_dir, subject)
    create_dir(out_subj_dir)

    mask_path = os.path.join(in_subj_dir, "mask.nii.gz")
    t1ce_path = os.path.join(in_subj_dir, "t1ce.nii.gz")
    flair_path = os.path.join(in_subj_dir, "flair.nii.gz")

    mask = load_nii(mask_path).astype(np.float32)
    t1ce = load_nii(t1ce_path)
    flair = load_nii(flair_path)

    tumor_mask_idx = []
    for i in range(mask.shape[2]):
        if np.sum(mask[..., i]) > 0:
            tumor_mask_idx.append(i)

    idx_len = len(tumor_mask_idx)
    mid_idx = idx_len // 2
    dif_idx = int(idx_len * 0.2)

    extract_idx = [tumor_mask_idx[mid_idx - dif_idx],
                   tumor_mask_idx[mid_idx],
                   tumor_mask_idx[mid_idx + dif_idx]]

    mask[np.where(mask > 0)] = 1
    mask[np.where(mask == 0)] = 0.333
    t1ce = np.multiply(t1ce, mask)
    flair = np.multiply(flair, mask)

    t1Gd_slices = [norm(resize(np.rot90(t1ce[..., idx], 1))) for idx in extract_idx]
    flair_slices = [norm(resize(np.rot90(flair[..., idx], 1))) for idx in extract_idx]
    # mask_slices = [np.rot90(mask[..., idx], 3) for idx in extract_idx]

    # plot_slices(t1Gd_slices)

    save_to_dir(t1Gd_slices, "t1ce", out_subj_dir)
    save_to_dir(flair_slices, "flair", out_subj_dir)
