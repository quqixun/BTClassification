# Script for testing to extract slices
# from brain volumes


import os
import shutil
import numpy as np
from btc_settings import *
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, DATA_FOLDER,
                        PREPROCESSED_FOLDER,
                        FULL_FOLDER)
mask_dir = os.path.join(parent_dir, DATA_FOLDER,
                        PREPROCESSED_FOLDER,
                        MASK_FOLDER)

slice_dir = os.path.join(parent_dir, DATA_FOLDER, "Slices")

file_names = os.listdir(data_dir)

file_idx = 0
# file_name = file_names[file_idx]
file_name = "TCGA-DU-A5TU.npy"
file_no = file_name.split(".")[0]

file_path = os.path.join(data_dir, file_name)
mask_path = os.path.join(mask_dir, file_name)

save_path = os.path.join(slice_dir, file_no)
if os.path.isdir(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

volume = np.load(file_path)
mask = np.load(mask_path)


def remove_edgespace(v):
    vshape = list(v.shape)
    return v[EDGE_SPACE:vshape[0] - EDGE_SPACE,
             EDGE_SPACE:vshape[1] - EDGE_SPACE,
             EDGE_SPACE:vshape[2] - EDGE_SPACE]


def plot_volume_mask(volume, mask):
    num_slices = mask.shape[-1]
    for i in range(num_slices):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title(i)
        plt.imshow(volume[:, :, i, 0], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.imshow(mask[:, :, i], cmap="gray")
        plt.show()

    return


def plot_slices(original, resized):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(resized, cmap="gray")
    plt.show()


volume = remove_edgespace(volume)
mask = remove_edgespace(mask)

mask_thresh = ET_MASK
prop_thresh = 0.2
num_slices = mask.shape[-1]

core_nums = []
for i in range(num_slices):
    core_pos = np.where(mask[:, :, i] >= mask_thresh)
    core_nums.append(len(core_pos[0]))

max_core_num = max(core_nums)
min_core_num = int(max_core_num * prop_thresh)

core_slice_idxs = [i for i in range(num_slices)
                   if core_nums[i] >= min_core_num]

min_core_slice_idx = min(core_slice_idxs)
max_core_slice_idx = max(core_slice_idxs) + 1

volume = volume[:, :, min_core_slice_idx:max_core_slice_idx, :]
vshape = list(volume.shape)

pad_size = vshape[0] - vshape[1]
left_pad_size = int(pad_size / 2.0)
right_pad_size = pad_size - left_pad_size

vshape[1] = left_pad_size
left_pad = np.zeros(vshape)
vshape[1] = right_pad_size
right_pad = np.zeros(vshape)

pad_volume = np.hstack((left_pad, volume, right_pad))
vshape = list(pad_volume.shape)

sshape = vshape.copy()
sshape.pop(2)
factor = [ns / ss for ns, ss in zip(SLICE_SHAPE, sshape)]

for i in range(vshape[2]):
    vslice = pad_volume[:, :, i, :]
    resized_slice = zoom(vslice, zoom=factor, order=1, prefilter=False)
    resized_slice = resized_slice.astype(vslice.dtype)
    save_file_name = str(i) + TARGET_EXTENSION
    np.save(os.path.join(save_path, save_file_name), resized_slice)
