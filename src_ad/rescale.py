from __future__ import print_function

import os
import numpy as np
from tqdm import *
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom


#
# Helper functions
#


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_nii(path):
    data = nib.load(path)
    return data.get_data(), data.affine


def save_nii(data, path, affine):
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, path)
    return


def rescale(data, shape=[96, 96, 112]):
    factors = [float(t) / float(s) for s, t in zip(data.shape, shape)]
    rescaled = zoom(data, zoom=factors, order=1, prefilter=False)
    return rescaled


def norm(data):
    none_bg_idx = np.where(data > 0)
    data_obj = data[none_bg_idx]

    obj_mean = np.mean(data_obj)
    obj_std = np.mean(data_obj)
    obj_norm = (data_obj - obj_mean) / obj_std

    data[none_bg_idx] = obj_norm
    return data


def plot_middle(data):
    plt.figure()
    plt.imshow(data[..., data.shape[-1] // 2], cmap="gray")
    plt.show()
    return


parent_dir = os.path.dirname(os.getcwd())

# For separate subjects
# in_dir = os.path.join(parent_dir, "data", "trimmed_110-110-110_sepsubj")
# out_dir = os.path.join(parent_dir, "data", "sepsubj")

# out_paths = []
# for in_data_set in tqdm(os.listdir(in_dir)):
#     in_data_set_dir = os.path.join(in_dir, in_data_set)
#     out_data_set_dir = os.path.join(out_dir, in_data_set)

#     for data_type in tqdm(os.listdir(in_data_set_dir)):
#         in_data_type_dir = os.path.join(in_data_set_dir, data_type)
#         out_data_type_dir = os.path.join(out_data_set_dir, data_type)
#         create_dir(out_data_type_dir)

#         for data_name in tqdm(os.listdir(in_data_type_dir)):
#             in_data_path = os.path.join(in_data_type_dir, data_name)
#             out_data_path = os.path.join(out_data_type_dir, data_name)

#             out_paths.append(out_data_path)
#             data, affine = load_nii(in_data_path)
#             rescaled = rescale(data)
#             # normed = norm(rescaled)
#             save_nii(rescaled, out_data_path, affine=affine)

# data, _ = load_nii(out_paths[0])
# data = np.rot90(np.transpose(data, [0, 2, 1]), 1)
# plot_middle(data)


# For full processed subjects
in_dir = os.path.join(parent_dir, "data", "trimmed")
out_dir = os.path.join(parent_dir, "data", "trimmed_112")

out_paths = []
for in_data_set in tqdm(os.listdir(in_dir)):
    in_data_set_dir = os.path.join(in_dir, in_data_set)
    out_data_set_dir = os.path.join(out_dir, in_data_set)
    create_dir(out_data_set_dir)

    for data_name in tqdm(os.listdir(in_data_set_dir)):
        in_data_path = os.path.join(in_data_set_dir, data_name)
        out_data_path = os.path.join(out_data_set_dir, data_name)

        out_paths.append(out_data_path)
        data, affine = load_nii(in_data_path)
        rescaled = rescale(data)
        # save_nii(rescaled, out_data_path, affine=affine)


data, _ = load_nii(out_paths[0])
data = np.rot90(np.transpose(data, [0, 2, 1]), 1)
plot_middle(data)
