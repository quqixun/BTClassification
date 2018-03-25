from __future__ import print_function

import os
import subprocess
# import numpy as np
from tqdm import *
import nibabel as nib
import matplotlib.pyplot as plt


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


def plot_middle(data):
    plt.figure()
    plt.imshow(data[..., data.shape[-1] // 2], cmap="gray")
    plt.show()
    return


parent_dir = os.path.dirname(os.getcwd())
data_src_dir = os.path.join(parent_dir, "data", "ADNI")
data_dst_dir = os.path.join(parent_dir, "data", "Processed")
create_dir(data_dst_dir)
labels = ["AD", "NC"]

scan_src_paths, scan_dst_paths = [], []
for label in labels:
    label_src_dir = os.path.join(data_src_dir, label)
    label_dst_dir = os.path.join(data_dst_dir, label)
    create_dir(label_dst_dir)

    subjects = os.listdir(label_src_dir)
    for subject in subjects:
        subj_src_dir = os.path.join(label_src_dir, subject)
        subj_dst_dir = os.path.join(label_dst_dir, subject)
        create_dir(subj_dst_dir)

        for scan in os.listdir(subj_src_dir):
            scan_src_paths.append(os.path.join(subj_src_dir, scan))
            scan_dst_paths.append(os.path.join(subj_dst_dir, scan))


# x, y, z = [], [], []
# for scan in tqdm(scan_src_paths):
#     data, affine = load_nii(scan)
#     data = data.reshape(data.shape[:3])
#     # data = np.rot90(data, 1)
#     # print(data.shape, affine)
#     # plot_middle(data)
#     ax, ay, az = np.where(data > 0)
#     x.append(np.max(ax) - np.min(ax) + 1)
#     y.append(np.max(ay) - np.min(ay) + 1)
#     z.append(np.max(az) - np.min(az) + 1)

# print("Max X:", max(x), "Max Y:", max(y), "Max Z:", max(z))


# for i in tqdm(range(len(scan_src_paths[:1]))):
#     data, affine = load_nii(scan_src_paths[i])
#     data = data.reshape(data.shape[:3])
#     data = data[32:-32, 16:-16, 32:-32]
#     save_nii(data, scan_dst_paths[i], affine)
#     command = ["mri_convert", "-ds", "2", "2", "2", scan_dst_paths[i], scan_dst_paths[i]]
#     subprocess.call(command, stdout=open(os.devnull), stderr=subprocess.STDOUT)

data, _ = load_nii(scan_dst_paths[0])
data = np.rot90(data, 1)
print(data.shape)
plot_middle(data)
