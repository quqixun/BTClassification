import os
import numpy as np
from btc_settings import *
import matplotlib.pyplot as plt


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, DATA_FOLDER, SLICES_FOLDER)
slices_dir = os.listdir(data_dir)

dir_idx = 3
#slice_dir = os.path.join(data_dir, slices_dir[dir_idx])
slice_dir = "/home/qixun/btc/data/Slices/TCGA-CS-6666"
files = os.listdir(slice_dir)

for f in files:
    v = np.load(os.path.join(slice_dir, f))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.imshow(v[:, :, 0], cmap="gray")
    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.imshow(v[:, :, 1], cmap="gray")
    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.imshow(v[:, :, 2], cmap="gray")
    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.imshow(v[:, :, 3], cmap="gray")
    plt.show()


# path = "/home/qixun/btc/data/Slices/TCGA-CS-6666/133.npy"
# v = np.load(path)
# plt.figure()
# plt.subplot(2, 2, 1)
# plt.axis("off")
# plt.imshow(v[:, :, 0], cmap="gray")
# plt.subplot(2, 2, 2)
# plt.axis("off")
# plt.imshow(v[:, :, 1], cmap="gray")
# plt.subplot(2, 2, 3)
# plt.axis("off")
# plt.imshow(v[:, :, 2], cmap="gray")
# plt.subplot(2, 2, 4)
# plt.axis("off")
# plt.imshow(v[:, :, 3], cmap="gray")
# plt.show()
