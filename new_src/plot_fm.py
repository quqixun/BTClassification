from __future__ import print_function


import os
import numpy as np
import nibabel as nib
from scipy.misc import imsave
import matplotlib.pyplot as plt


def get_middle_slice(path):
    volume = nib.load(path).get_data()
    volume = np.transpose(volume, [1, 0, 2])
    volume = np.flipud(volume)
    return volume[..., volume.shape[-1] // 2]


def gallery(array, ncols=16):
    nindex, height, width = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    result = (array.reshape(nrows, ncols, height, width)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols))
    return result


def get_slices(dir_path):
    fm = []
    for n in os.listdir(dir_path):
        one_slice = get_middle_slice(os.path.join(dir_path, n))
        one_slice /= np.max(one_slice)
        rows, cols = one_slice.shape
        roww, colw = int(rows * 0.2), int(cols * 0.2)
        new_slice = np.ones([rows + 2 * roww, cols + 2 * colw])
        new_slice[roww:rows + roww, colw:cols + colw] = one_slice
        fm.append(new_slice)
    return np.array(fm)


def plot_fm(dir_path, save_path, ncols=16):
    fm = get_slices(dir_path)
    results = gallery(fm, ncols)
    imsave(save_path, results)
    plt.imshow(results, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


parent_dir = os.path.dirname(os.getcwd())
fm_dir = os.path.join(parent_dir, "feature_maps")
# plot_fm("conv4", "conv4.png", ncols=16)
# plot_fm("conv5", "conv5.png", ncols=16)
# plot_fm("conv6", "conv6.png", ncols=8)
# plot_fm("conv7", "conv7.png", ncols=8)

# plot_fm(os.path.join(fm_dir, "scale0"), "scale0.png", ncols=16)
plot_fm(os.path.join(fm_dir, "scale1"), "scale1.png", ncols=16)
# plot_fm(os.path.join(fm_dir, "scale2"), "scale2.png", ncols=8)
# plot_fm(os.path.join(fm_dir, "scale3"), "scale3.png", ncols=8)
