import os
import numpy as np
import pandas as pd
from skimage import measure
from btc_settings import *
import matplotlib.pyplot as plt
import scipy.ndimage as sn
import scipy.ndimage.morphology as snm
from math import factorial
from scipy.ndimage.interpolation import zoom


# Display Original Volume and Mask

parent_dir = os.path.dirname(os.getcwd())
input_dir = os.path.join(parent_dir, DATA_FOLDER, PREPROCESSED_FOLDER)
input_mask_dir = os.path.join(input_dir, MASK_FOLDER)
input_full_dir = os.path.join(input_dir, FULL_FOLDER)
labels_path = os.path.join(parent_dir, DATA_FOLDER, "label.csv")

labels = pd.read_csv(labels_path)
grade_ii_index = np.where(labels["Grade_Label"] == GRADE_II_LABEL)[0]
grade_ii_cases = labels["Case"][grade_ii_index].values.tolist()

volume_idx = 9
volume_name = grade_ii_cases[volume_idx]
volume_full_path = os.path.join(input_full_dir, volume_name + TARGET_EXTENSION)
volume_mask_path = os.path.join(input_mask_dir, volume_name + TARGET_EXTENSION)

full = np.load(volume_full_path)
mask = np.load(volume_mask_path)

slice_no = int((mask.shape[2] + 1) / 2)
vmin = np.min(full)
vmax = np.max(full)


def plot_one_slice(full, mask, slice_no):
    plt.subplot(2, 3, 1)
    plt.axis("off")
    plt.title("flair")
    plt.imshow(full[:, :, slice_no, 0], cmap="gray", vmin=vmin, vmax=vmax)
    plt.subplot(2, 3, 2)
    plt.axis("off")
    plt.title("t1")
    plt.imshow(full[:, :, slice_no, 1], cmap="gray", vmin=vmin, vmax=vmax)
    plt.subplot(2, 3, 4)
    plt.axis("off")
    plt.title("t1Gd")
    plt.imshow(full[:, :, slice_no, 2], cmap="gray", vmin=vmin, vmax=vmax)
    plt.subplot(2, 3, 5)
    plt.axis("off")
    plt.title("t2")
    plt.imshow(full[:, :, slice_no, 3], cmap="gray", vmin=vmin, vmax=vmax)
    plt.subplot(2, 3, (3, 6))
    plt.axis("off")
    plt.title("mask")
    plt.imshow(mask[:, :, slice_no], cmap="gray")
    plt.show()


plot_one_slice(full, mask, slice_no)


# Erode and Dilate Mask
kernel = sn.generate_binary_structure(3, 1).astype(np.float32)
binary_mask = np.logical_and(mask != ED_MASK, mask != ELSE_MASK) * 1.0
dilated_mask = snm.binary_dilation(binary_mask, structure=kernel, iterations=5)
eroded_mask = snm.binary_erosion(binary_mask, structure=kernel, iterations=5)


binary_mask = snm.binary_fill_holes(binary_mask)
dilated_mask = snm.binary_fill_holes(dilated_mask)
eroded_mask = snm.binary_fill_holes(eroded_mask)


def plot_masks(binary, dilated, eroded, slice_no):
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Dilated")
    plt.imshow(dilated_mask[:, :, slice_no], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("Original")
    plt.imshow(binary_mask[:, :, slice_no], cmap="gray")
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("Eroded")
    plt.imshow(eroded_mask[:, :, slice_no], cmap="gray")
    plt.show()


plot_masks(binary_mask, dilated_mask, eroded_mask, slice_no)


def extract_tumors(mask, full):

    def remove_small_object(mask):
        temp = (mask > 0) * 1.0
        blobs = measure.label(temp, background=0)
        labels = np.unique(blobs)[1:]
        labels_num = [len(np.where(blobs == l)[0]) for l in labels]
        label_idx = np.where(np.array(labels_num) > TUMOT_MIN_SIZE)[0]
        temp = np.zeros(mask.shape)
        for li in label_idx:
            temp[blobs == labels[li]] = 1.0

        return np.where(temp > 0)

    def compute_dims_range(index):
        dims_min = np.array([np.min(i) for i in index])
        dims_max = np.array([np.max(i) + 1 for i in index])
        dims_len = dims_max - dims_min + 1
        dims_len_max = np.max(dims_len)
        dims_begin = []
        dims_end = []
        for dmin, dmax, dlen in zip(dims_min, dims_max, dims_len):
            if dlen == dims_len_max:
                dims_begin.append(dmin)
                dims_end.append(dmax)
            else:
                diff = dims_len_max - dlen
                diff_left = int(diff / 2)
                diff_right = diff - diff_left
                dims_begin.append(dmin - diff_left)
                dims_end.append(dmax + diff_right)

        return dims_begin, dims_end

    # Function to extract sub-array from given array
    # according to ranges of indices of three axes
    def sub_array(arr, begin, end):
        arr_shape = arr.shape
        if len(arr_shape) == CHANNELS:
            bg = [np.min(arr[..., i]) for i in range(CHANNELS)]
            bg = np.array(bg)
        else:
            bg = np.min(arr)
        new_begin = []
        begin_diff = []
        new_end = []
        end_diff = []
        for i in range(len(begin)):
            if begin[i] >= 0:
                new_begin.append(begin[i])
                begin_diff.append(0)
            else:
                new_begin.append(0)
                begin_diff.append(np.abs(begin[i]))
            if end[i] <= arr_shape[i] - 1:
                new_end.append(end[i] + 1)
                end_diff.append(0)
            else:
                new_end.append(arr_shape[i] - 1)
                end_diff.append(end[i] - arr_shape[i] + 2)

        sub_arr = arr[new_begin[0]:new_end[0],
                      new_begin[1]:new_end[1],
                      new_begin[2]:new_end[2]]

        for i in range(len(begin_diff)):
            temp_shape = list(sub_arr.shape)
            if begin_diff[i] > 0:
                temp_shape[i] = begin_diff[i]
                temp_arr = np.multiply(np.ones(temp_shape), bg)
                sub_arr = np.concatenate((temp_arr, sub_arr), axis=i)
            if end_diff[i] > 0:
                temp_shape[i] = end_diff[i]
                temp_arr = np.multiply(np.ones(temp_shape), bg)
                sub_arr = np.concatenate((sub_arr, temp_arr), axis=i)

        return sub_arr.astype(arr.dtype)

    tumor_index = remove_small_object(mask)
    dims_begin, dims_end = compute_dims_range(tumor_index)

    tumor_mask = sub_array(mask, dims_begin, dims_end)
    tumor_full = sub_array(full, dims_begin, dims_end)

    return tumor_mask, tumor_full


bmask, bfull = extract_tumors(binary_mask, full)
dmask, dfull = extract_tumors(dilated_mask, full)
emask, efull = extract_tumors(eroded_mask, full)


def plot_three_volume(bmask, bfull, dmask, dfull, emask, efull):
    bno = int((bmask.shape[0] + 1) / 2)
    dno = int((dmask.shape[0] + 1) / 2)
    eno = int((emask.shape[0] + 1) / 2)

    plt.subplot(2, 3, 1)
    plt.axis("off")
    plt.title("Dilated " + str(dmask.shape[0]))
    plt.imshow(dfull[:, :, dno, 0], cmap="gray")
    plt.subplot(2, 3, 4)
    plt.axis("off")
    plt.imshow(dmask[:, :, dno], cmap="gray")

    plt.subplot(2, 3, 2)
    plt.axis("off")
    plt.title("Original " + str(bmask.shape[0]))
    plt.imshow(bfull[:, :, bno, 0], cmap="gray")
    plt.subplot(2, 3, 5)
    plt.axis("off")
    plt.imshow(bmask[:, :, bno], cmap="gray")

    plt.subplot(2, 3, 3)
    plt.axis("off")
    plt.title("Eroded " + str(emask.shape[0]))
    plt.imshow(efull[:, :, eno, 0], cmap="gray")
    plt.subplot(2, 3, 6)
    plt.axis("off")
    plt.imshow(emask[:, :, eno], cmap="gray")

    plt.show()


plot_three_volume(bmask, bfull, dmask, dfull, emask, efull)


def resize_tumor(full, mask, shape=[59, 59, 59, 4]):
    bg = np.array([np.min(full[..., i]) for i in range(CHANNELS)])
    temp_full = np.multiply(np.ones(full.shape), bg)
    non_bg_index = np.where(mask > 0)
    temp_full[non_bg_index] = full[non_bg_index]

    full_shape = list(full.shape)
    factor = [ns / float(vs) for ns, vs in zip(shape, full_shape)]
    resize_full = zoom(temp_full, zoom=factor, order=3, prefilter=False)
    resize_full = resize_full.astype(full.dtype)

    return resize_full


rbfull = resize_tumor(bfull, bmask, shape=[59, 59, 59, 4])
rdfull = resize_tumor(dfull, dmask, shape=[59, 59, 59, 4])
refull = resize_tumor(efull, emask, shape=[59, 59, 59, 4])


def plot_resize(rbfull, rdfull, refull):
    no = int((rbfull.shape[0] + 1) / 2)

    vmax = np.max([np.max(rbfull[..., 0]),
                   np.max(rdfull[..., 0]),
                   np.max(refull[..., 0])])

    vmin = np.min([np.min(rbfull[..., 0]),
                   np.min(rdfull[..., 0]),
                   np.min(refull[..., 0])])

    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Dilated")
    plt.imshow(rdfull[:, :, no, 0], cmap="gray", vmin=vmin, vmax=vmax)

    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("Original")
    plt.imshow(rbfull[:, :, no, 0], cmap="gray", vmin=vmin, vmax=vmax)

    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("Eroded")
    plt.imshow(refull[:, :, no, 0], cmap="gray", vmin=vmin, vmax=vmax)

    plt.show()


plot_resize(rbfull, rdfull, refull)


# Modify Intensity


def compute_hist(volume):
    rvolume = np.round(volume)
    hist = np.histogram(rvolume, bins=np.arange(np.min(rvolume), np.max(rvolume)), density=True)
    x = hist[1][1:]
    y = hist[0]
    i = np.where(y > 0)
    bg_index = i[0][1] + int((i[0][-1] - i[0][1]) / 20.)
    bg = x[bg_index]
    x = x[bg_index:]
    y = y[bg_index:]
    ysg = savitzky_golay(y, window_size=31, order=4)
    return x, y, ysg, bg


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


channel = 0
original = rbfull[..., channel]
hmirror = np.copy(original)
vmirror = np.copy(original)
amirror = np.copy(original)
x, y, ysg, bg = compute_hist(rbfull[..., channel])
print("Original mean: {0}, original std: {1}".format(np.mean(original), np.std(original)))


def horizontal_mirror(volume):
    temp = np.copy(volume)
    for i in range(temp.shape[2]):
        temp[:, :, i] = np.fliplr(temp[:, :, i])
    return temp


def vertical_mirror(volume):
    temp = np.copy(volume)
    for i in range(temp.shape[2]):
        temp[:, :, i] = np.flipud(temp[:, :, i])
    return temp


def axisymmetric_mirror(volume):
    temp = np.copy(volume)
    temp = horizontal_mirror(temp)
    temp = vertical_mirror(temp)
    return temp


hmirror = horizontal_mirror(hmirror)
vmirror = vertical_mirror(vmirror)
amirror = axisymmetric_mirror(amirror)


def plot_two_curves(x1, y1, label1, x2, y2, label2):
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.legend()
    plt.show()


def modify_intensity(volume):
    temp = np.reshape(volume, ((1, -1)))[0]
    non_bg_index = np.where(temp > 0)
    for i in non_bg_index:
        sign = np.random.randint(2, size=1)[0] * 2 - 1
        scope = np.random.randint(3, 11, size=1)[0] / 100
        temp[i] = temp[i] * (1 + sign * scope)
    return np.reshape(temp, ((59, 59, 59)))


def plot_compare(original, mirror, x1, y1, label1, x2, y2, label2):
    no = int((original.shape[0] + 1) / 2)

    vmax = np.max([np.max(original),
                   np.max(mirror)])

    vmin = np.min([np.min(original),
                   np.min(mirror)])

    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.title(label1)
    plt.imshow(original[:, :, no], cmap="gray", vmin=vmin, vmax=vmax)

    plt.subplot(2, 2, 3)
    plt.axis("off")
    plt.title(label2)
    plt.imshow(mirror[:, :, no], cmap="gray", vmin=vmin, vmax=vmax)

    plt.subplot(2, 2, (2, 4))
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.legend()

    plt.show()


mi_hmirror = modify_intensity(hmirror)
mi_vmirror = modify_intensity(vmirror)
mi_amirror = modify_intensity(amirror)

mhx, mhy, mhysg, mhbg = compute_hist(mi_hmirror)
mvx, mvy, mvysg, mhbg = compute_hist(mi_vmirror)
maax, may, maysg, mhbg = compute_hist(mi_amirror)

print("Horizontal mirror - modified mean: {0}, modified std: {1}".format(np.mean(mi_hmirror), np.std(mi_hmirror)))
print("Vertical mirror - modified mean: {0}, modified std: {1}".format(np.mean(mi_vmirror), np.std(mi_vmirror)))
print("Axisymmetric mirror - modified mean: {0}, modified std: {1}".format(np.mean(mi_amirror), np.std(mi_amirror)))

# plot_two_curves(x, ysg, "Original", mhx, mhysg, "Intensity Modified")
# plot_two_curves(x, ysg, "Original", mvx, mvysg, "Intensity Modified")
# plot_two_curves(x, ysg, "Original", maax, maysg, "Intensity Modified")

plot_compare(original, mi_hmirror, x, ysg, "Original", mhx, mhysg, "Horizontal")
plot_compare(original, mi_vmirror, x, ysg, "Original", mvx, mvysg, "Vertical")
plot_compare(original, mi_amirror, x, ysg, "Original", maax, maysg, "Axisymmetric")
