import os
import shutil
import zipfile
import subprocess

import dicom
import dcmstack
from tqdm import *
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import nipype.interfaces.fsl as fsl
from scipy.ndimage.interpolation import zoom


def load_data(path, rotate=None):
    data = nib.load(path)
    data = data.get_data()
    if rotate:
        data = np.rot90(data, rotate)
    return data


def plot_data(data, name):
    plt.figure(num=name)
    row = 5
    col = data.shape[-1] // row + 1
    for i in range(data.shape[-1]):
        plt.subplot(row, col, i + 1)
        plt.axis("off")
        plt.imshow(data[..., i], cmap="gray")
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


def plot_middle_one(data):
    plt.figure()
    plt.axis("off")
    plt.imshow(data[..., data.shape[-1] // 2], cmap="gray")
    plt.show()
    return


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data")
new_data_dir = os.path.join(parent_dir, "new_data")
if not os.path.isdir(new_data_dir):
    os.makedirs(new_data_dir)

zipfiles = [os.path.join(data_dir, zf) for zf in os.listdir(data_dir)]
exafiles = [os.path.join(new_data_dir, zf.split(".")[0]) for zf in os.listdir(data_dir)]

index = 4
bet_frac = 0.2
zoom_order = 1
new_shape = [110, 110, 110]

zf = zipfiles[index]
ef = exafiles[index]

print "Unzipping files in %s" % zf
with zipfile.ZipFile(zf, "r") as zip_ref:
    zip_ref.extractall(ef)

case_dir = os.path.join(ef, os.listdir(ef)[0])
print(case_dir)
new_case_dir = os.path.join(ef, "new")

for subcase in os.listdir(case_dir):
    deco_dir = os.path.join(new_case_dir, "decompress")
    if not os.path.isdir(deco_dir):
        os.makedirs(deco_dir)

    input_dir = os.path.join(case_dir, subcase)
    input_dcms = [os.path.join(input_dir, dcm_file) for dcm_file in os.listdir(input_dir)]
    deco_dcms = [os.path.join(deco_dir, dcm_file) for dcm_file in os.listdir(input_dir)]

    print "Decompress DICOM files ..."
    for i in tqdm(range(len(input_dcms))):
        command = ["gdcmconv", "-w", input_dcms[i], deco_dcms[i]]
        subprocess.call(command)

    my_stack = dcmstack.DicomStack()
    for src_path in deco_dcms:
        src_dcm = dicom.read_file(src_path)
        my_stack.add_dcm(src_dcm)

    stack_data = my_stack.get_data()
    stack_shape = stack_data.shape
    print(stack_shape)
    if max(stack_shape) > 256 or stack_shape.index(min(stack_shape)) != 2:
        continue

    nii = my_stack.to_nifti()
    original_file_path = os.path.join(new_case_dir, subcase + ".nii.gz")
    nii.to_filename(original_file_path)

    noskull_file_path = os.path.join(new_case_dir, subcase + "_noskull.nii")
    mybet = fsl.BET(in_file=original_file_path,
                    out_file=noskull_file_path,
                    frac=bet_frac)
    result = mybet.run()

    original = load_data(original_file_path, 1)
    noskull = load_data(noskull_file_path, 1)

    plot_middle_two(original, noskull)

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

    bgtrimed = np.zeros([len_of_side, len_of_side, noskull.shape[-1]])
    for i in range(noskull.shape[-1]):
        bgtrimed[..., i] = noskull[row_begin:row_end + 1,
                                   col_begin:col_end + 1, i]

    old_shape = list(bgtrimed.shape)
    factor = [n / float(o) for n, o in zip(new_shape, old_shape)]
    resized = zoom(bgtrimed, zoom=factor, order=1, prefilter=False)

    plot_middle_two(bgtrimed, resized)

    resized = resized.astype(np.int16)
    resized_nii = nib.Nifti1Image(resized, np.eye(4))
    resized_nii_path = os.path.join(new_case_dir, subcase + ".nii.gz")
    nib.save(resized_nii, resized_nii_path)

    new_data = load_data(resized_nii_path)
    plot_middle_one(new_data)
    plt.show()

    shutil.rmtree(deco_dir)
    os.remove(noskull_file_path)
