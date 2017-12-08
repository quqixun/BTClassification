import os
import shutil
import zipfile
import subprocess

import dicom
import dcmstack
from tqdm import *
import numpy as np
import pandas as pd
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
# data_dir = os.path.join(parent_dir, "data")
data_dir = "/home/user1/transfer/datasets/BT/Asgeir UCSF/Zipped DICOM"
new_data_dir = os.path.join(parent_dir, "new_data")
if not os.path.isdir(new_data_dir):
    os.makedirs(new_data_dir)

zipfiles = [os.path.join(data_dir, zf) for zf in os.listdir(data_dir)]
exafiles = [os.path.join(new_data_dir, zf.split(".")[0]) for zf in os.listdir(data_dir)]
case_names = [zf.split(".")[0] for zf in os.listdir(data_dir)]

csv_file = os.path.join(new_data_dir, "info.csv")

# index = 0
# FRAC_DEFAULT= 0.2
# zoom_order = 1
# new_shape = [110, 110, 110]

case_no = []
subcase_no = []
case_des = []
run_infos = []

for zf, ef, cn in tqdm(zip(zipfiles, exafiles, case_names)):

    if os.path.isdir(ef):
        shutil.rmtree(ef)
    os.makedirs(ef)

    # print "Unzipping files in: %s" % zf
    with zipfile.ZipFile(zf, "r") as zip_ref:
        zip_ref.extractall(ef)

    case_dir = os.path.join(ef, os.listdir(ef)[0])
    new_case_dir = os.path.join(ef, "new")
    if os.path.isdir(new_case_dir):
        shutil.rmtree(new_case_dir)
    os.makedirs(new_case_dir)

    for subcase in tqdm(os.listdir(case_dir)):
        run_info = ""
        case_no.append(cn)
        subcase_no.append(subcase)
        deco_dir = os.path.join(ef, "decompress")
        if os.path.isdir(deco_dir):
            shutil.rmtree(deco_dir)
        os.makedirs(deco_dir)

        input_dir = os.path.join(case_dir, subcase)
        input_dcms = [os.path.join(input_dir, dcm_file) for dcm_file in os.listdir(input_dir)]
        deco_dcms = [os.path.join(deco_dir, dcm_file) for dcm_file in os.listdir(input_dir)]

        # print "Decompress DICOM files of case %s ..." % subcase
        undecomp = 0
        for i in tqdm(range(len(input_dcms))):
            try:
                command = ["gdcmconv", "-w", input_dcms[i], deco_dcms[i]]
                subprocess.call(command)
            except:
                undecomp += 1
                continue
        run_info += "%dud" % undecomp

        unstack = 0
        my_stack = dcmstack.DicomStack()
        for src_path in deco_dcms:
            try:
                src_dcm = dicom.read_file(src_path)
                my_stack.add_dcm(src_dcm)
            except:
                unstack += 1
                continue
        run_info += " %dus" % unstack

        try:
            case_des.append(src_dcm.SeriesDescription)
        except:
            run_info += " nodes"

        try:
            nii = my_stack.to_nifti()
            original_file_path = os.path.join(new_case_dir, subcase + ".nii.gz")
            nii.to_filename(original_file_path)
        except:
            run_info += " unsaved"

        run_infos.append(run_info)
        shutil.rmtree(deco_dir)

        try:
            case_csv_file = os.path.join(new_case_dir, subcase + ".csv")
            case_info_df = pd.DataFrame(data={"subject": [case_no[-1]], "scans": [subcase_no[-1]], "info": [case_des[-1]], "log": [run_infos[-1]]})
            case_info_df.to_csv(case_csv_file, columns=["subject", "scans", "info", "log"], index=False)
        except:
            continue

info_df = pd.DataFrame(data={"subject": case_no, "scans": subcase_no, "info": case_des, "log": run_infos})
info_df.to_csv(csv_file, columns=["subject", "scans", "info", "log"], index=False)
print("\n")

    # noskull_file_path = os.path.join(new_case_dir, subcase + "_noskull.nii.gz")
    
    # frac = FRAC_DEFAULT
    # is_continue = False
    # while not is_continue:
    #     mybet = fsl.BET(in_file=original_file_path,
    #                     out_file=noskull_file_path)
    #                     # frac=frac)
    #     result = mybet.run()

    #     original = load_data(original_file_path, 1)
    #     noskull = load_data(noskull_file_path, 1)
    #     plot_middle_two(original, noskull)
    #     print "Is continue ? n or y"
    #     continue_symbol = raw_input()
    #     if continue_symbol == "n":
    #         print "Change frac (now is %f)" % frac
    #         frac = float(raw_input())
    #         continue
    #     else:
    #         is_continue = True

    # case_frac.append(frac)
    # shutil.rmtree(deco_dir)

    # non_zero_slice_indices = [i for i in range(noskull.shape[-1]) if np.sum(noskull[..., i]) > 0]
    # noskull = noskull[..., non_zero_slice_indices]

    # row_begins, row_ends = [], []
    # col_begins, col_ends = [], []
    # for i in range(noskull.shape[-1]):
    #     non_zero_pixel_indices = np.where(noskull > 0)
    #     row_begins.append(np.min(non_zero_pixel_indices[0]))
    #     row_ends.append(np.max(non_zero_pixel_indices[0]))
    #     col_begins.append(np.min(non_zero_pixel_indices[1]))
    #     col_ends.append(np.max(non_zero_pixel_indices[1]))

    # row_begin, row_end = min(row_begins), max(row_ends)
    # col_begin, col_end = min(col_begins), max(col_ends)

    # rows_num = row_end - row_begin
    # cols_num = col_end - col_begin
    # more_col_len = rows_num - cols_num
    # more_col_len_left = more_col_len // 2
    # more_col_len_right = more_col_len - more_col_len_left
    # col_begin -= more_col_len_left
    # col_end += more_col_len_right
    # len_of_side = rows_num + 1

    # bgtrimed = np.zeros([len_of_side, len_of_side, noskull.shape[-1]])
    # for i in range(noskull.shape[-1]):
    #     bgtrimed[..., i] = noskull[row_begin:row_end + 1,
    #                                col_begin:col_end + 1, i]

    # old_shape = list(bgtrimed.shape)
    # factor = [n / float(o) for n, o in zip(new_shape, old_shape)]
    # resized = zoom(bgtrimed, zoom=factor, order=1, prefilter=False)

    # plot_middle_two(bgtrimed, resized)

    # resized = resized.astype(np.int16)
    # resized_nii = nib.Nifti1Image(resized, np.eye(4))
    # resized_nii_path = os.path.join(new_case_dir, subcase + "_resize.nii.gz")
    # nib.save(resized_nii, resized_nii_path)

    # new_data = load_data(resized_nii_path)
    # plot_middle_one(new_data)
    # plt.show()
