# Brain Tumor Classification
# Script for Extracting Tumor Patches
# Author: Qixun Qu
# Create on: 2017/09/18
# Modify on: 2017/09/27


import os
import warnings
import numpy as np
from btc_settings import *
from skimage import measure
from multiprocessing import Pool, cpu_count
from scipy.ndimage.interpolation import zoom


# Ignore the warning caused by SciPy
warnings.simplefilter("ignore", UserWarning)


# Helper function to do multiprocessing of
# BTCTumorPatches._extract_tumors
def unwrap_extract_tumors(arg, **kwarg):
    return BTCTumorPatches._extract_tumors(*arg, **kwarg)


# Helper function to do multiprocessing of
# BTCTumorPatches._resize_tumors
def unwrap_resize_tumor(arg, **kwarg):
    return BTCTumorPatches._resize_tumor(*arg, **kwarg)


class BTCTumorPatches():

    def __init__(self, input_dir, output_dir, temp_dir="temp"):
        '''__INIT__

            Initialization of class BTCTumorPatches, and finish
            extracting tumor patches at the mean time.

            The structure as follows:
            - Check whether each brain volume has relevant mask.
            - Create folders to keep output files.
            - Extract primary tumor region from volume according
              to its mask.
            - Resize all tumor region to a size-fixed volume.

            Inputs:

            - input_dir: path of the directory which keeps
                         preprocessed data
            - output_dir: path for the directory that all outputs
                          will be saved in
            - temp_dir: path of the directory which
                        keeps template files during the
                        preprocessing, default is "temp"

        '''

        # Set template directory
        self.temp_mask = os.path.join(temp_dir, MASK_FOLDER)
        self.temp_full = os.path.join(temp_dir, FULL_FOLDER)
        self.temp_resize = os.path.join(temp_dir, RESIZE_FOLDER)

        # Set sub-directory of output folder
        self.output_mask = os.path.join(output_dir, MASK_FOLDER)
        self.output_full = os.path.join(output_dir, FULL_FOLDER)

        # Set sub-directory of input folder
        self.input_mask = os.path.join(input_dir, MASK_FOLDER)
        self.input_full = os.path.join(input_dir, FULL_FOLDER)

        # Obtain file's name of each preprocessed data
        self.mask_files = os.listdir(self.input_mask)
        self.full_files = os.listdir(self.input_full)

        self.shape_file = os.path.join(TEMP_FOLDER, SHAPE_FILE)

        # Main process pipline
        self._check_volumes_amount()
        self._create_folders()
        # self._extract_tumors_multi()
        self._resize_tumor_multi()

        return

    def _check_volumes_amount(self):
        '''_CHECK_VOLUMES_AMOUNT
        '''

        fulls_match_to_masks = all([f == m for f, m in zip(self.full_files,
                                                           self.mask_files)])
        if not fulls_match_to_masks:
            print("Full volumes do not match to masks.")
            raise

        return

    def _create_folders(self):
        '''_CREATE_FOLDERS

            Create folders for template files and outputs.
            All folders are as below.

            Folder for template files:
            ----- temp_dir (default is "temp")
              |----- mask
              |----- full
              |----- resize

            Folders for outputs:
            ----- self.output_dir
              |----- full
              |----- mask

            Input:

            - temp_dir: path of the directory which
                        keeps template files during the
                        preprocessing, default is "temp"

            The other two arguments, self.temp_mask, self.temp_full,
            self.temp_resize, self.output_mask and self.output_full,
            has already assigned while the instance is initialized.

        '''

        if not os.path.isdir(self.temp_mask):
            os.makedirs(self.temp_mask)

        if not os.path.isdir(self.temp_full):
            os.makedirs(self.temp_full)

        if not os.path.isdir(self.temp_resize):
            os.makedirs(self.temp_resize)

        if not os.path.isdir(self.output_mask):
            os.makedirs(self.output_mask)

        if not os.path.isdir(self.output_full):
            os.makedirs(self.output_full)

        return

    def _extract_tumors_multi(self):
        '''_EXTRACT_TUMORS_MULTI
        '''

        mask_paths = [os.path.join(self.input_mask, mf) for mf in self.mask_files]
        full_paths = [os.path.join(self.input_full, ff) for ff in self.full_files]
        volume_no = [ff.split(".")[0] for ff in self.full_files]

        if os.path.isfile(self.shape_file):
            os.remove(self.shape_file)

        open(self.shape_file, "a").close()

        print("Extract tumor patches from full volume\n")
        paras = zip([self] * len(volume_no),
                    mask_paths, full_paths, volume_no)
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_extract_tumors, paras)

        return

    def _extract_tumors(self, mask_path, full_path, volume_no):
        '''_EXTRACT_TUMORS
        '''

        def remove_small_object(mask):
            temp = np.logical_and(mask != ED_MASK, mask != ELSE_MASK) * 1.0
            blobs = measure.label(temp, background=ELSE_MASK)
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

        print("Extract tumor from patient: " + volume_no)

        mask = np.load(mask_path)
        full = np.load(full_path)

        tumor_index = remove_small_object(mask)
        dims_begin, dims_end = compute_dims_range(tumor_index)

        tumor_mask = sub_array(mask, dims_begin, dims_end)
        tumor_full = sub_array(full, dims_begin, dims_end)

        file_name = volume_no + TARGET_EXTENSION
        tumor_mask_path = os.path.join(self.temp_mask, file_name)
        tumor_full_path = os.path.join(self.temp_full, file_name)

        np.save(tumor_mask_path, tumor_mask)
        np.save(tumor_full_path, tumor_full)

        with open(self.shape_file, "a") as txt:
            txt.write(str(tumor_mask.shape[0]) + SHAPE_FILE_SPLIT)

        return

    def _resize_tumor_multi(self):
        '''_RESIZE_TUMOR_MULTI
        '''

        all_shapes = open(self.shape_file, "r")
        shapes_txt = all_shapes.read()
        shapes_list = shapes_txt.split(SHAPE_FILE_SPLIT)
        shapes_list = list(filter(None, shapes_list))
        shapes_list = list(map(int, shapes_list))

        median_shape = int(np.median(shapes_list))
        new_shape = [median_shape] * 3 + [CHANNELS]

        temp_mask_paths = [os.path.join(self.temp_mask, mf) for mf in self.mask_files]
        temp_full_paths = [os.path.join(self.temp_full, ff) for ff in self.full_files]
        volume_no = [ff.split(".")[0] for ff in self.full_files]

        print("Resize tumor patches to ", new_shape, "\n")
        paras = zip([self] * len(volume_no),
                    temp_full_paths,
                    temp_mask_paths,
                    volume_no,
                    [new_shape] * len(volume_no))
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_resize_tumor, paras)

        return

    def _resize_tumor(self, full_path, mask_path, volume_no, shape):
        '''_RESIZE_TUMOR
        '''

        print("Resize tumor on: " + volume_no)
        full = np.load(full_path)
        mask = np.load(mask_path)
        bg = np.array([np.min(full[..., i]) for i in range(CHANNELS)])
        temp_full = np.multiply(np.ones(full.shape), bg)
        non_bg_index = np.where(mask > 0)
        temp_full[non_bg_index] = full[non_bg_index]

        full_shape = list(full.shape)
        factor = [ns / float(vs) for ns, vs in zip(shape, full_shape)]
        resize_full = zoom(temp_full, zoom=factor, order=3, prefilter=False)
        resize_full = resize_full.astype(full.dtype)

        file_name = volume_no + TARGET_EXTENSION
        temp_path = os.path.join(self.temp_resize, file_name)
        np.save(temp_path, resize_full)

        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())

    input_dir = os.path.join(parent_dir, DATA_FOLDER, PREPROCESSED_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, TUMOR_FOLDER)
    temp_dir = TEMP_FOLDER

    BTCTumorPatches(input_dir, output_dir, temp_dir)
