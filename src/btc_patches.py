# Brain Tumor Classification
# Script for Extracting Tumor Patches
# Author: Qixun Qu
# Create on: 2017/09/18
# Modify on: 2017/11/28

#     ,,,         ,,,
#   ;"   ';     ;'   ",
#   ;  @.ss$$$$$$s.@  ;
#   `s$$$$$$$$$$$$$$$'
#   $$$$$$$$$$$$$$$$$$
#  $$$$P""Y$$$Y""W$$$$$
#  $$$$  p"$$$"q  $$$$$
#  $$$$  .$$$$$.  $$$$'
#   $$$DaU$$O$$DaU$$$'
#    '$$$$'.^.'$$$$'
#       '&$$$$$&'

'''

Class BTCPatches

-1- Extract tumor regions in preprocessed brain volumes,
    and save all tumors patches in square cubes.
-2- Resize tumors patches into same shape.

Pipline of Patches Generation:

   Check whether All Input Brain Volumes
       Match with Input Mask Volumes
                     |
 Create Folders for Temporary and Output Files
                     |
        ----------------------------
        |        |        |        |
     Extract  Extract  Extract  Extract  <=== Multi-process of
        |        |        |        |          Extracting Minimum Tumor Patches
        ----------------------------
                     |
       Save Outputs in Temporary Folder
     Compute Median Shape of All Patches
                     |
        ----------------------------
        |        |        |        |
      Resize   Resize   Resize   Resize  <=== Multi-process of
        |        |        |        |          Resizeing Patches into Same Shape
        ----------------------------
                     |
       Save Outputes in Output Folder
          Delete Temporary Folder

'''


from __future__ import print_function

import os
import shutil
import argparse
import warnings
import numpy as np
from btc_settings import *
import scipy.ndimage as sn
from skimage import measure
import scipy.ndimage.morphology as snm
from multiprocessing import Pool, cpu_count
from scipy.ndimage.interpolation import zoom


# Ignore the warning caused by SciPy
warnings.simplefilter("ignore", UserWarning)


# Helper function to do multiprocessing of
# BTCPatches._extract_tumors
def unwrap_extract_tumors(arg, **kwarg):
    return BTCPatches._extract_tumor(*arg, **kwarg)


# Helper function to do multiprocessing of
# BTCPatches._resize_tumors
def unwrap_resize_tumor(arg, **kwarg):
    return BTCPatches._resize_tumor(*arg, **kwarg)


class BTCPatches():

    def __init__(self, input_dir, output_dir, temp_dir="temp", is_morph=1):
        '''__INIT__

            Initialization of class BTCPatches, and finish
            extracting tumor patches at the mean time.

            The structure as follows:
            - Check whether each brain volume has relevant mask.
            - Create folders to keep temporary and output files.
            - Extract primary tumor region from volume according
              to its mask.
            - Resize all tumor region to a size-fixed volume.

            Inputs:
            -------
            - input_dir: path of the directory which keeps
                         preprocessed data
            - output_dir: path for the directory that all outputs
                          will be saved in
            - temp_dir: path of the directory which
                        keeps temporary files during the
                        preprocessing, default is "temp"

        '''

        # Set temporary directory for both tumor and mask patches
        self.temp_mask = os.path.join(temp_dir, MASK_FOLDER)
        self.temp_tumor = os.path.join(temp_dir, TUMOR_FOLDER)

        # Set directory of output
        self.output_tumor = output_dir

        # Set sub-directory of input volumes
        self.input_mask = os.path.join(input_dir, MASK_FOLDER)
        self.input_full = os.path.join(input_dir, FULL_FOLDER)

        # Obtain file's name of each preprocessed data
        self.mask_files = os.listdir(self.input_mask)
        self.full_files = os.listdir(self.input_full)

        # Create a file name to record each tumor patch's size
        self.shape_file = os.path.join(temp_dir, SHAPE_FILE)

        self.is_morph = bool(int(is_morph))
        print(self.is_morph)

        # Patches generation pipline
        self._check_volumes_amount()
        self._create_folders()
        self._extract_tumors_multi()
        self._resize_tumors_multi()

        # Delete temporary folder and all files in it
        self._delete_temp_files()

        return

    def _check_volumes_amount(self):
        '''_CHECK_VOLUMES_AMOUNT

            In this case, it is required that every brain volume
            has its relevant mask to carry out following steps.
            The process will be stopped if any brain volumes have
            no matched masks.

        '''

        fulls_match_to_masks = all([f == m for f, m in zip(self.full_files,
                                                           self.mask_files)])
        if not fulls_match_to_masks:
            print("Full volumes do not match to masks.")
            raise

        return

    def _create_folders(self):
        '''_CREATE_FOLDERS

            Create folders for temporary files and outputs.
            All folders are as below.

            Folder for temporary files:
            ----- temp_dir (default is "temp")
              |----- mask
              |----- tumor

            Folders for outputs:
            ----- self.output_dir

            Input:
            ------
            - temp_dir: path of the directory which
                        keeps temporary files during the
                        preprocessing, default is "temp"

            The other two arguments, self.temp_mask, self.temp_tumor,
            self.output_tumor, has been already assigned while the
            instance is initialized.

        '''

        if not os.path.isdir(self.temp_mask):
            os.makedirs(self.temp_mask)

        if not os.path.isdir(self.temp_tumor):
            os.makedirs(self.temp_tumor)

        if not os.path.isdir(self.output_tumor):
            os.makedirs(self.output_tumor)

        return

    def _extract_tumors_multi(self):
        '''_EXTRACT_TUMORS_MULTI

            Main function of extracting tumors to map tasks
            on different cpus to accelerate processing speed.
            The number of subprocesses equals to the number of cpus.

            - Generate paths of all input volumes of brain and mask.
            - Create a text file to keep shape of every patches.
            - Map parameters (mask path, brain path and case number)
              to function BTCPatches._extract_tumor.

        '''

        # Generate paths of masks, paths of brain volumes and case serial numbers
        mask_paths = [os.path.join(self.input_mask, mf) for mf in self.mask_files]
        full_paths = [os.path.join(self.input_full, ff) for ff in self.full_files]
        case_nos = [ff.split(".")[0] for ff in self.full_files]

        # If the text file is exist already, delete it
        if os.path.isfile(self.shape_file):
            os.remove(self.shape_file)

        # Create empty text file to keep all pathes' shape
        open(self.shape_file, "a").close()

        print("\nStep 1: Extract tumor patches from full volume\n")
        paras = zip([self] * len(case_nos),
                    mask_paths,
                    full_paths,
                    case_nos)
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_extract_tumors, paras)

        return

    def _extract_tumor(self, mask_path, full_path, case_no):
        '''_EXTRACT_TUMORS

            Extract the patch of tumor's core according to its mask
            and save outputs in temporary folder. There are three steps
            in this stage, which are:
            - Do morphology operations on tumor core's mask. Each of
              them will be dilated. Some masks shall be eroded if they
              are larger than the threshold volume. After this step,
              three masks (original, dilated and maybe eroded) can be
              obtained at most.
            - Compute the range of tumor core's index based on each mask,
              and make sure that each dimention has same size.
            - Extract tumor patch according to indices and save it in
              temporary folder.

            Inputs:
            -------
            - mask_path: the path of mask volume
            - full_path: the path of brain volume
            - case_no: the serial number of input volume, this is used to
                       format the name of output file

        '''

        # This function is used to get tumor indices.
        # Firstly, compute the size of individual connected region,
        # if the region has a larger size than given threshold, it
        # will be keeped; if not, it will be considered as a part of
        # tumor. Then, get and return indices of reserved region.
        def remove_small_object(mask):

            # Obtain all connected regions
            blobs = measure.label((mask > 0) * 1.0, background=ELSE_MASK)
            labels = np.unique(blobs)[1:]
            labels_num = [len(np.where(blobs == l)[0]) for l in labels]

            # Find which regions are larger than threshold
            label_idx = np.where(np.array(labels_num) > TUMOT_MIN_SIZE)[0]
            temp = np.zeros(mask.shape)

            # Remain those regions
            for li in label_idx:
                temp[blobs == labels[li]] = 1.0

            # Retuen indices of remain mask
            return np.where(temp > 0)

        # In this function, the indices' range of each dimention is
        # going to be computed. What's more, every dimention has the
        # same size, which will lead to a square cube.
        def compute_dims_range(index):

            # Get original indices' range of all dimentions
            dims_min = np.array([np.min(i) for i in index])
            dims_max = np.array([np.max(i) + 1 for i in index])

            # Find the maximum size in all dimentions
            dims_len = dims_max - dims_min + 1
            dims_len_max = np.max(dims_len)

            # Compute new range to make sure each dimention
            # has same size
            dims_begin, dims_end = [], []
            for dmin, dmax, dlen in zip(dims_min, dims_max, dims_len):
                if dlen == dims_len_max:
                    # No need to change range
                    dims_begin.append(dmin)
                    dims_end.append(dmax)
                else:  # Widen the range
                    diff = dims_len_max - dlen
                    diff_left = int(diff / 2)
                    diff_right = diff - diff_left
                    dims_begin.append(dmin - diff_left)
                    dims_end.append(dmax + diff_right)

            return dims_begin, dims_end

        # Function to extract sub-array from given array
        # according to ranges of indices of three dimentions.
        # If the indices is out of range of given array,
        # the extraction will be padded by its background.
        def sub_array(arr, begin, end):

            # Get the background intensity of input array
            arr_shape = arr.shape
            if len(arr_shape) == CHANNELS:  # Brainvolume
                bg = np.array([np.min(arr[..., i]) for i in range(CHANNELS)])
            else:  # Mask volume
                bg = np.min(arr)

            # Compute new range of indices, as well as how many slices
            # of background will be padded in each dimentions
            new_begin, begin_diff = [], []
            new_end, end_diff = [], []
            for i in range(len(begin)):
                # Before first slice
                if begin[i] >= 0:  # No need to pad
                    new_begin.append(begin[i])
                    begin_diff.append(0)
                else:  # Need to pad befor the first slice
                    new_begin.append(0)
                    begin_diff.append(np.abs(begin[i]))

                # After last slice
                if end[i] <= arr_shape[i] - 1:  # No need to pad
                    new_end.append(end[i] + 1)
                    end_diff.append(0)
                else:  # Need to pad after the last slice
                    new_end.append(arr_shape[i] - 1)
                    end_diff.append(end[i] - arr_shape[i] + 2)

            # Extract tumor region
            sub_arr = arr[new_begin[0]:new_end[0],
                          new_begin[1]:new_end[1],
                          new_begin[2]:new_end[2]]

            for i in range(len(begin_diff)):
                temp_shape = list(sub_arr.shape)

                # Pad background before the first slice
                if begin_diff[i] > 0:
                    temp_shape[i] = begin_diff[i]
                    temp_arr = np.multiply(np.ones(temp_shape), bg)
                    sub_arr = np.concatenate((temp_arr, sub_arr), axis=i)

                # Pad background after the last slice
                if end_diff[i] > 0:
                    temp_shape[i] = end_diff[i]
                    temp_arr = np.multiply(np.ones(temp_shape), bg)
                    sub_arr = np.concatenate((sub_arr, temp_arr), axis=i)

            return sub_arr.astype(arr.dtype)

        print("Extract tumor from patient: " + case_no)

        # Load brain and mask volumes
        mask = np.load(mask_path)
        full = np.load(full_path)

        # Generate kernel for dilatation and erosion
        kernel = sn.generate_binary_structure(3, 1).astype(np.float32)
        # Get the original tumor core's mask
        original_core_mask = np.logical_and(mask != ED_MASK,
                                            mask != ELSE_MASK) * 1.0

        # If the size of tumor's core is too small,
        # enable_eroded will be assigned to False
        # to disable erosion on this case
        enable_eroded = True

        # Loop order can be found in bts_settings.py
        # --- NOTE ---
        # "eroded" should always be the last item
        for morp in MORPHOLOGY:

            if (not self.is_morph) and (morp == "dilated" or morp == "eroded"):
                continue

            if morp == "dilated":  # Dilatation
                core_mask = snm.binary_dilation(original_core_mask,
                                                structure=kernel,
                                                iterations=MORP_ITER_NUM)
            elif morp == "eroded":  # Erosion
                if enable_eroded:  # The tumor can be eroded.
                    core_mask = snm.binary_erosion(original_core_mask,
                                                   structure=kernel,
                                                   iterations=MORP_ITER_NUM)
                else:  # The tumor cannot be eroded since it is too small
                    continue
            else:  # Original mask
                core_mask = original_core_mask

            # Get tumor indices
            tumor_index = remove_small_object(core_mask)
            # If no tumor availabel, start next loop
            if len(tumor_index[0]) == 0:
                continue

            # Compute the range of indices in each dimention
            dims_begin, dims_end = compute_dims_range(tumor_index)
            if morp == "original":
                if (dims_end[0] - dims_begin[0]) < ERODABLE_THRESH:
                    # If the size of tumor is smaller than the threshold
                    # this tumor will not be eroded
                    enable_eroded = False

            # Extract patch from brain and mask volumes
            tumor_mask = sub_array(mask, dims_begin, dims_end)
            tumor_full = sub_array(full, dims_begin, dims_end)

            # Save patches into temporary folder
            file_name = case_no + "_" + morp + TARGET_EXTENSION
            tumor_mask_path = os.path.join(self.temp_mask, file_name)
            tumor_full_path = os.path.join(self.temp_tumor, file_name)

            np.save(tumor_mask_path, tumor_mask)
            np.save(tumor_full_path, tumor_full)

            # Save the shape of patch extracted according to the original
            # tumor core's mask into text file
            if morp == "original":
                with open(self.shape_file, "a") as txt:
                    txt.write(str(tumor_full.shape[0]) + SHAPE_FILE_SPLIT)

        return

    def _resize_tumors_multi(self):
        '''_RESIZE_TUMOR_MULTI

            Main function of resizing tumor patches to map tasks
            on different cpus to accelerate processing speed.
            The number of subprocesses equals to the number of cpus.

            - Compute the median shape of all tumor patches and
              generate new shape that all patches will be resized to.
            - Generate paths of all input volumes of tumor and mask.
            - Map parameters (mask patch's path, tumor patch's path,
              patch's name and new shape) to function
              BTCPatches._extract_tumor.

        '''

        # Read original tumors' shape from text file
        all_shapes = open(self.shape_file, "r")
        shapes_txt = all_shapes.read()
        shapes_list = shapes_txt.split(SHAPE_FILE_SPLIT)
        shapes_list = list(filter(None, shapes_list))
        shapes_list = list(map(int, shapes_list))

        # Compute median shape of all tumors as the new shape
        median_shape = int(np.median(shapes_list))
        new_shape = [median_shape] * 3 + [CHANNELS]

        # Generate mask patches' path, tumor patches' path
        # and patches' names
        temp_file_names = os.listdir(self.temp_tumor)
        temp_mask_paths = [os.path.join(self.temp_mask, tfn) for tfn in temp_file_names]
        temp_tumor_paths = [os.path.join(self.temp_tumor, tfn) for tfn in temp_file_names]
        patch_names = [tfn.split(".")[0] for tfn in temp_file_names]

        print("\nStep 2: Resize tumor patches to ", new_shape, "\n")
        paras = zip([self] * len(patch_names),
                    temp_mask_paths,
                    temp_tumor_paths,
                    patch_names,
                    [new_shape] * len(patch_names))
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_resize_tumor, paras)

        return

    def _resize_tumor(self, mask_path, tumor_path, patch_name, shape):
        '''_RESIZE_TUMOR

            Resize tumor patch into the given shape. Three steps
            are contained in this function:
            - Remove surrounding brain tissues but keep tumor left.
              Brain tissues will be replaced by background.
            - Resize the patch into given shape.
            - Save resized patch into output folder.

            Inputs:
            -------
            - mask_path: the path of mask patch
            - tumor_path: the path of tumor patch
            - patch_name: the name of patch file, which is used
                          to format output's name
            - shape: the shape that patch will be resized into

        '''

        print("Resize tumor on: " + patch_name)

        # Load tumor and mask patch
        tumor = np.load(tumor_path)
        mask = np.load(mask_path)

        # Remove surrounding brain tissues around tumor,
        # and replace tissues with background
        bg = np.array([np.min(tumor[..., i]) for i in range(CHANNELS)])
        temp_tumor = np.multiply(np.ones(tumor.shape), bg)
        non_bg_index = np.where(mask > 0)
        temp_tumor[non_bg_index] = tumor[non_bg_index]

        # Resize tumor into given shape.
        # Settings can be found in btc_settings.py
        tumor_shape = list(tumor.shape)
        # Compute zoom factor, a warning may be appear
        # The warning has been ignored by the code at line 68
        factor = [ns / float(vs) for ns, vs in zip(shape, tumor_shape)]
        resize_tumor = zoom(temp_tumor, zoom=factor,
                            order=ZOOM_ORDER, prefilter=ZOOM_FILTER)
        resize_tumor = resize_tumor.astype(tumor.dtype)

        # Generate the path of output folder
        case_no = patch_name.split("_")[0]
        case_no_folder = os.path.join(self.output_tumor, case_no)

        # Create output folder for one case
        if not os.path.isdir(case_no_folder):
            os.makedirs(case_no_folder)

        # Save output into the folder
        morp_type = patch_name.split("_")[1]
        file_name = morp_type + TARGET_EXTENSION
        output_path = os.path.join(case_no_folder, file_name)
        np.save(output_path, resize_tumor)

        return

    def _delete_temp_files(self):
        '''_DELETE_TEMP_FILES

            Delete temporary files in temporary folder except
            the text file of all original tumors' shapes.

        '''

        if os.path.isdir(self.temp_tumor):
            shutil.rmtree(self.temp_tumor)

        if os.path.isdir(self.temp_mask):
            shutil.rmtree(self.temp_mask)

        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    help_str = "Whether use morphology '1' or '0'."
    parser.add_argument("--morph", action="store", default=1,
                        dest="morph", help=help_str)
    args = parser.parse_args()

    parent_dir = os.path.dirname(os.getcwd())

    input_dir = os.path.join(parent_dir, DATA_FOLDER, PREPROCESSED_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, PATCHES_FOLDER)
    temp_dir = os.path.join(TEMP_FOLDER, PATCHES_FOLDER)

    BTCPatches(input_dir, output_dir, temp_dir, args.morph)
