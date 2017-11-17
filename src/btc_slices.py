# Brain Tumor Classification
# Script for Extracting Slices
# Author: Qixun Qu
# Create on: 2017/11/09
# Modify on: 2017/11/17

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

Class BTCSlices

Extract slices which have tumor core, resize all
slices into the same shape, and save slices into folder.

'''


from __future__ import print_function

import os
import shutil
import warnings
import numpy as np
from btc_settings import *
from multiprocessing import Pool, cpu_count
from scipy.ndimage.interpolation import zoom


# Ignore the warning caused by SciPy
warnings.simplefilter("ignore", UserWarning)


# Helper function to do multiprocessing of
# BTCSlices._resize_slice
def unwrap_resize_slice(arg, **kwarg):
    return BTCSlices._resize_slice(*arg, **kwarg)


class BTCSlices():

    def __init__(self, input_dir, output_dir):
        '''__INIT__

            Initialization of class to generate input path
            and create folders for outputs.

            Inputs:
            -------
            - input_dir: string, the path of input data
            - output_dir: string, the path of directory to
                          store outputs

        '''

        self.full_dir = os.path.join(input_dir, FULL_FOLDER)
        self.mask_dir = os.path.join(input_dir, MASK_FOLDER)

        # If the input folder is not exist,
        # raise errors and quit program
        self._check_folder()

        # Create output folder if it is not exist
        self.slice_dir = output_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # Extract and resize slices
        self._resize_slice_multi()

        return

    def _check_folder(self):
        '''_CHECK_FOLDER

            Check whether input folders are exists.

        '''

        if not os.path.isdir(self.full_dir):
            raise IOError("Brain volumes folder cannot be found.")

        if not os.path.isdir(self.mask_dir):
            raise IOError("Brain masks folder cannot be found.")

        return

    def _resize_slice_multi(self):
        '''_RESIZE_SLICES_MULTI

            The function to map tasks into different CPUs to
            extract and resize slices.

        '''

        # Files' names of input volume
        file_names = os.listdir(self.full_dir)

        print("Resize and save brain slices\n")
        paras = zip([self] * len(file_names), file_names)
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_resize_slice, paras)

        return

    def _resize_slice(self, file_name):
        '''_RESIZE_SLICE

            Extract slice from brai volumes at where
            the tumor core locates, and resize all slices
            into same shape.

            Input:
            ------
            - file_name: string, the name of input volume

        '''

        # The function to create folder to save slices
        # which extracted from given volume
        def create_folder(file_name):
            file_no = file_name.split(".")[0]
            save_dir = os.path.join(self.slice_dir, file_no)
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            return save_dir

        # The function to remove edge space of a volume,
        # which is background with value zero
        def remove_edgespace(v):
            vshape = list(v.shape)
            return v[EDGE_SPACE:vshape[0] - EDGE_SPACE,
                     EDGE_SPACE:vshape[1] - EDGE_SPACE,
                     EDGE_SPACE:vshape[2] - EDGE_SPACE]

        # Extract sub-volume that has tumor's core
        def extract_core_volume(volume, mask):
            # Obtain the number of slices
            num_slices = mask.shape[-1]
            # Obtain the area of each slice
            slice_area = mask.shape[0] * mask.shape[1]
            # Inialize a empty list to save slice number
            # that needs to be kept
            core_nums = []

            # In each slice of the volume, if there is tumor's core
            # this slice will be extracted
            for i in range(num_slices):
                core_pos = np.where(mask[:, :, i] >= MASK_THRESHOLD)
                core_nums.append(len(core_pos[0]))

            # Compute a threshold to remove some slices in which
            # the area of tumor's core is too small
            min_core_num = int(max(core_nums) * PROP_THRESHOLD)
            core_slice_idxs = []
            for i in range(num_slices):
                if core_nums[i] >= min_core_num:
                    large_object = True
                    for c in range(CHANNELS):
                        # Check brain's area in each channel
                        non_bg_area = len(np.where(volume[:, :, i, c] > 0)[0])
                        # If brain's area is too small, the slice will be
                        # be taken into consideration
                        if non_bg_area / slice_area < PROP_NON_BG:
                            large_object = False
                    if large_object:
                        core_slice_idxs.append(i)

            if len(core_slice_idxs) > 0:
                # Extract sub-volumes
                min_core_slice_idx = min(core_slice_idxs)
                max_core_slice_idx = max(core_slice_idxs) + 1
                return volume[:, :, min_core_slice_idx:max_core_slice_idx, :]
            else:
                return None

        # The function to pad zero around input volume
        # to make the output volume be a square cube
        def pad_zero(volume):
            vshape = list(volume.shape)
            pad_size = vshape[0] - vshape[1]
            left_pad_size = int(pad_size / 2.0)
            right_pad_size = pad_size - left_pad_size

            vshape[1] = left_pad_size
            left_pad = np.zeros(vshape)
            vshape[1] = right_pad_size
            right_pad = np.zeros(vshape)
            return np.hstack((left_pad, volume, right_pad))

        print("Resize slices in " + file_name)

        # Load volume and its mask
        file_path = os.path.join(self.full_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
        volume = remove_edgespace(np.load(file_path))
        mask = remove_edgespace(np.load(mask_path))

        # Obtain sub-volume that contains tumor's core
        volume = extract_core_volume(volume, mask)

        if volume is None:
            return

        # Pad volume to square cube
        pad_volume = pad_zero(volume)
        vshape = list(pad_volume.shape)

        # Compute zoom factor
        sshape = vshape.copy()
        sshape.pop(2)
        factor = [ns / ss for ns, ss in zip(SLICE_SHAPE, sshape)]

        # Create folder for output
        save_dir = create_folder(file_name)

        # Resize slices in sub-volumes and save it into folder
        for i in range(vshape[2]):
            vslice = pad_volume[:, :, i, :]
            resized_slice = zoom(vslice, zoom=factor, order=1, prefilter=False)
            resized_slice = resized_slice.astype(vslice.dtype)
            save_file_name = str(i) + TARGET_EXTENSION
            np.save(os.path.join(save_dir, save_file_name), resized_slice)

        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    input_dir = os.path.join(parent_dir, DATA_FOLDER, PREPROCESSED_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, SLICES_FOLDER)

    BTCSlices(input_dir, output_dir)
