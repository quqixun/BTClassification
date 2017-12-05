# Brain Tumor Classification
# Script for Extracting Slices
# Author: Qixun Qu
# Create on: 2017/11/09
# Modify on: 2017/12/04

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
import pandas as pd
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

    def __init__(self, input_dir, output_dir, label_file):
        '''__INIT__

            Initialization of class to generate input path
            and create folders for outputs.

            Inputs:
            -------
            - input_dir: string, the path of input data
            - output_dir: string, the path of directory to
                          store outputs
            - label_file: the path of file which has labels
                          of all cases

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

        # Check whether the label file is exist
        if not os.path.isfile(label_file):
            raise IOError("The label file is not exist.")

        # Read labels of all cases from label file
        self.labels = pd.read_csv(label_file)

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
        case_names = os.listdir(self.full_dir)

        print("\nResize and save brain slices\n")
        paras = zip([self] * len(case_names), case_names)
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_resize_slice, paras)

        return

    def _resize_slice(self, case_name):
        '''_RESIZE_SLICE

            Extract slice from brai volumes at where
            the tumor core locates, and resize all slices
            into same shape.

            Input:
            ------
            - case_name: string, the name of input volume

        '''

        # The function to create folder to save slices
        # which extracted from given volume
        def create_folder(case_name):
            case_no = case_name.split(".")[0]
            save_dir = os.path.join(self.slice_dir, case_no)
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            return case_no, save_dir

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
                        # If brain's area is too small, the slice will
                        # not be taken into consideration
                        if non_bg_area / slice_area < PROP_NON_BG:
                            large_object = False
                    if large_object:
                        core_slice_idxs.append(i)

            # print(len(core_slice_idxs))

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

        # Slightly modify intensity of given volume.
        # The scope of modification can be set in btc_settings.py.
        def modify_intensity(image):
            temp = np.copy(image)
            # Modify intensity in each channel respectively
            for c in range(CHANNELS):
                ctemp = np.reshape(temp[..., c], ((1, -1)))[0]
                non_bg_index = np.where(ctemp > 0)
                # Randomly generate the sign,
                # if positive, increase intensity of each pixel;
                # if negative, decrease intensity of each pixel
                sign = np.random.randint(2, size=1)[0] * 2 - 1

                for i in non_bg_index:
                    # Randomly generate how much, in percentage, the intrensity
                    # of a voxel will be modified
                    scope = np.random.randint(10, 31, size=1)[0] / 100
                    ctemp[i] = ctemp[i] * (1 + sign * scope)

                temp[..., c] = np.reshape(ctemp, temp[..., c].shape)

            return temp

        # The function to obtain the augmentations
        # of input slice to enlarge dataset
        def augmentation(image, grade):
            slices = [image, np.fliplr(image)]
            modins_slices = []

            # Set the number of augmented slices to be generated
            if grade == GRADE_II:
                augment_num = 4
            elif grade == GRADE_III:
                augment_num = 3
            elif grade == GRADE_IV:
                return slices
            else:
                raise ValueError("Unknown grade.")

            # Modify intensity of the slice selected randomly
            for i in range(augment_num):
                index = np.random.randint(2, size=1)[0]
                modins_slices.append(modify_intensity(slices[index]))

            slices += modins_slices

            return slices

        print("Resize slices in " + case_name)

        # Load volume and its mask
        case_path = os.path.join(self.full_dir, case_name)
        mask_path = os.path.join(self.mask_dir, case_name)
        volume = remove_edgespace(np.load(case_path))
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
        case_no, save_dir = create_folder(case_name)
        # Get the grade of the case
        case_grade = self.labels[GRADE_LABEL][self.labels[CASE_NO] == case_no].values[0]
        # If the grade is unknown, no more process on this case
        if case_grade == GRADE_UNKNOWN:
            print("The grade of case " + case_no + " is unknown")
            return

        # Resize slices in sub-volumes and save it into folder
        for i in range(vshape[2]):
            vslice = pad_volume[:, :, i, :]
            resized_slice = zoom(vslice, zoom=factor, order=1, prefilter=False)
            resized_slice = resized_slice.astype(vslice.dtype)

            # Obtain the augmentations of resized slice
            # slices = augmentation(resized_slice, case_grade)'
            slices = [resized_slice]
            if case_grade != GRADE_IV:
                slices += [np.fliplr(resized_slice)]

            # Write file into folder
            for j in range(len(slices)):
                save_file_name = str(i) + "_" + str(j) + TARGET_EXTENSION
                np.save(os.path.join(save_dir, save_file_name), slices[j])

        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    input_dir = os.path.join(parent_dir, DATA_FOLDER, PREPROCESSED_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, SLICES_FOLDER)
    label_file = os.path.join(parent_dir, DATA_FOLDER, LABEL_FILE)

    BTCSlices(input_dir, output_dir, label_file)
