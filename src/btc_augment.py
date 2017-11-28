# Brain Tumor Classification
# Script for Data Augmentation
# Author: Qixun Qu
# Create on: 2017/10/06
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

Class BTCAugment

-1- Increase the number of patches through data augmentation.
-2- Details of data augmentation methods applied in this calss.
    (1) Create mirrors of original tumor;
    (2) Slightly modify intensity of mirrors;
    (3) Randomly extract partial patches;
    (4) Save patches into output folder.

Pipline of Data Augmentation:

 Check whether all Cases can be Found in Labels File
                         |
            ---------------------------
            |        |        |       |
         Augment  Augment  Augment  Augment  <=== Multi-process of
            |        |        |       |           Data Augmentation
            ---------------------------
                         |
                       Done

Details of Augmentation Process:

                                   Original Patch
                                         |
          ----------------------------------------------------------------
          |                    |                    |                    |
       Original            Horizontal            Vertical           Axisymmetric
        Patch                Mirror               Mirror               Mirror
          |                    |                    |                    |
          |                  Modify               Modify               Modify
          |                Intensity            Intensity            Intensity
          |                    |                    |                    |
    -------------        -------------        -------------        -------------
    |           |        |           |        |           |        |           |
 Partial ... Partial  Partial ... Partial  Partial ... Partial  Partial ... Partial
    |           |        |           |        |           |        |           |
  Save        Save     Save         Save     Save        Save     Save        Save

'''


from __future__ import print_function

import os
import numpy as np
import pandas as pd
from btc_settings import *
from multiprocessing import Pool, cpu_count


# Helper function to do multiprocessing of
# BTCAugment._augment_data
def unwrap_augment_data(arg, **kwarg):
    return BTCAugment._augment_data(*arg, **kwarg)


class BTCAugment():

    def __init__(self, input_dir, output_dir, label_file):
        '''__INIT__

            Initialize the instance of calss BTCAugment, and
            finish data augmentation. The function consists
            of two steps:
            - Check whether all cases can be found in labels file.
            - Carry out data augmentation and save outputs.

            Inputs:
            -------
            - input_dir: the path of directory which stores
                         tumor and mask patches
            - output_dir: the path of directory that will keep
                          all outputs
            - label_file: the path of file which has labels
                          of all cases

        '''

        # Check whether the input folder is exist
        if not os.path.isdir(input_dir):
            raise IOError("Input directory is not exist.")

        # Check whether the label file is exist
        if not os.path.isfile(label_file):
            raise IOError("The label file is not exist.")

        # Create folder to save outputs
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # Obtain serial numbers of cases
        self.case_no = os.listdir(input_dir)

        # Read labels of all cases from label file
        self.labels = pd.read_csv(label_file)

        # Data augmentation pipline
        self._check_case_no()
        self._augment_data_multi(input_dir, output_dir)

        return

    def _check_case_no(self):
        '''_CHECK_CASE_NO

            If cases cannot be found in label file, the
            process will be stopped.

        '''

        # Put unfound cases into list
        not_found_cases = []
        all_cases_no = self.labels[CASE_NO].values.tolist()
        for cv in self.case_no:
            if cv not in all_cases_no:
                not_found_cases.append(cv)

        # If the list is not empty, quit program
        if len(not_found_cases) != 0:
            raise IOError("Cannot find case in label file.")

        return

    def _augment_data_multi(self, input_dir, output_dir):
        '''_AUGMENT_DATA_MULTI

            Main function of data augmentation to map tasks
            on different cpus to accelerate processing speed.
            The number of subprocesses equals to the number of cpus.

            - Generate paths of all cases.
            - Map parameters (serial number of case, the folder path of
              a case's patches and the output folder) to function
              BTCAugment._augment_data.

            Inputs:
            -------
            - input_dir: the path of directory which stores
                         tumor and mask patches
            - output_dir: the path of directory that will keep
                          all outputs

        '''

        # Generate path for all cases folder which patches are kept in
        case_num = len(self.case_no)
        case_paths = [os.path.join(input_dir, cn) for cn in self.case_no]

        print("\nData augmentation for tumor patches\n")
        paras = zip([self] * case_num,
                    self.case_no,
                    case_paths,
                    [output_dir] * case_num)
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_augment_data, paras)

        return

    def _augment_data(self, case_no, case_path, output_dir):
        '''_AUGMENT_DATA
            The function is used to increase the number of
            patches by those augmentation methods:
            - Create mirrors of original tumor.
            - Slightly modify intensity of mirrors.
            - Randomly extract partial patches.
            - Save patches into output folder.

            Settings can be found in btc_settings.py.

            Inputs:
            -------
            - case_no: the serial number of a case
            - case_path: the path of folder that has patches
                         generated by BTCPatches
            - output_dir: the path of folder that keeps
                          outputs patches

        '''

        # Compute range of indices of 15 partial volumes
        def compute_partial_index(volume):

            # Compute parameters
            full_size = volume.shape[0]
            partial_size = PARTIAL_SIZE
            diff_size = full_size - partial_size
            half_diff_size = int(diff_size / 2) - 1

            partial_begins = []
            partial_ends = []

            # Compute first index of all dimentions
            # of each partial volume
            # Group 1
            partial_begins.append([0, 0, 0])
            partial_begins.append([diff_size, 0, 0])
            partial_begins.append([0, diff_size, 0])
            partial_begins.append([diff_size, diff_size, 0])
            partial_begins.append([half_diff_size, half_diff_size, 0])

            # Group 2
            partial_begins.append([0, 0, diff_size])
            partial_begins.append([diff_size, 0, diff_size])
            partial_begins.append([0, diff_size, diff_size])
            partial_begins.append([diff_size, diff_size, diff_size])
            partial_begins.append([half_diff_size, half_diff_size, diff_size])

            # Group 3
            partial_begins.append([half_diff_size, 0, half_diff_size])
            partial_begins.append([0, half_diff_size, half_diff_size])
            partial_begins.append([half_diff_size, diff_size, half_diff_size])
            partial_begins.append([diff_size, half_diff_size, half_diff_size])
            partial_begins.append([half_diff_size, half_diff_size, half_diff_size])

            # Compute last index of all dimentions
            # of each partial volume
            for pb in partial_begins:
                partial_ends.append(list(np.array([partial_size] * 3) + pb))

            return partial_begins, partial_ends

        # The function that is used to extract sub-array from given array
        def partial_array(arr, begin, end):
            return arr[begin[0]:end[0],
                       begin[1]:end[1],
                       begin[2]:end[2]]

        # Return the horizontal mirror of given volume
        def horizontal_mirror(volume):

            temp = np.copy(volume)
            for i in range(temp.shape[2]):
                temp[:, :, i] = np.fliplr(temp[:, :, i])

            return temp

        # Return the vertical mirror of given volume
        def vertical_mirror(volume):

            temp = np.copy(volume)
            for i in range(temp.shape[2]):
                temp[:, :, i] = np.flipud(temp[:, :, i])

            return temp

        # Return axisymmtric mirror of given volume
        def axisymmetric_mirror(volume):

            temp = np.copy(volume)
            temp = horizontal_mirror(temp)
            temp = vertical_mirror(temp)

            return temp

        # Slightly modify intensity of given volume.
        # The scope of modification can be set in btc_settings.py.
        def modify_intensity(volume):

            temp = np.copy(volume)
            # Modify intensity in each channel respectively
            for c in range(CHANNELS):
                ctemp = np.reshape(temp[..., c], ((1, -1)))[0]
                non_bg_index = np.where(ctemp > 0)
                # Randomly generate the sign,
                # if positive, increase intensity of each voxel;
                # if negative, decrease intensity of each voxel
                sign = np.random.randint(2, size=1)[0] * 2 - 1

                for i in non_bg_index:
                    # Randomly generate how much, in percentage, the intrensity
                    # of a voxel will be modified
                    scope = np.random.randint(SCOPE_MIN, SCOPE_MAX + 1, size=1)[0] / 100
                    ctemp[i] = ctemp[i] * (1 + sign * scope)

                temp[..., c] = np.reshape(ctemp, temp[..., c].shape)

            return temp

        print("Data augmentation on: ", case_no)

        # Create output folder for each case
        case_output_dir = os.path.join(output_dir, case_no)
        if not os.path.isdir(case_output_dir):
            os.makedirs(case_output_dir)

        # Get the grade of the case
        case_grade = self.labels[GRADE_LABEL][self.labels[CASE_NO] == case_no].values[0]
        # If the grade is unknown, no more process on this case
        if case_grade == GRADE_UNKNOWN:
            print("The grade of case " + case_no + " is unknown")
            return
        # Set the number of partial patches to be generated
        elif case_grade == GRADE_II:
            partial_num = GRADE_II_PARTIALS
        elif case_grade == GRADE_III:
            partial_num = GRADE_III_PARTIALS
        elif case_grade == GRADE_IV:
            partial_num = GRADE_IV_PARTIALS
        # If the grade is invalid, quit the program
        else:
            print("The grade of case " + case_no + " is invalid")
            raise

        # Obtain patches' names of a case, at most three patches,
        # which are original, dilated and eroded tumor patches
        case_names = os.listdir(case_path)
        for cn in case_names:
            # Load patch volume
            volume = np.load(os.path.join(case_path, cn))

            # Compute range of indices of 15 partial patches
            partial_begins, partial_ends = compute_partial_index(volume)

            # If the grade of a case is IV (GBM), only one mirror can be made
            # randomly from three types of mirror. Otherwise, all mirrors will
            # be created
            if case_grade == GRADE_IV:
                # Randomly select a mirror
                # 0 for horizontal mirror
                # 1 for vertical mirror
                # 2 for axisymmetric mirror
                mirror_type = np.random.randint(0, 3, 1)
                gbm_tumor = True
            else:
                mirror_type = -1
                gbm_tumor = False

            # Generate mirrors and put them into list
            volume_mirrors = []
            if (mirror_type == 0) or (not gbm_tumor):
                volume_mirrors.append(horizontal_mirror(volume))

            if (mirror_type == 1) or (not gbm_tumor):
                volume_mirrors.append(vertical_mirror(volume))

            if (mirror_type == 2) or (not gbm_tumor):
                volume_mirrors.append(axisymmetric_mirror(volume))

            # Modity all volumes' intensity, but the original one,
            # modified mirrors are put into list
            volume_augmented = [volume]
            for vm in volume_mirrors:
                volume_augmented.append(modify_intensity(vm))

            # Extract partial patches and save them into output folder
            # Code for patches
            partial_no = 0
            # Morphology type: "original", "dilated", "eroded"
            morp_type = cn.split(".")[0]
            for vm in volume_augmented:
                # Randomly select several partial patches from 15 patches
                all_partial_num = len(partial_begins)
                ridx = np.random.randint(0, all_partial_num, partial_num)

                # Get these patches indices
                rbegins = [partial_begins[i] for i in ridx]
                rends = [partial_ends[i] for i in ridx]

                # Save into folder
                for rb, re in zip(rbegins, rends):
                    # Extract a partial patch
                    partial = partial_array(vm, rb, re)
                    # Format file name and save
                    partial_name = "_".join([case_no, morp_type, str(partial_no)]) + TARGET_EXTENSION
                    partial_path = os.path.join(case_output_dir, partial_name)
                    np.save(partial_path, partial)

                    # Increase the code for next patch
                    partial_no += 1

        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())

    input_dir = os.path.join(parent_dir, DATA_FOLDER, PATCHES_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, AUGMENT_FOLDER)
    label_file = os.path.join(parent_dir, DATA_FOLDER, LABEL_FILE)

    BTCAugment(input_dir, output_dir, label_file)
