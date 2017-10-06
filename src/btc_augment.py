# Brain Tumor Classification
# Script for Data Augmentation
# Author: Qixun Qu
# Create on: 2017/10/06
# Modify on: 2017/10/06

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
        '''

        if not os.path.isdir(input_dir):
            print("Input directory is not exist.")
            raise

        if not os.path.isfile(label_file):
            print("The label file is not exist.")
            raise

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.case_no = os.listdir(input_dir)
        self.labels = pd.read_csv(label_file)

        self._check_case_no()
        self._augment_data_multi(input_dir, output_dir)

        return

    def _check_case_no(self):
        '''_CHECK_CASE_NO
        '''

        not_found_cases = []
        all_cases_no = self.labels[CASE_NO].values.tolist()
        for cv in self.case_no:
            if cv not in all_cases_no:
                not_found_cases.append(cv)

        if len(not_found_cases) != 0:
            print("Cannot find case in label file.")
            raise

        return

    def _augment_data_multi(self, input_dir, output_dir):
        '''_AUGMENT_DATA_MULTI
        '''

        case_num = len(self.case_no)
        case_paths = [os.path.join(input_dir, cn) for cn in self.case_no]

        print("Data augmentation for tumor patches\n")
        paras = zip([self] * case_num,
                    self.case_no,
                    case_paths,
                    [output_dir] * case_num)
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_augment_data, paras)

        return

    def _augment_data(self, case_no, case_path, output_dir):
        '''_AUGMENT_DATA
        '''

        def compute_partial_index(volume):
            full_size = volume.shape[0]
            partial_size = PARTIAL_SIZE
            partial_index = partial_size - 1
            diff_size = full_size - partial_size
            half_diff_size = int(diff_size / 2) - 1

            partial_begins = []
            partial_ends = []

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

            for pb in partial_begins:
                partial_ends.append(list(np.array([partial_index] * 3) + pb))

            return partial_begins, partial_ends

        def partial_array(arr, begin, end):
            return arr[begin[0]:end[0],
                       begin[1]:end[1],
                       begin[2]:end[2]]

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

        def modify_intensity(volume):
            temp = np.copy(volume)
            for c in range(CHANNELS):
                ctemp = np.reshape(temp[..., c], ((1, -1)))[0]
                non_bg_index = np.where(ctemp > 0)
                sign = np.random.randint(2, size=1)[0] * 2 - 1
                for i in non_bg_index:
                    scope = np.random.randint(5, 11, size=1)[0] / 100
                    ctemp[i] = ctemp[i] * (1 + sign * scope)
                temp[..., c] = np.reshape(ctemp, temp[..., c].shape)
            return temp

        print("Data augmentation on: ", case_no)

        case_output_dir = os.path.join(output_dir, case_no)
        if not os.path.isdir(case_output_dir):
            os.makedirs(case_output_dir)

        case_grade = self.labels[GRADE_LABEL][self.labels[CASE_NO] == case_no].values[0]
        if case_grade == GRADE_UNKNOWN:
            print("The grade of case " + case_no + "is unknown")
            return

        case_names = os.listdir(case_path)
        for cn in case_names:
            volume = np.load(os.path.join(case_path, cn))
            partial_begins, partial_ends = compute_partial_index(volume)

            volume_mirrors = [volume]
            if case_grade == GRADE_IV:
                mirror_type = np.random.randint(0, 3, 1)
                gbm_tumor = True
            else:
                mirror_type = -1
                gbm_tumor = False

            if (mirror_type == 0) or (not gbm_tumor):
                volume_mirrors.append(horizontal_mirror(volume))

            if (mirror_type == 1) or (not gbm_tumor):
                volume_mirrors.append(vertical_mirror(volume))

            if (mirror_type == 2) or (not gbm_tumor):
                volume_mirrors.append(axisymmetric_mirror(volume))

            volume_modified = [modify_intensity(vm) for vm in volume_mirrors]

            partial_no = 0
            morp_type = cn.split(".")[0]
            for vm in volume_modified:
                partial_num = len(partial_begins)
                ridx = np.random.randint(0, partial_num, 5)
                rbegins = [partial_begins[i] for i in ridx]
                rends = [partial_ends[i] for i in ridx]
                for rb, re in zip(rbegins, rends):
                    partial = partial_array(vm, rb, re)
                    partial_name = case_no + "_" + morp_type + "_" + str(partial_no) + TARGET_EXTENSION
                    partial_path = os.path.join(case_output_dir, partial_name)
                    np.save(partial_path, partial)
                    partial_no += 1

        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())

    input_dir = os.path.join(parent_dir, DATA_FOLDER, PATCHES_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, AUGMENT_FOLDER)
    label_file = os.path.join(parent_dir, DATA_FOLDER, LABEL_FILE)

    BTCAugment(input_dir, output_dir, label_file)
