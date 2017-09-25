# Brain Tumor Classification
# Script for Extracting Tumor Patches
# Author: Qixun Qu
# Create on: 2017/09/18
# Modify on: 2017/09/25


import os
import numpy as np
from btc_settings import *
from skimage import measure
from multiprocessing import Pool, cpu_count


# Helper function to do multiprocessing of
# BTCTumorPatches._extract_tumors
def unwrap_extract_tumors(arg, **kwarg):
    return BTCTumorPatches._extract_tumors(*arg, **kwarg)


class BTCTumorPatches():

    def __init__(self, input_dir, output_dir, temp_dir):
        '''__INIT__

            Initialization of class BTCTumorPatches, and finish
            extracting tumor patches at the mean time.

            The structure as follows:
            - Check whether each brain volume has relevant mask.
            - Create folders to keep output files.
            - Extract primary tumor region from volume according
              to its mask.
            (- Mapping all tumor region to a size-fixed volume.)

            Inputs:

            - input_dir: path of the directory which keeps
                         preprocessed data
            - output_dir: path for the directory that all outputs
                          will be saved in

        '''

        # Set template directory
        self.temp_mask = os.path.join(temp_dir, MASK_FOLDER)
        self.temp_full = os.path.join(temp_dir, FULL_FOLDER)

        # Set sub-directory of output folder
        self.output_mask = os.path.join(output_dir, MASK_FOLDER)
        self.output_full = os.path.join(output_dir, FULL_FOLDER)

        # Set sub-directory of input folder
        self.input_mask = os.path.join(input_dir, MASK_FOLDER)
        self.input_full = os.path.join(input_dir, FULL_FOLDER)

        # Obtain file's name of each preprocessed data
        self.mask_files = os.listdir(self.input_mask)
        self.full_files = os.listdir(self.input_full)

        # Main process pipline
        self._check_volumes_amount()
        self._create_folders()
        self._extract_tumors_multi()

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
        '''

        if not os.path.isdir(self.temp_mask):
            os.makedirs(self.temp_mask)

        if not os.path.isdir(self.temp_full):
            os.makedirs(self.temp_full)

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

        open("Temp/shape.txt", 'a').close()

        print("Extract tumor patches from full volume\n")
        paras = zip([self] * len(volume_no),
                    mask_paths,
                    full_paths,
                    volume_no)
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_extract_tumors, paras)

        return

    def _extract_tumors(self, mask_path, full_path, volume_no):
        '''_EXTRACT_TUMORS
        '''

        def remove_small_object(mask):
            temp = (mask > 0) * 1.0
            blobs = measure.label(temp, background=0)
            labels = np.unique(blobs)[1:]
            labels_num = [len(np.where(blobs == l)[0]) for l in labels]
            label_idx = np.where(np.array(labels_num) > 10000)[0]
            temp = np.zeros(mask.shape)
            for li in label_idx:
                temp[blobs == labels[li]] = 1.0

            return np.where(temp > 0)

        # Function to extract sub-array from given array
        # according to ranges of indices of three axes
        def sub_array(arr, index_begin, index_end):
            return arr[index_begin[0]:index_end[0],
                       index_begin[1]:index_end[1],
                       index_begin[2]:index_end[2]]

        print("Extract tumor from patient: " + volume_no)

        mask = np.load(mask_path)
        full = np.load(full_path)

        tumor_index = remove_small_object(mask)
        dims_begin = [np.min(ti) for ti in tumor_index]
        dims_end = [np.max(ti) + 1 for ti in tumor_index]

        tumor_mask = sub_array(mask, dims_begin, dims_end)
        tumor_full = sub_array(full, dims_begin, dims_end)

        file_name = volume_no + TARGET_EXTENSION
        tumor_mask_path = os.path.join(self.temp_mask, file_name)
        tumor_full_path = os.path.join(self.temp_full, file_name)

        np.save(tumor_mask_path, tumor_mask)
        np.save(tumor_full_path, tumor_full)

        with open("Temp/shape.txt", "a") as txt:
            txt.write(",".join([str(n) for n in tumor_mask.shape]) + "\n")

        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())

    input_dir = os.path.join(parent_dir, DATA_FOLDER, PREPROCESSED_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, TUMOR_FOLDER)
    temp_dir = TEMP_FOLDER

    BTCTumorPatches(input_dir, output_dir, temp_dir)
