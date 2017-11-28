# Brain Tumor Classification
# Script for None Preprocessing
# Author: Qixun Qu
# Create on: 2017/11/28
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

Class BTCNonePreprocess

'''


import os
import numpy as np
import nibabel as nib
from btc_settings import *
from multiprocessing import Pool, cpu_count


# Helper function to do multiprocessing of
# BTCNonePreprocess._merge_to_one_volume
def unwrap_merge_to_one_volume(arg, **kwarg):
    return BTCNonePreprocess._merge_to_one_volume(*arg, **kwarg)


class BTCNonePreprocess():

    def __init__(self, input_dir, output_dir):

        if not os.path.isdir(input_dir):
            raise IOError("Cannot find input directory.")

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.volume_no = os.listdir(input_dir)
        self.full_folder = os.path.join(output_dir, FULL_FOLDER)
        self.mask_folder = os.path.join(output_dir, MASK_FOLDER)

        self._create_folders()
        self._merge_to_one_volume_multi(input_dir, output_dir)

        return

    def _merge_to_one_volume_multi(self, input_dir, output_dir):
        '''_MERGE_TO_ONE_VOLUME_MULTI

            Main function of merging four types volumes and saving outputs
            to map tasks on different cpus to accelerate processing speed.
            The number of subprocesses equals to the number of cpus.

            Inputs:
            -------
            - input_dir: path of the directory which keeps mask volumes
            - output_dir: path of directory which keeps the outputs

        '''

        print("\nMerge flair, t1, t1Gd and t2 into One Volume\n")
        volume_no_len = len(self.volume_no)
        paras = zip([self] * volume_no_len,
                    [input_dir] * volume_no_len,
                    [output_dir] * volume_no_len,
                    self.volume_no)
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_merge_to_one_volume, paras)

        return

    def _create_folders(self):
        '''_CREATE_FOLDERS

            Create folders for temporary files and outputs.
            All folders are as below.

            Folders for outputs:
            ----- output_dir
              |----- full
              |----- mask

        '''

        if not os.path.isdir(self.mask_folder):
            os.makedirs(self.mask_folder)

        if not os.path.isdir(self.full_folder):
            os.makedirs(self.full_folder)

        return

    def _merge_to_one_volume(self, input_dir, output_dir, vno):
        '''_MERGE_TO_ONE_VOLUME

            Merge normalized flair, t1, t1Gd and t2 volumes of one patient
            to one volume. Remove surrounding backgrounds, and save output
            into output folder as the result of preprocessing.

            Inputs:
            -------
            - input_dir: path of the directory which keeps mask volumes
            - output_dir: path of directory which keeps the outputs
            - vno: serial number of volumes, which is also the folder name
                   of one patient's volumes

        '''

        print("NO." + vno + ": Save brain volume and mask volume")
        full_volume = np.zeros(FULL_SHAPE)
        volume_folder = os.path.join(input_dir, vno)
        for file in os.listdir(volume_folder):
            for i in range(len(VOLUME_TYPES)):
                if VOLUME_TYPES[i] in file:
                    volume_path = os.path.join(volume_folder, file)
                    volume = nib.load(volume_path).get_data()
                    volume = np.rot90(volume, 3, axes=(0, 1))
                    # full_volume[..., 0] <== flair volume
                    # full_volume[..., 1] <== t1 volume
                    # full_volume[..., 2] <== t1Gd volume
                    # full_volume[..., 3] <== t2 volume
                    full_volume[..., i] = volume
            if MASK_NAME in file:
                # Load relevant mask
                mask_path = os.path.join(volume_folder, file)
                mask_volume = nib.load(mask_path).get_data()
                mask_volume = np.rot90(mask_volume, 3, axes=(0, 1))

        # Remove surrounding backgrounds from ensemble volume and mask volume
        full_volume, mask_volume = self._keep_minimum_volume(full_volume, mask_volume)

        # Save volume into output folders
        full_volume_path = os.path.join(self.full_folder, vno + TARGET_EXTENSION)
        mask_volume_path = os.path.join(self.mask_folder, vno + TARGET_EXTENSION)

        np.save(full_volume_path, full_volume)
        np.save(mask_volume_path, mask_volume)

        return

    def _keep_minimum_volume(self, full, mask):
        '''_KEEP_MINIMUM_VOLUME

            Remove surrounding backgrounds from ensemble volume
            and mask volume to keep the minimum volume.
            Based on input volumes, compute the range of indices
            of three axes, extract sub-volume and return it back.

            Inputs:
            -------
            - full: ensemble volume
            - mask: relevant mask volume

            Outputs:
            --------
            - new_full: new ensemble volume after being processed
            - new_mask: new mask volume after being processed

        '''

        # Function to extract sub-array from given array
        # according to ranges of indices of three axes
        def sub_array(arr, index_begin, index_end):
            return arr[index_begin[0]:index_end[0],
                       index_begin[1]:index_end[1],
                       index_begin[2]:index_end[2]]

        # Compute background value of volume
        full_sum = np.sum(full, axis=3)
        min_full_sum = np.min(full_sum)

        # Compute range of indices of each axes
        non_bg_index = np.where(full_sum > min_full_sum)
        dims_begin = [np.min(nzi) for nzi in non_bg_index]
        dims_end = [np.max(nzi) + 1 for nzi in non_bg_index]

        # Add a bit more space around the minimum brain volume
        for i in range(len(dims_begin)):
            dims_begin[i] = dims_begin[i] - EDGE_SPACE
            # if the beginning index is lower than 0
            if dims_begin[i] < 0:
                dims_begin[i] = 0
            dims_end[i] = dims_end[i] + EDGE_SPACE
            # if the ending index is larger than the maximum index
            if dims_end[i] > BRAIN_SHAPE[i] - 1:
                dims_end[i] = BRAIN_SHAPE[i] - 1

        # Obtain sub-volumes from input volumes
        new_full = sub_array(full, dims_begin, dims_end)
        new_full = new_full.astype(np.float32)
        new_mask = sub_array(mask, dims_begin, dims_end)

        return new_full, new_mask


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    input_dir = os.path.join(parent_dir, DATA_FOLDER, ORIGINAL_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, NONEPREPROCESSED_FOLDER)

    BTCNonePreprocess(input_dir, output_dir)
