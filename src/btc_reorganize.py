# Brain Tumor Classification
# Script for Reorganizing
# Author: Qixun Qu
# Create on: 2017/09/12
# Modify on: 2017/09/24

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

Class BTCReorganize

Reorganize volumes from source folder
to target folder as shown below.

Source folder:
----- BraTS2017
  |----- HGG
     |----- TCGA-xx-xxxx
        |----- TCGA-xx-xxxx_date_flair.nii.gz
        |----- TCGA-xx-xxxx_date_GlistrBoost.nii.gz
        |----- TCGA-xx-xxxx_date_GlistrBoost_ManuallyCorrected.nii.gz
        |----- TCGA-xx-xxxx_date_t1.nii.gz
        |----- TCGA-xx-xxxx_date_t1Gd.nii.gz
        |----- TCGA-xx-xxxx_date_t2.nii.gz
  |----- LGG
     |----- (same structure as HGG)

102 samples in HGG, 65 volumes in LGG.

Drop TGlistrBoost.nii.gz.
Rename ManuallyCorrected.nii.gz as mask.nii.gz.

Target Folder:
----- data
  |----- HGG
     |----- TCGA-xx-xxxx
        |----- TCGA-xx-xxxx_flair.nii.gz
        |----- TCGA-xx-xxxx_t1.nii.gz
        |----- TCGA-xx-xxxx_t1Gd.nii.gz
        |----- TCGA-xx-xxxx_t2.nii.gz
        |----- TCGA-xx-xxxx_mask.nii.gz
  |----- LGG
     |----- (same structure as HGG)

'''


import os
from tqdm import *
from shutil import copy
from btc_settings import *


class BTCReorganize():

    def __init__(self, input_dir, output_dir):
        '''__INIT__

            Copy brain volume files from input folder
            to output folder to reorganize the structure
            of all files.

            Inputs:
            -------
            - input_dir: original path of directory where
                         keeps brain volume data
            - output_dir: volume data files will be copied
                          to this directory

        '''

        self._reorganize(input_dir, output_dir)

        return

    def _reorganize(self, input_dir, output_dir):
        '''_REPRGANIZE

            The process of this function:

            for target in ["HGG", "LGG"]:
                for patient in [1, ..., n]:
                    for volume in ["flair", "t1", "t1gd", "t2"]:
                        Obtain path for original volume
                        Generate path for copied volume
                        Copy volume from original path to new path

            Inputs:
            -------
            - input_dir: original path of directory where
                         keeps brain volume data
            - output_dir: volume data files will be copied
                          to this directory

        '''

        # Create output folder
        to_path = os.path.join(output_dir, ORIGINAL_FOLDER)
        if not os.path.isdir(to_path):
            os.makedirs(to_path)

        print("Copy files from: " + input_dir)
        print("to: " + to_path)

        sub_source = os.listdir(input_dir)
        for st in sub_source:
            print("Starting copy files in ", st)
            from_path = os.path.join(input_dir, st)
            from_dirs = os.listdir(from_path)

            for fd in tqdm(from_dirs):
                sub_to_path = os.path.join(to_path, fd)
                sub_from_path = os.path.join(from_path, fd)
                source_files = os.listdir(sub_from_path)

                if not os.path.isdir(sub_to_path):
                    os.makedirs(sub_to_path)

                has_manually_corrected = any(REPLACE_MASK_NAME in s for s in source_files)

                for sf in source_files:
                    source_split = sf.split("_")
                    source_no = source_split[0]
                    source_name = source_split[-1]
                    source_type = source_name.split(".")[0]

                    if has_manually_corrected:
                        # Drop GlistrBoost.nii.gz if
                        # ManuallyCorrected.nii.gz is exist
                        if source_type == NON_USEFUL_VOLUME:
                            continue
                        elif source_type == REPLACE_MASK_NAME:
                            # Rename ManuallyCorrected.nii.gz as mask.nii.gz
                            target_file = MASK_NAME + SOURCE_EXTENSION
                        else:
                            target_file = source_name
                    else:
                        # Set GlistrBoost.nii.gz as mask if
                        # ManuallyCorrected.nii.gz is not exist
                        if source_type == NON_USEFUL_VOLUME:
                            target_file = MASK_NAME + SOURCE_EXTENSION
                        else:
                            target_file = source_name

                    source_path = os.path.join(sub_from_path, sf)
                    target_file = source_no + "_" + target_file
                    target_path = os.path.join(sub_to_path, target_file)

                    copy(source_path, target_path)

        print("Done")

        return


if __name__ == "__main__":

    input_dir = "/home/quqixun/data/BraTS2017"
    output_dir = os.path.join(os.path.dirname(os.getcwd()), DATA_FOLDER)

    BTCReorganize(input_dir, output_dir)
