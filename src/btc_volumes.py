# Brain Tumor Classification
# Script for Resizing Brain Volumes
# Author: Qixun Qu
# Create on: 2017/10/29
# Modify on: 2017/10/29

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

Class BTCVolumes

'''


from __future__ import print_function

import os
import warnings
import numpy as np
from btc_settings import *
from multiprocessing import Pool, cpu_count
from scipy.ndimage.interpolation import zoom


# Ignore the warning caused by SciPy
warnings.simplefilter("ignore", UserWarning)


# Helper function to do multiprocessing of
# BTCVolumes._resize_volume
def unwrap_resize_volume(arg, **kwarg):
    return BTCVolumes._resize_volume(*arg, **kwarg)


class BTCVolumes():

    def __init__(self, input_dir, output_dir):
        '''__INIT__

            Initialization of instance.
            - generate folders to store resized brain volume
            - resize brain volumes in multi-processes

            Inputs:
            -------
            - input_dir: string, the path of the directory that
                         keeps preprocessed volume
            - output_dir: string, the path of the directory that
                          will store resized volumes

        '''

        # Check whether the input folder is exist
        if not os.path.isdir(input_dir):
            raise ValueError("Input directory is not exist.")

        # Create output folder if it is not exist
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # Obtain volumes' serial numbers
        self.names = os.listdir(input_dir)

        # Multi-process on resizing volumes
        self._resize_volume_multi(input_dir, output_dir)

        return

    def _resize_volume_multi(self, input_dir, output_dir):
        '''_RESIZE_VOLUME_MULTI

            The function to distribute tasks on processes.

            Inputs:
            -------
            - input_dir: string, the path of the directory that
                         keeps preprocessed volume
            - output_dir: string, the path of the directory that
                          will store resized volumes

        '''

        # Generate paths for all preprocssed volumes
        input_paths = [os.path.join(input_dir, name) for name in self.names]

        print("Resize brain volumes into same shape\n")
        paras = zip([self] * len(self.names),
                    input_paths,
                    [output_dir] * len(self.names),
                    self.names)
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_resize_volume, paras)

        return

    def _resize_volume(self, input_path, output_dir, name):
        '''_RESIZE_VOLUME

            Perform resizing on one brain volume, and it will
            be saved into the given directory.

            Inputs:
            -------
            - input_dir: string, the path of the input volume
            - output_dir: string, the path of the directory that
                          will store resized volume
            - name: string, the serial number of input volume

        '''

        # The function to obtain the horizontal mirror
        # of input volume to enlarge dataset
        def horizontal_mirror(volume):
            return np.fliplr(volume)

        case_no = name.split(".")[0]
        print("Resize brain volume of " + case_no)

        # Load data from input path
        volume = np.load(input_path)
        vshape = list(volume.shape)

        # Remove space around edge, which is zero background
        volume = volume[EDGE_SPACE:vshape[0] - EDGE_SPACE,
                        EDGE_SPACE:vshape[1] - EDGE_SPACE,
                        EDGE_SPACE:vshape[2] - EDGE_SPACE]

        # Obtain original shape
        vshape = list(volume.shape)

        # Compute padding size of right and left sides of volume
        # to make sure that each slice of volume is square
        pad_size = vshape[0] - vshape[1]
        left_pad_size = int(pad_size / 2.0)
        right_pad_size = pad_size - left_pad_size

        # Generate padding part
        vshape[1] = left_pad_size
        left_pad = np.zeros(vshape)
        vshape[1] = right_pad_size
        right_pad = np.zeros(vshape)

        # Pad the volume with zero background, and get new shape
        pad_volume = np.hstack((left_pad, volume, right_pad))
        vshape = list(pad_volume.shape)

        # Resize brain volume by interpolation
        factor = [ns / float(vs) for ns, vs in zip(VOLUME_SHAPE, vshape)]
        resized_volume = zoom(pad_volume, zoom=factor, order=1, prefilter=False)
        resized_volume = resized_volume.astype(volume.dtype)

        # Obtain the horizontal mirror of resized volume
        # to carry out augmentation
        volumes = [resized_volume, horizontal_mirror(resized_volume)]

        # Create folder to keep resized volume
        # if the folder is not exist
        output_sub_dir = os.path.join(output_dir, case_no)
        if not os.path.isdir(output_sub_dir):
            os.makedirs(output_sub_dir)

        # Write file into folder
        for i in range(len(volumes)):
            output_name = case_no + "_" + str(i) + ".npy"
            output_path = os.path.join(output_sub_dir, output_name)
            np.save(output_path, volumes[i])

        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())

    input_dir = os.path.join(parent_dir, DATA_FOLDER,
                             PREPROCESSED_FOLDER, FULL_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, VOLUMES_FOLDER)
    temp_dir = os.path.join(TEMP_FOLDER, VOLUMES_FOLDER)

    BTCVolumes(input_dir, output_dir)
