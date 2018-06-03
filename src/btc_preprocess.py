# Brain Tumor Classification
# Enhance tumor region in each image.
# Author: Qixun QU
# Copyleft: MIT Licience

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


from __future__ import print_function


import os
import warnings
import numpy as np
import nibabel as nib

from multiprocessing import Pool, cpu_count
from scipy.ndimage.interpolation import zoom


# Ignore the warning caused by SciPy
warnings.simplefilter("ignore", UserWarning)


# Helper function to run in multiple processes
def unwrap_preprocess(arg, **kwarg):
    return BTCPreprocess._preprocess(*arg, **kwarg)


class BTCPreprocess(object):

    def __init__(self, input_dirs, output_dirs, volume_type="t1ce"):
        '''__INIT__

            Generates paths for preprocessing.
            Variables:
            - self.in_paths: a list contains path of each input image.
            - self.out_paths: a list provides path for each output image.
            - self.mask_paths: a list contains path of mask for each input image.

            Inputs:
            -------

            - input_dirs: a list with two lists, [hgg_input_dir, lgg_input_dir],
                          path of the directory which saves input images of\
                          HGG and LGG subjects.
            - output_dirs: a list with teo lists, [hgg_output_dir, lgg_output_dir],
                           path of output directory for every subject in HGG and LGG.
            - volume_type: string, type of brain volume, one of "t1ce", "t1", "t2"
                           or "flair". Default is "t1ce".

        '''

        self.in_paths, self.out_paths, self.mask_paths = \
            self.generate_paths(input_dirs, output_dirs, volume_type)

        return

    def run(self, is_mask=True, non_mask_coeff=0.333, processes=-1):
        '''RUN

            Function to map task to multiple processes.

            Inputs:
            -------

            - is_mask: boolearn, if True, enhance tumor region.
                       Default is True.
            - non_mask_coeff: float from 0 to 1, the coefficient of
                              voxels in non-tumor region. Default is 0.333.
            - processes: int, the number of processes used. Default is -1,
                         which means use all processes.

        '''

        print("\nPreprocessing on the sample in BraTS dataset.\n")
        num = len(self.in_paths)

        # Generate parameters
        paras = zip([self] * num, self.in_paths, self.out_paths, self.mask_paths,
                    [is_mask] * num, [non_mask_coeff] * num)

        # Set the number of processes
        if processes == -1 or processes > cpu_count():
            processes = cpu_count()

        # Map task
        pool = Pool(processes=processes)
        pool.map(unwrap_preprocess, paras)

        return

    def _preprocess(self, in_path, to_path, mask_path,
                    is_mask=True, non_mask_coeff=0.333):
        '''_PREPROCESS

            For each input image, four steps are done:
            -1- If is_mask, enhance tumor region.
            -2- Remove background.
            -3- Resize image.
            -4- Save image.

            Inputs:
            -------

            - in_path: string, path of input image.
            - to_path: string, path of output image.
            - mask_path: string, path of the mask of input image.
            - is_mask: boolearn, if True, enhance tumor region.
                       Default is True.
            - non_mask_coeff: float from 0 to 1, the coefficient of
                              voxels in non-tumor region. Default is 0.333.

        '''

        try:
            print("Preprocessing on: " + in_path)
            # Load image
            volume = self.load_nii(in_path)
            if is_mask:
                # Enhance tumor region
                mask = self.load_nii(mask_path)
                volume = self.segment(volume, mask, non_mask_coeff)
            # Removce background
            volume = self.trim(volume)
            # Resize image
            volume = self.resize(volume, [112, 112, 96])
            # Save image
            self.save2nii(to_path, volume)
        except RuntimeError:
            print("\tFailed to rescal:" + in_path)
            return

        return

    @staticmethod
    def generate_paths(in_dirs, out_dirs, volume_type=None):
        '''GENERATE_PATHS

            Generates three lists with files' paths for prerprocessing.

            Inputs:
            -------

            - input_dirs: a list with two lists, [hgg_input_dir, lgg_input_dir],
                          path of the directory which saves input images of\
                          HGG and LGG subjects.
            - output_dirs: a list with teo lists, [hgg_output_dir, lgg_output_dir],
                           path of output directory for every subject in HGG and LGG.
            - volume_type: string, type of brain volume, one of "t1ce", "t1", "t2"
                           or "flair". Default is "t1ce".

            Outputs:
            --------

            - in_paths: a list contains path of each input image.
            - out_paths: a list provides path for each output image.
            - mask_paths: a list contains path of mask for each input image.

        '''

        # Function to create new directory
        # according to given path
        def create_dir(path):
            if not os.path.isdir(path):
                os.makedirs(path)
            return

        in_paths, out_paths, mask_paths = [], [], []
        for in_dir, out_dir in zip(in_dirs, out_dirs):
            # For HGG or LFF subjects
            if not os.path.isdir(in_dir):
                print("Input folder {} is not exist.".format(in_dir))
                continue

            # Create output folder for HGG or LGG subjects
            create_dir(out_dir)

            for subject in os.listdir(in_dir):
                # For each subject in HGG or LGG
                subject_dir = os.path.join(in_dir, subject)
                subject2dir = os.path.join(out_dir, subject)
                # Create folder for output
                create_dir(subject2dir)

                scan_names = os.listdir(subject_dir)
                # Get path of mask file
                for scan_name in scan_names:
                    if "seg" in scan_name:
                        scan_mask_path = os.path.join(subject_dir, scan_name)

                for scan_name in scan_names:
                    if "seg" in scan_name:
                        continue

                    if volume_type is not None:
                        if volume_type not in scan_name:
                            continue

                    # When find the target volume, save its path
                    # and save paths for its output and mask
                    in_paths.append(os.path.join(subject_dir, scan_name))
                    out_paths.append(os.path.join(subject2dir, scan_name))
                    mask_paths.append(scan_mask_path)

        return in_paths, out_paths, mask_paths

    @staticmethod
    def load_nii(path):
        '''LOAD_NII

            Load image to numpy ndarray from NIfTi file.

            Input:
            ------

            - path: string , path of input image.

            Ouput:
            ------

            - A numpy array of input imgae.

        '''

        return np.rot90(nib.load(path).get_data(), 3)

    @staticmethod
    def segment(volume, mask, non_mask_coeff=0.333):
        '''SEGMENT

            Enhance tumor region by suppressing non-tumor region
            with a coefficient.

            Inuuts:
            -------

            - volume: numpy ndarray, input image.
            - mask: numpy ndarray, mask with segmentation labels.
            - non_mask_coeff: float from 0 to 1, the coefficient of
                              voxels in non-tumor region. Default is 0.333.

            Output:
            -------

            - segged: numpy ndarray, tumor enhanced image.

        '''

        # Set background to 0
        if np.min(volume) != 0:
            volume -= np.min(volume)

        # Suppress non-tumor region
        non_mask_idx = np.where(mask == 0)
        segged = np.copy(volume)
        segged[non_mask_idx] = segged[non_mask_idx] * non_mask_coeff

        return segged

    @staticmethod
    def trim(volume):
        '''TRIM

            Remove unnecessary background around brain.

            Input:
            ------

            - volume: numpy ndarray, input image.

            Output:
            -------

            - trimmed: numpy ndarray, image without unwanted background.

        '''

        # Get indices of slices that have brain's voxels
        non_zero_slices = [i for i in range(volume.shape[-1])
                           if np.sum(volume[..., i]) > 0]
        # Remove slices that only have background
        volume = volume[..., non_zero_slices]

        # In each slice, find the minimum area of brain
        # Coordinates of area are saved
        row_begins, row_ends = [], []
        col_begins, col_ends = [], []
        for i in range(volume.shape[-1]):
            non_zero_pixels = np.where(volume > 0)
            row_begins.append(np.min(non_zero_pixels[0]))
            row_ends.append(np.max(non_zero_pixels[0]))
            col_begins.append(np.min(non_zero_pixels[1]))
            col_ends.append(np.max(non_zero_pixels[1]))

        # Find the maximum area from all minimum areas
        row_begin, row_end = min(row_begins), max(row_ends)
        col_begin, col_end = min(col_begins), max(col_ends)

        # Generate a minimum square area taht includs the maximum area
        rows_num = row_end - row_begin
        cols_num = col_end - col_begin
        more_col_len = rows_num - cols_num
        more_col_len_left = more_col_len // 2
        more_col_len_right = more_col_len - more_col_len_left
        col_begin -= more_col_len_left
        col_end += more_col_len_right
        len_of_side = rows_num + 1

        # Remove unwanted background
        trimmed = np.zeros([len_of_side, len_of_side, volume.shape[-1]])
        for i in range(volume.shape[-1]):
            trimmed[..., i] = volume[row_begin:row_end + 1,
                                     col_begin:col_end + 1, i]
        return trimmed

    @staticmethod
    def resize(volume, target_shape=[112, 112, 96]):
        '''RESIZE

            Resize input image to target shape.
            -1- Resize to [112, 112, 96].
            -2- Crop image to [112, 96, 96].

        '''

        # Shape of input image
        old_shape = list(volume.shape)

        # Resize image
        factor = [n / float(o) for n, o in zip(target_shape, old_shape)]
        resized = zoom(volume, zoom=factor, order=1, prefilter=False)

        # Crop image
        resized = resized[:, 8:104, :]

        return resized

    @staticmethod
    def save2nii(to_path, volume):
        '''SAVE2NII

            Save numpy ndarray to NIfTi image.

            Input:
            ------

            - to_path: string, path of output image.
            - volume: numpy ndarray, preprocessed image.

        '''
        # Rotate image to standard space
        volume = volume.astype(np.int16)
        volume = np.rot90(volume, 3)

        # Convert to NIfTi
        volume_nii = nib.Nifti1Image(volume, np.eye(4))
        # Save image
        nib.save(volume_nii, to_path)

        return


if __name__ == "__main__":

    # Set path for input directory
    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data")
    hgg_input_dir = os.path.join(data_dir, "HGG")
    lgg_input_dir = os.path.join(data_dir, "LGG")
    input_dirs = [hgg_input_dir, lgg_input_dir]

    # Generate Enhanced Tumor
    is_mask = True
    non_mask_coeff = 0.333
    # Set path for output directory
    hgg_output_dir = os.path.join(data_dir, "HGGSegTrimmed")
    lgg_output_dir = os.path.join(data_dir, "LGGSegTrimmed")
    output_dirs = [hgg_output_dir, lgg_output_dir]

    prep = BTCPreprocess(input_dirs, output_dirs, "t1ce")
    prep.run(non_mask_coeff=non_mask_coeff,
             is_mask=is_mask, processes=-1)

    # Generate Non-Enhanced Tumor
    is_mask = False
    # Set path for output directory
    hgg_output_dir = os.path.join(data_dir, "HGGTrimmed")
    lgg_output_dir = os.path.join(data_dir, "LGGTrimmed")
    output_dirs = [hgg_output_dir, lgg_output_dir]

    prep = BTCPreprocess(input_dirs, output_dirs, "t1ce")
    prep.run(is_mask=is_mask, processes=-1)
