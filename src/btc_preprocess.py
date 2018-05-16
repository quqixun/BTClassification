from __future__ import print_function


import os
import warnings
import numpy as np
import nibabel as nib
from multiprocessing import Pool, cpu_count
from scipy.ndimage.interpolation import zoom


# Ignore the warning caused by SciPy
warnings.simplefilter("ignore", UserWarning)


def unwrap_preprocess(arg, **kwarg):
    return BTCPreprocess._preprocess(*arg, **kwarg)


class BTCPreprocess(object):

    def __init__(self, input_dirs, output_dirs, volume_type=None):
        self.in_paths, self.out_paths, self.mask_paths = \
            self.generate_paths(input_dirs, output_dirs, volume_type)
        return

    def run(self, is_mask=True, non_mask_coeff=0.333, processes=-1):
        print("\nPreprocessing on the sample in BraTS dataset.\n")
        num = len(self.in_paths)
        paras = zip([self] * num, self.in_paths, self.out_paths, self.mask_paths,
                    [non_mask_coeff] * num, [is_mask] * num)
        if processes == -1 or processes > cpu_count():
            processes = cpu_count()
        pool = Pool(processes=processes)
        pool.map(unwrap_preprocess, paras)
        return

    def _preprocess(self, in_path, to_path, mask_path,
                    non_mask_coeff=0.333, is_mask=True):
        try:
            print("Preprocessing on: " + in_path)
            volume = self.load_nii(in_path)
            if is_mask:
                mask = self.load_nii(mask_path)
                volume = self.segment(volume, mask, non_mask_coeff)
            volume = self.trim(volume)
            volume = self.resize(volume, [112, 112, 96])
            self.save2nii(to_path, volume)
        except RuntimeError:
            print("\tFailed to rescal:" + in_path)
            return
        return

    @staticmethod
    def generate_paths(in_dirs, out_dirs, volume_type=None):
        def create_dir(path):
            if not os.path.isdir(path):
                os.makedirs(path)
            return

        in_paths, out_paths, mask_paths = [], [], []
        for in_dir, out_dir in zip(in_dirs, out_dirs):
            if not os.path.isdir(in_dir):
                print("Input folder {} is not exist.".format(in_dir))
                continue
            create_dir(out_dir)

            for subject in os.listdir(in_dir):
                subject_dir = os.path.join(in_dir, subject)
                subject2dir = os.path.join(out_dir, subject)
                create_dir(subject2dir)

                scan_names = os.listdir(subject_dir)
                for scan_name in scan_names:
                    if "seg" in scan_name:
                        scan_mask_path = os.path.join(subject_dir, scan_name)

                for scan_name in scan_names:
                    if "seg" in scan_name:
                        continue

                    if volume_type is not None:
                        if volume_type not in scan_name:
                            continue

                    in_paths.append(os.path.join(subject_dir, scan_name))
                    out_paths.append(os.path.join(subject2dir, scan_name))
                    mask_paths.append(scan_mask_path)

        return in_paths, out_paths, mask_paths

    @staticmethod
    def load_nii(path):
        return np.rot90(nib.load(path).get_data(), 3)

    @staticmethod
    def segment(volume, mask, non_mask_coeff=0.333):

        if np.min(volume) != 0:
            volume -= np.min(volume)

        non_mask_idx = np.where(mask == 0)
        segged = np.copy(volume)
        segged[non_mask_idx] = segged[non_mask_idx] * non_mask_coeff

        return segged

    @staticmethod
    def trim(volume):
        non_zero_slices = [i for i in range(volume.shape[-1])
                           if np.sum(volume[..., i]) > 0]
        volume = volume[..., non_zero_slices]

        row_begins, row_ends = [], []
        col_begins, col_ends = [], []
        for i in range(volume.shape[-1]):
            non_zero_pixels = np.where(volume > 0)
            row_begins.append(np.min(non_zero_pixels[0]))
            row_ends.append(np.max(non_zero_pixels[0]))
            col_begins.append(np.min(non_zero_pixels[1]))
            col_ends.append(np.max(non_zero_pixels[1]))

        row_begin, row_end = min(row_begins), max(row_ends)
        col_begin, col_end = min(col_begins), max(col_ends)

        rows_num = row_end - row_begin
        cols_num = col_end - col_begin
        more_col_len = rows_num - cols_num
        more_col_len_left = more_col_len // 2
        more_col_len_right = more_col_len - more_col_len_left
        col_begin -= more_col_len_left
        col_end += more_col_len_right
        len_of_side = rows_num + 1

        trimmed = np.zeros([len_of_side, len_of_side, volume.shape[-1]])
        for i in range(volume.shape[-1]):
            trimmed[..., i] = volume[row_begin:row_end + 1,
                                     col_begin:col_end + 1, i]
        return trimmed

    @staticmethod
    def resize(volume, target_shape):
        old_shape = list(volume.shape)
        factor = [n / float(o) for n, o in zip(target_shape, old_shape)]
        resized = zoom(volume, zoom=factor, order=1, prefilter=False)
        resized = resized[:, 8:104, :]
        return resized

    @staticmethod
    def save2nii(to_path, volume):
        volume = volume.astype(np.int16)
        volume = np.rot90(volume, 3)
        volume_nii = nib.Nifti1Image(volume, np.eye(4))
        nib.save(volume_nii, to_path)
        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data")
    hgg_input_dir = os.path.join(data_dir, "HGG")
    lgg_input_dir = os.path.join(data_dir, "LGG")
    input_dirs = [hgg_input_dir, lgg_input_dir]

    # Enhanced Tumor
    is_mask = True
    non_mask_coeff = 0.333
    hgg_output_dir = os.path.join(data_dir, "HGGSegTrimmed")
    lgg_output_dir = os.path.join(data_dir, "LGGSegTrimmed")
    output_dirs = [hgg_output_dir, lgg_output_dir]

    prep = BTCPreprocess(input_dirs, output_dirs, "t1ce")
    prep.run(non_mask_coeff=non_mask_coeff,
             is_mask=is_mask, processes=-1)

    # Non-Enhanced Tumor
    is_mask = False
    hgg_output_dir = os.path.join(data_dir, "HGGTrimmed")
    lgg_output_dir = os.path.join(data_dir, "LGGTrimmed")
    output_dirs = [hgg_output_dir, lgg_output_dir]

    prep = BTCPreprocess(input_dirs, output_dirs, "t1ce")
    prep.run(is_mask=is_mask, processes=-1)
