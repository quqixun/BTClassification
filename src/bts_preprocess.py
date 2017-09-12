# Brain Tumor Segmentation
# Script for Preprocessing
# Author: Qixun Qu
# Date: 2017/09/10

'''

Class BTSPreprocess

--- Correct bias via N4BiasFieldCorrection
--- Simple Image Enhancement and Normalization
--- Merge four volumes (Flair, T1, T1c and T2) into one volume
--- Removing Surrounding Zeros to Keep Minimum Volume

'''



import os
import subprocess
import numpy as np
from skimage import io
from bts_settings import *
from multiprocessing import Pool, cpu_count
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection



def unwrap_preprocess(arg, **kwarg):
    return BTSPreprocess._preprocess(*arg, **kwarg)


class BTSPreprocess():

    def __init__(self, input_dir, output_dir):

        self.t1 = None
        self.t2 = None
        self.t1c = None
        self.mask = None
        self.flair = None
        self.brain = np.zeros(BRAIN_SHAPE)

        self.input_dir = input_dir
        self.mask_folder = os.path.join(output_dir, "Mask")
        self.volume_folder = os.path.join(output_dir, "Volume")

        if not os.path.isdir("temp"):
            os.makedirs("temp")
        
        if not os.path.isdir(self.volume_folder):
            os.makedirs(self.volume_folder)
        
        if not os.path.isdir(self.mask_folder):
            os.makedirs(self.mask_folder)

        print("Preprocessing on volumes in " + input_dir + "\n")
        self._multi_preprocess()
        print()

        return


    def _multi_preprocess(self):

        dirs = os.listdir(self.input_dir)
        
        pool = Pool(processes=cpu_count())
        pool.map(unwrap_preprocess, zip([self]*len(dirs), dirs))

        return


    def _preprocess(self, d):

        print("Preprocessing on no." + d + " volum.")

        self.flair = self._get_volume(d, "Flair")
        self.t1 = self._get_volume(d, "T1")
        self.t1c = self._get_volume(d, "T1c")
        self.t2 = self._get_volume(d, "T2")

        mask_path = os.path.join(self.input_dir, d, d + "_Mask.mha")
        self.mask = io.imread(mask_path, plugin="simpleitk")

        self._merge_to_one_volume()
        self._keep_minimum_volume(d)

        return


    def _get_volume(self, volume_dir, volume_name):
        
        volume_file = volume_dir + "_" + volume_name + ".mha"
        volume_path = os.path.join(self.input_dir, volume_dir, volume_file)
        
        if volume_name == "T1" or volume_name == "T1c":
            temp_path = "temp\\" + volume_dir + volume_name + ".mha"
            self._correction_bias(volume_path, temp_path)
            volume = io.imread(temp_path, plugin="simpleitk")
            os.remove(temp_path)
        else:
            volume = io.imread(volume_path, plugin="simpleitk")

        volume_min = np.min(volume)
        if volume_min < 0:
            volume[np.where(volume == 0)] = volume_min
            volume = volume - np.min(volume)

        volume = self._volume_enhance_norm(volume)

        return volume


    def _correction_bias(self, volume_path, temp_path):

        n4 = N4BiasFieldCorrection()
        n4.inputs.dimension = 3
        n4.inputs.input_image = volume_path
        n4.inputs.output_image = temp_path

        n4.inputs.n_iterations = N4_ITERATION
        n4.inputs.shrink_factor = N4_SHRINK_FACTOR
        n4.inputs.convergence_threshold = N4_THRESHOLD
        n4.inputs.bspline_fitting_distance = N4_BSPLINE

        # subprocess.call(n4.cmdline.split(" "))
        devnull = open(os.devnull, 'w')
        subprocess.call(n4.cmdline.split(" "), stdout=devnull, stderr=devnull)

        return


    def _volume_enhance_norm(self, volume):

        volume = volume / np.max(volume)
        volume = np.power(volume, 1.5)
        volume = (volume - np.mean(volume)) / np.std(volume)

        return volume


    def _merge_to_one_volume(self):

        self.brain[..., 0] = self.flair
        self.brain[..., 1] = self.t1
        self.brain[..., 2] = self.t1c
        self.brain[..., 3] = self.t2

        return


    def _keep_minimum_volume(self, volume_no):

        def sub_array(arr, index_begin, index_end):
            return arr[index_begin[0] : index_end[0],
                       index_begin[1] : index_end[1],
                       index_begin[2] : index_end[2]]


        def replace_array(arr, rep_arr, index_begin, index_end):
            arr[index_begin[0] : index_end[0],
                index_begin[1] : index_end[1],
                index_begin[2] : index_end[2]] = rep_arr
            return arr


        volume_sum = np.sum(self.brain, axis=3)
        min_volume_sum = np.min(volume_sum)

        non_zero_index = np.where(volume_sum > min_volume_sum)
        dims_begin = [np.min(nzi) for nzi in non_zero_index]
        dims_end = [np.max(nzi) + 1 for nzi in non_zero_index]

        new_volume = sub_array(self.brain, dims_begin, dims_end)
        new_mask = sub_array(self.mask, dims_begin, dims_end)

        new_volume_path = os.path.join(self.volume_folder, volume_no + ".npy")
        new_mask_path = os.path.join(self.mask_folder, volume_no + ".npy")

        np.save(new_volume_path, new_volume.astype(np.float32))
        np.save(new_mask_path, new_mask.astype(np.uint8))

        return



if __name__ == "__main__":

    input_dir = "E:\\ms\\data\\HGG"
    output_dir = "E:\\ms\\data\\HGGPre"
    BTSPreprocess(input_dir, output_dir)

    input_dir = "E:\\ms\\data\\LGG"
    output_dir = "E:\\ms\\data\\LGGPre"
    BTSPreprocess(input_dir, output_dir)
