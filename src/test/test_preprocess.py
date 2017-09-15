# Test for preprocess od a volume

import os
# import time
import subprocess
import numpy as np
# from tqdm import *
from skimage import io
import matplotlib.pyplot as plt
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection


flair_path = "E:\\ms\\data\\HGG\\0\\0_Flair.mha"
t1_path = "E:\\ms\\data\\HGG\\0\\0_T1.mha"
t1c_path = "E:\\ms\\data\\HGG\\0\\0_T1c.mha"
t2_path = "E:\\ms\\data\\HGG\\0\\0_T2.mha"
mask_path = "E:\\ms\\data\\HGG\\0\\0_Mask.mha"

path = [flair_path, t1_path, t1c_path, t2_path]

'''
Part 1. Preprocess - N4FieldBiasCorrection
'''

def bias_correction(img_path, output_path):

    if not os.path.exists(output_path):
	    n4 = N4BiasFieldCorrection()
	    n4.inputs.dimension = 3
	    n4.inputs.input_image = img_path
	    n4.inputs.output_image = output_path

	    n4.inputs.bspline_fitting_distance = 500
	    n4.inputs.shrink_factor = 10
	    n4.inputs.n_iterations = [100, 100, 60, 40]
	    n4.inputs.convergence_threshold = 1e-4

	    # subprocess.call(n4.cmdline.split(" "))
	    devnull = open(os.devnull, 'w')
	    subprocess.call(n4.cmdline.split(" "), stdout=devnull, stderr=devnull)

    return

output_path = "temp\\output.mha"



# print("Removing bias")
# bias_correction(volm_path, output_path)

# volm = io.imread(volm_path, plugin="simpleitk")
# mask = io.imread(mask_path, plugin="simpleitk")
# output = io.imread(output_path, plugin="simpleitk")

# os.remove(output_path)

# idx = 80
# plt.figure()
# plt.subplot(2, 2, 1)
# plt.axis("off")
# plt.imshow(volm[idx, :, :], cmap="gray")
# plt.subplot(2, 2, 2)
# plt.axis("off")
# plt.imshow(mask[idx, :, :], cmap="gray")
# plt.subplot(2, 2, 3)
# plt.axis("off")
# plt.imshow(output[idx, :, :], cmap="gray")
# plt.show()


'''
Part 2. Modify Histogram
'''

# temp = output / np.max(output)
# temp = np.power(temp, 1.5) - 0.5

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.axis("off")
# plt.imshow(output[idx, :, :], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.axis("off")
# plt.imshow(temp[idx, :, :], cmap="gray")
# plt.show()

'''
Part 3. Resize the volume
'''

def sub_array(arr, index_begin, index_end):
	return arr[index_begin[0] : index_end[0],
               index_begin[1] : index_end[1],
               index_begin[2] : index_end[2]]


def replace_array(arr, rep_arr, index_begin, index_end):
	arr[index_begin[0] : index_end[0],
        index_begin[1] : index_end[1],
        index_begin[2] : index_end[2]] = rep_arr
	return arr




flair = io.imread(flair_path, plugin="simpleitk")
t1 = io.imread(t1_path, plugin="simpleitk")
t1c = io.imread(t1c_path, plugin="simpleitk")
t2 = io.imread(t2_path, plugin="simpleitk")
mask = io.imread(mask_path, plugin="simpleitk")

volm_shape = list(flair.shape) + [4]
volm = np.zeros(volm_shape)
volm[..., 0] = flair
volm[..., 1] = t1
volm[..., 2] = t1c
volm[..., 3] = t2
# volm = np.dstack((flair, t1, t1c, t2))
# print(volm.shape)

volm_sum = flair + t1 + t1c + t2

# volm = read_volume(volm_path)
# mask = read_volume(mask_path)

patch_shape = [32, 32, 32]

non_zero_index = np.where(volm_sum > 0)
dims_begin = []
dims_end = []
for nzi in non_zero_index:
	dims_begin.append(np.min(nzi))
	dims_end.append(np.max(nzi) + 1)

volm_obj = sub_array(volm, dims_begin, dims_end)
mask_obj = sub_array(mask, dims_begin, dims_end)

print(volm_obj.shape)

volm_obj_shape = volm_obj.shape[:3]
new_volm_obj_shape = [(vs//ps+1)*ps for vs, ps in zip(volm_obj_shape, patch_shape)] + [4]
new_dims_begin = [(nvs-vs)//2 for nvs, vs in zip(new_volm_obj_shape, volm_obj_shape)]
new_dims_end = [ndb+vs for ndb, vs in zip(new_dims_begin, volm_obj_shape)]

new_volm_obj = np.zeros(new_volm_obj_shape)
new_volm_obj = replace_array(new_volm_obj, volm_obj, new_dims_begin, new_dims_end)
new_volm_obj = new_volm_obj.astype(np.float32)

new_mask_obj = np.zeros(new_volm_obj_shape[:3])
new_mask_obj = replace_array(new_mask_obj, mask_obj, new_dims_begin, new_dims_end)
new_mask_obj = new_mask_obj.astype(np.uint8)

new_volm_path = "new_volm.npy"
new_volm_mask_path = "new_volm_mask.npy"

print(new_volm_obj.shape)

np.save(new_volm_path, new_volm_obj)
np.save(new_volm_mask_path, new_mask_obj)

img_index = 70
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(new_volm_obj[img_index, :, :, 0], cmap="gray")
plt.subplot(2, 3, 4)
plt.imshow(new_volm_obj[img_index, :, :, 1], cmap="gray")
plt.subplot(2, 3, 2)
plt.imshow(new_volm_obj[img_index, :, :, 2], cmap="gray")
plt.subplot(2, 3, 5)
plt.imshow(new_volm_obj[img_index, :, :, 3], cmap="gray")
plt.subplot(2, 3, (3, 6))
plt.imshow(new_mask_obj[img_index, :, :], cmap="gray")
plt.show()

# Output:
# new_volm_obj
# new_mask_obj


# def get_new_volume(ori_volume, new_shape, begin, end, dtype, vtype):
        #     new_volume = np.zeros(new_shape)
        #     if vtype == "brain":
        #         new_volume[..., 0] = np.min(self.flair)
        #         new_volume[..., 1] = np.min(self.t1)
        #         new_volume[..., 2] = np.min(self.t1c)
        #         new_volume[..., 3] = np.min(self.t2)
        #     new_volume = replace_array(new_volume, ori_volume, begin, end)
        #     new_volume = new_volume.astype(dtype)
        #     return new_volume


# volume_shape = volume.shape[:3]
# new_volume_shape = [(vs // ps + 1) * ps for vs, ps in zip(volume_shape, PATCH_SHAPE)] + [4]
# new_dims_begin = [(nvs - vs) // 2 for nvs, vs in zip(new_volume_shape, volume_shape)]
# new_dims_end = [ndb + vs for ndb, vs in zip(new_dims_begin, volume_shape)]

# new_volume = get_new_volume(volume, new_volume_shape, new_dims_begin,
# 	                        new_dims_end, np.float32, "brain")
# new_mask = get_new_volume(mask, new_volume_shape[:3], new_dims_begin,
# 	                      new_dims_end, np.uint8, "mask")
