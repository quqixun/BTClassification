# Test Maximum Intensity Projection

import os
import cv2
import imageio
from tqdm import *
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import rotate as rot


work_path = os.path.dirname(os.getcwd())
data_path = os.path.join(work_path, "dataset")
file_name = os.listdir(data_path)
file_path = [os.path.join(data_path, f) for f in file_name]

volm = nib.load(file_path[1])
data = np.array(volm.get_data())
data = np.rot90(data, 1, axes=(1, 2))
dmax = np.max(data)

mips = []
angles_num = 360
img_size = (200, 200)
angles = range(0, angles_num)
for a in tqdm(angles):
	rdata = rot(data, a, axes=(0, 2), reshape=False)
	temp = np.amax(rdata, axis=0) / dmax * 255
	temp = cv2.resize(temp, dsize=img_size, interpolation=cv2.INTER_LINEAR)
	mips.append(temp.astype(np.uint8))

imageio.mimsave("mips.gif", mips)
