import os
import time
from tqdm import *
import numpy as np
import SimpleITK as sitk


source_path = "E:\\ms\\data\\HGG\\Mask"
target_path = "E:\\ms\\data\\HGGPre\\Mask"
files = os.listdir(source_path)

for f in tqdm(files):
	save_path = os.path.join(target_path, f.replace(".mha", ".npy"))
	file_path = os.path.join(source_path, f)
	if not os.path.exists(save_path):
		mask = sitk.ReadImage(file_path)
		mask = sitk.GetArrayFromImage(mask)
		if mask.dtype != np.uint8:
			mask = mask.astype(np.uint8)
		
		np.save(save_path, mask)
		time.sleep(0.1)


# for i in range(220):
# 	mask1_path = "E:\\ms\\data\\HGG\\Mask\\" + str(i) + ".mha"
# 	mask2_path = "E:\\ms\\data\\HGGPre\\Mask\\" + str(i) + ".npy"

# 	mask1 = sitk.ReadImage(mask1_path)
# 	mask1 = sitk.GetArrayFromImage(mask1)
# 	mask2 = np.load(mask2_path)

# 	diff = mask1 - mask2
# 	print(i, np.sum(diff), np.max(diff), np.min(diff))
