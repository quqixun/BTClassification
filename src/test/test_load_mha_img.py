# Test load volume from files

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.cbook
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


print("Read data path")
work_path = os.path.dirname(os.getcwd())
data_path = os.path.join(work_path, "dataset", "mha")
file_name = os.listdir(data_path)
file_path = [os.path.join(data_path, f) for f in file_name]

vidx = 2
print("Load data: ", file_path[vidx])
volm = sitk.ReadImage(file_path[vidx])
data = sitk.GetArrayFromImage(volm)

idx = 50
plt.figure()
plt.imshow(data[idx, :, :] == 4, cmap="gray")
plt.show()
print(np.unique(data[idx, :, :]))

# idxs = range(0, 155)
# plt.figure(file_name[vidx], figsize=(3, 3))
# plt.axis("off")
# for i in idxs:
#     img = data[i, :, :]
#     plt.imshow(img, cmap="gray")
#     plt.title(i)
#     plt.draw()
#     plt.pause(.5)

# plt.show()
# print("Done")
