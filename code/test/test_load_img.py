# Test load volume from files

import os
import numpy as np
import nibabel as nib
import matplotlib.cbook
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


print("Read data path")
work_path = os.path.dirname(os.getcwd())
data_path = os.path.join(work_path, "dataset")
file_name = os.listdir(data_path)
file_path = [os.path.join(data_path, f) for f in file_name]

vidx = 0
print("Load data: ", file_path[vidx])
volm = nib.load(file_path[vidx])
data = np.array(volm.get_data())
data = np.rot90(data, 1, axes=(0, 1))

idxs = range(45, 75)  # vidx = 0
# idxs = range(55, 90)  # vidx = 1
plt.figure(file_name[vidx], figsize=(3, 3))
plt.axis("off")
for i in idxs:
  img = np.flip(data[:, :, i], 0)
  plt.imshow(img, cmap="gray")
  plt.title(i)
  plt.draw()
  plt.pause(.2)

plt.show()
print("Done")
