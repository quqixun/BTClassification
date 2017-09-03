# Test load volume from files

import os
import nibabel as nib
import matplotlib.pyplot as plt


work_path = os.path.dirname(os.getcwd())
data_path = os.path.join(work_path, "dataset")
file_name = os.listdir(data_path)
file_path = [os.path.join(data_path, f) for f in file_name]


volm = nib.load(file_path[1])
data = volm.get_data()
data_shape = data.shape

plt.ion()
plt.figure(file_name[1])
plt.axis("off")
for i in range(data_shape[2]):
    img = data[:, :, i]
    plt.imshow(img, cmap="gray")
    plt.pause(0.5)
    plt.draw()
