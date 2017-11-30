import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


parent_dir = os.path.dirname(os.getcwd())
mats_dir = os.path.join(parent_dir, "data", "BT", "mat")

mats_file = os.listdir(mats_dir)
index = 200

mat_path = os.path.join(mats_dir, mats_file[index])
mat_content = {}
mat = h5py.File(mat_path)
mat_content = mat["/cjdata"]

label = int(mat_content["label"][0, 0])
pid = "".join([str(int(p)) for p in mat_content["PID"][:]])
image = np.rot90(np.array(mat_content["image"][:]), 3)
mask = np.rot90(np.array(mat_content["tumorMask"][:]), 3)

plt.figure()
plt.subplot(1, 2, 1)
plt.title(index)
plt.axis("off")
plt.imshow(image, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("mask")
plt.axis("off")
plt.imshow(mask, cmap="gray")
plt.show()
