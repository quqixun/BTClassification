import os
import numpy as np
import matplotlib.pyplot as plt

# a = np.arange(27)
# a = a.reshape((3, 3, 3))
# print(a)
# print(np.amax(a, axis=1))

# a = np.arange(256)
# print(a[0], a[255])


# x, y = np.meshgrid(np.linspace(-1,1,110), np.linspace(-1,1,110))
# d = np.sqrt(x*x+y*y)
# sigma, mu = 0.05, 0.0
# g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
# g = g / np.max(g)
# print(np.max(g))
# print(g.shape)
# plt.figure()
# plt.imshow(g, cmap="gray")
# plt.show()


# a = np.arange(64)
# print(a)
# a = np.reshape(a, (4, 4, 4))
# print(a)
# b = a[0:2, 0:2, 0:2]
# print(b)
# c = [[[1, 1], [1, 1]], [[1, 1], [1, 1]]]
# a[1:3, 1:3, 1:3] = c
# print(a)

# a = 100
# b = 3
# print(a // b)

# b = np.unique(a)
# print(a)
# print(list(a))

# from bts_settings import *

# print(PATCH_SHAPE)

# names = ["Flair", "T1", "T1c", "T2", "Mask"]
# print(sorted(names))

# for vol_index in range(220):
# 	# vol_index = 6

# 	volume = np.load("E:\\ms\\data\\HGGPre\\Volume\\" + str(vol_index) + ".npy")
# 	mask = np.load("E:\\ms\\data\\HGGPre\\Mask\\" + str(vol_index) + ".npy")

# 	img_index = 70
# 	plt.figure(figsize=(10, 8))
# 	plt.subplot(2, 3, 1)
# 	plt.axis("off")
# 	plt.imshow(volume[img_index, :, :, 0], cmap="gray")
# 	plt.subplot(2, 3, 4)
# 	plt.axis("off")
# 	plt.imshow(volume[img_index, :, :, 1], cmap="gray")
# 	plt.subplot(2, 3, 2)
# 	plt.axis("off")
# 	plt.imshow(volume[img_index, :, :, 2], cmap="gray")
# 	plt.subplot(2, 3, 5)
# 	plt.axis("off")
# 	plt.imshow(volume[img_index, :, :, 3], cmap="gray")
# 	plt.subplot(2, 3, (3, 6))
# 	plt.axis("off")
# 	plt.title(vol_index)
# 	plt.imshow(mask[img_index, :, :], cmap="gray")
# 	plt.show()

# if not os.path.isdir("temp"):
	# os.makedirs("temp")



# pct = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.998]
# imin = 1
# imax = 4000
# pct = np.array([1, 186, 320, 348, 370, 390, 408, 429, 462, 625, 1107])
# temp = (imax - imin) * (pct - pct[0]) / (pct[-1] - pct[0]) + imin
# print(temp)



i = 177
path = "norm\\" + str(i) + ".npy"
v = np.load(path)
print(v.shape, v.dtype)

for p in range(155):
	plt.figure(figsize=(5, 5))
	plt.axis("off")
	plt.title(p)
	plt.imshow(v[p, :, :], cmap="gray")
	plt.show()

