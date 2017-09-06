# Test for image enhancement

import os
import cv2
import imageio
import subprocess
from tqdm import *
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate as rot
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection


def load_volume(path):

    volm = nib.load(path)
    data = np.array(volm.get_data())
    data = np.rot90(data, 1, axes=(1, 2))

    return data


def plot_img(img, name=None, title=None):

	plt.figure(name, figsize=(6, 6))
	plt.axis("off")
	plt.title(title)
	plt.imshow(img, cmap="gray")
	plt.show()

	return


def plot_hist(img, t=1):

	img[np.where(img < t)] = 0
	hist = cv2.calcHist([img], [0], None, [255], [1, 255])
	plt.figure(figsize=(15, 6))
	plt.subplot(1, 2, 1)
	plt.axis("off")
	plt.imshow(img, cmap="gray")
	plt.subplot(1, 2, 2)
	plt.bar(np.arange(1, 256), hist)
	plt.show()

	return


def plot_imgs(imgs, titles):

    plt.figure(figsize=(15, 6))
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        plt.axis("off")
        plt.title(titles[i])
        plt.imshow(imgs[i], cmap="gray")

    plt.show()

    return


def gaussian2D(size, sigma, mu, filt="lowpass"):

	w1 = np.linspace(-1, 1, size[0])
	w2 = np.linspace(-1, 1, size[1])
	x, y = np.meshgrid(w1, w2)
	d = np.sqrt(x*x + y*y)
	g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
	g = g / np.max(g)

	if filt == "highpass":
		return 1 - g

	return g


def dtf_filt(img, filt):

	dft = np.fft.fft2(img)
	dft = np.fft.fftshift(dft)
	filt_dft = np.multiply(dft, filt)
	filt_img = np.abs(np.fft.ifft2(filt_dft))

	return filt_img


def vol_filt(data, sigma, filt="lowpass"):

	slices_num = data.shape[2]
	filt_data = np.zeros(data.shape)
	hfil = gaussian2D(data.shape[0:2], sigma, 0, filt)
	for i in range(slices_num):
		temp = data[:, :, i]
		mask = np.where(temp == 0)
		ftemp = dtf_filt(temp, hfil)
		ftemp[mask] = 0
		filt_data[:, :, i] = ftemp

	return filt_data


def mip(data, start, end, size):

	mips = []
	fdmax = np.max(data)
	angles = range(start, end)
	for a in tqdm(angles):
		rdata = rot(data, a, axes=(0, 2), reshape=False)
		temp = np.amax(rdata, axis=0) / fdmax * 255
		temp = cv2.resize(temp, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
		mips.append(temp.astype(np.uint8))

	return mips


def remove_bias(img_path, output_path, bias_path):

    if not os.path.exists(output_path):
        n4 = N4BiasFieldCorrection()
        n4.inputs.dimension = 3
        n4.inputs.input_image = img_path
        n4.inputs.save_bias = True
        n4.inputs.bias_image = bias_path
        n4.inputs.output_image = output_path

        n4.inputs.bspline_fitting_distance = 500
        n4.inputs.shrink_factor = 20
        n4.inputs.n_iterations = [200, 200, 50, 50]
        n4.inputs.convergence_threshold = 1e-4

        # subprocess.call(n4.cmdline.split(" "))
        devnull = open(os.devnull, 'w')
        subprocess.call(n4.cmdline.split(" "), stdout=devnull, stderr=devnull)

    return


'''
Part 0. Load Original Data
'''

print("Read data path")
work_path = os.path.dirname(os.getcwd())
data_path = os.path.join(work_path, "dataset")
file_name = os.listdir(data_path)
file_path = [os.path.join(data_path, f) for f in file_name]

vidx = 0
idx = 65
start, end, size = 0, 180, 200

print("Load data: ", file_path[vidx])
data = load_volume(file_path[vidx])
print("Save images into gifs")
smips = mip(data, start, end, size)
imageio.mimsave(str(vidx) + "\mips_s_" + str(vidx) + ".gif", smips)

bias_path = str(vidx) + "\\" + file_name[vidx].replace(".nii", "_b.nii")
output_path = str(vidx) + "\\" + file_name[vidx].replace(".nii", "_rb.nii")

'''
Part 1. Test DFT
'''

print("Filter each slice")
filt_data = vol_filt(data, 0.01, "highpass")

print("Maximum intensity projection")
mips = mip(filt_data, start, end, size)

print("Save projections into gif file")
imageio.mimsave(str(vidx) + "\mips_dft_" + str(vidx) + ".gif", mips)

print("Done")


'''
Part 2. Field Bias Correction
'''

print("Field Bias Correlation on: ", file_path[vidx])
remove_bias(file_path[vidx], output_path, bias_path)

output = load_volume(output_path)
bias = load_volume(bias_path)

stemp = np.rot90(data, 3, axes=(1, 2))
otemp = np.rot90(output, 3, axes=(1, 2))
btemp = np.rot90(bias, 3, axes=(1, 2))
imgs = [stemp[:, :, idx], otemp[:, :, idx], btemp[:, :, idx]]
imgs = [np.flip(np.rot90(i, 3), 1) for i in imgs]
plot_imgs(imgs, ["Original", "Removed Bias", "Bias"])

print("Maximum intensity projection")
omips = mip(output, start, end, size)
bmips = mip(bias, start, end, size)

print("Save projections into gif file")
imageio.mimsave(str(vidx) + "\mips_rb_" + str(vidx) + ".gif", omips)
imageio.mimsave(str(vidx) + "\mips_b_" + str(vidx) + ".gif", bmips)

print("Done")


'''
Part 3. 
'''

print("Filter the output without bias")
filt_output = vol_filt(output, 0.01, "highpass")

fdtemp = np.rot90(filt_data, 3, axes=(1, 2))
fotemp = np.rot90(filt_output, 3, axes=(1, 2))
imgs = [fdtemp[:, :, idx], fotemp[:, :, idx]]
imgs = [np.flip(np.rot90(i, 3), 1) for i in imgs]
plot_imgs(imgs, ["DFT of Original", "DFT of Removed Bias"])

print("Maximum intensity projection")
fomips = mip(filt_output, start, end, size)

print("Save projections into gif file")
imageio.mimsave(str(vidx) + "\mips_dft_rb_" + str(vidx) + ".gif", fomips)
