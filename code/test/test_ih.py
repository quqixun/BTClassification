# Test for image enhancement

import os
import cv2
import imageio
from tqdm import *
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate as rot


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


def plot_dft(img, mag, pha):

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Image")
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("Magnitude")
    plt.imshow(mag, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("Phase")
    plt.imshow(pha, cmap="gray")
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
    for i in tqdm(range(slices_num)):
        temp = data[:, :, i]
        mask = np.where(temp == 0)
        ftemp = dtf_filt(temp, hfil)
        ftemp[mask] = 0
        filt_data[:, :, i] = ftemp

    return filt_data


def mip(data, start, end, size):

    mips = []
    fdmax = np.max(filt_data)
    angles = range(start, end)
    for a in tqdm(angles):
        rdata = rot(data, a, axes=(0, 2), reshape=False)
        temp = np.amax(rdata, axis=0) / fdmax * 255
        temp = cv2.resize(temp, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        mips.append(temp.astype(np.uint8))

    return mips


print("Read data path")
work_path = os.path.dirname(os.getcwd())
data_path = os.path.join(work_path, "dataset")
file_name = os.listdir(data_path)
file_path = [os.path.join(data_path, f) for f in file_name]

vidx = 0
print("Load data: ", file_path[vidx])
volm = nib.load(file_path[vidx])
data = np.array(volm.get_data())
data = np.rot90(data, 1, axes=(1, 2))

print("Filter each slice")
filt_data = vol_filt(data, 0.01, "highpass")

print("Maximum intensity projection")
start, end, size = 0, 180, 200
mips = mip(filt_data, start, end, size)

print("Save projections into gif file")
imageio.mimsave("mips_dft_" + str(vidx) + ".gif", mips)

print("Done")
