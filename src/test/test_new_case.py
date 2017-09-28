import os
import numpy as np
import pandas as pd
from btc_settings import *
import matplotlib.pyplot as plt


def plot_volumes(volume, temp, title):
    for i in range(shape[2]):
        plt.figure()
        plt.subplot(2, 4, 1)
        plt.axis("off")
        plt.title(title)
        plt.imshow(volume[:, :, i, 0], cmap="gray")
        plt.subplot(2, 4, 2)
        plt.axis("off")
        plt.imshow(temp[:, :, i, 0], cmap="gray")
        plt.subplot(2, 4, 3)
        plt.axis("off")
        plt.imshow(volume[:, :, i, 1], cmap="gray")
        plt.subplot(2, 4, 4)
        plt.axis("off")
        plt.imshow(temp[:, :, i, 1], cmap="gray")
        plt.subplot(2, 4, 5)
        plt.axis("off")
        plt.imshow(volume[:, :, i, 2], cmap="gray")
        plt.subplot(2, 4, 6)
        plt.axis("off")
        plt.imshow(temp[:, :, i, 2], cmap="gray")
        plt.subplot(2, 4, 7)
        plt.axis("off")
        plt.imshow(volume[:, :, i, 3], cmap="gray")
        plt.subplot(2, 4, 8)
        plt.axis("off")
        plt.imshow(temp[:, :, i, 3], cmap="gray")
        plt.show()


def plot_hist(hist):
    x = hist[1][1:]
    y = hist[0]
    i = np.where(y > 0)
    x = x[i]
    y = y[i]
    plt.plot(x, y)
    plt.show()


parent_dir = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(parent_dir, "data", "label.csv"))

grade_ii_index = np.where(labels["Grade_Label"] == GRADE_II_LABEL)[0]
grade_ii_cases = labels["Case"][grade_ii_index].values.tolist()


idx = 1
folder = "Temp/resize"
file = os.path.join(folder, grade_ii_cases[idx] + ".npy")
volume = np.load(file)
shape = volume.shape
temp = np.copy(volume)

for i in range(shape[2]):
    temp[:, :, i, :] = np.fliplr(temp[:, :, i, :])

# plot_volumes(volume, temp, title=grade_ii_cases[idx])
# print(np.min(temp), np.max(temp))
temp = np.round(temp)
# print(np.min(temp), np.max(temp))
hist = np.histogram(temp[..., 0], bins=np.arange(1, np.max(temp[..., 0])), density=True)
plot_hist(hist)

# Smooth the histogram curve
# by Savitzky-Golay filter
# http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
# https://www.wire.tu-bs.de/OLDWEB/mameyer/cmr/savgol.pdf
# https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
