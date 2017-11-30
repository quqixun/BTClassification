import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from btc_settings import *


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, DATA_FOLDER, PATCHES_FOLDER)
labels_path = os.path.join(parent_dir, DATA_FOLDER, LABEL_FILE)

file_names = os.listdir(data_dir)
labels = pd.read_csv(labels_path)

grade2_paths = []
grade3_paths = []
grade4_paths = []

for file in file_names:
    grade = labels[GRADE_LABEL][labels[CASE_NO] == file].values[0]

    case_path = os.path.join(data_dir, file, "original.npy")
    if grade == GRADE_II:
        grade2_paths.append(case_path)
    elif grade == GRADE_III:
        grade3_paths.append(case_path)
    elif grade == GRADE_IV:
        grade4_paths.append(case_path)
    else:
        continue

channel = 1
slice_no = 30
num = 20
row = 4


def randomly_select_path(path_list, num):
    indices = np.random.choice(range(len(path_list)), num, replace=False)
    return [path_list[i] for i in indices]


grad2_rnd_paths = randomly_select_path(grade2_paths, num)
grad3_rnd_paths = randomly_select_path(grade3_paths, num)
grad4_rnd_paths = randomly_select_path(grade4_paths, num)


def load_patches(path_list):
    return [np.load(path) for path in path_list]


grade2 = load_patches(grad2_rnd_paths)
grade3 = load_patches(grad3_rnd_paths)
grade4 = load_patches(grad4_rnd_paths)


def plot_volumes(volumes, grade, row, channel, slce_no, name):
    num = len(volumes)
    col = np.ceil(num / row)
    plt.figure(num=name + "_" + grade)
    for i in range(num):
        plt.subplot(row, col, i + 1)
        plt.axis("off")
        plt.imshow(volumes[i][:, :, slice_no, channel], cmap="gray")


plot_volumes(grade2, "grade2", row, channel, slice_no, VOLUME_TYPES[channel])
plot_volumes(grade3, "grade3", row, channel, slice_no, VOLUME_TYPES[channel])
plot_volumes(grade4, "grade4", row, channel, slice_no, VOLUME_TYPES[channel])
plt.show()
