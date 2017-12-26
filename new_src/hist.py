import os
import numpy as np
from tqdm import *
import nibabel as nib
import matplotlib.pyplot as plt


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data", "Original", "BraTS")


def get_volume_path(dir_path, grade, vtype):
    volume_paths = []
    grade_dir = os.path.join(dir_path, grade)
    subjects = os.listdir(grade_dir)
    for subject in subjects:
        subject_dir = os.path.join(grade_dir, subject)
        for scan_name in os.listdir(subject_dir):
            if vtype in scan_name:
                scan_path = os.path.join(subject_dir, scan_name)
                volume_paths.append(scan_path)
    return volume_paths


def load_volumes(volume_paths):
    volumes = []
    for path in tqdm(volume_paths):
        volumes.append(nib.load(path).get_data())
    return volumes


def compute_hist(volume):
    volume = np.round(volume)
    bins = np.arange(np.min(volume), np.max(volume))
    hist = np.histogram(volume, bins=bins, density=True)

    x = hist[1][2:]
    y = hist[0][1:]

    return x, y


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data", "Original", "BraTS")

vtype = "flair"
hgg_paths = get_volume_path(data_dir, "HGGTrimmed", vtype)
lgg_paths = get_volume_path(data_dir, "LGGTrimmed", vtype)

print("Loading HGG volumes ...")
hgg_volumes = load_volumes(hgg_paths)
print("Loading LGG volumes ...")
lgg_volumes = load_volumes(lgg_paths)

plt.figure()
plt.title("Histogram of " + vtype + " Volumes", fontsize=12)
for volume in hgg_volumes:
    x, y = compute_hist(volume)
    # plt.plot(x / np.max(volume), y, "r", lw=0.3, label="HGG")
    volume_obj = volume[volume > 0]
    plt.plot((x - np.mean(volume_obj)) / np.std(volume_obj), y, "r", lw=0.3, label="HGG")
for volume in lgg_volumes:
    x, y = compute_hist(volume)
    # plt.plot(x / np.max(x), y, "b", lw=0.3, label="LGG")
    volume_obj = volume[volume > 0]
    plt.plot((x - np.mean(volume_obj)) / np.std(volume_obj), y, "b", lw=0.3, label="LGG")
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc='best')
plt.xlabel("Intensity", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid("on", linestyle="--", linewidth=0.5)
plt.show()
