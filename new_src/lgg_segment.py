import os
import scipy.misc
import numpy as np
from tqdm import *
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_nii(path):
    return np.fliplr(np.rot90(nib.load(path).get_data(), 1))


def save_nii(data, path):
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return


input_dir = "/home/user4/Desktop/Dataset/LGG"
label_path = os.path.join(input_dir, "TCIA_LGG_cases_159.xlsx")
output_dir = "/home/user4/Desktop/Dataset/LGG-Segment"
class0_output_dir = os.path.join(output_dir, "0")
class1_output_dir = os.path.join(output_dir, "1")

subjects = os.listdir(input_dir)

no_ud = ["LGG-210", "LGG-219", "LGG-229", "LGG-240", "LGG-269"
         "LGG-273", "LGG-277", "LGG-288", "LGG-289", "LGG-304",
         "LGG-308", "LGG-320", "LGG-325"]

labels = pd.read_excel(label_path)

# subjects = ["LGG-310", "LGG-326", "LGG-387", "LGG-515"]
# subjects = []
target_size = [128, 128]
for subject in tqdm(subjects):
    subj_in_dir = os.path.join(input_dir, subject)

    if not os.path.isdir(subj_in_dir):
        continue

    label = labels["1p/19q"][labels["Filename"].values.tolist().index(subject)]

    if label == "n/n":
        class_output_dir = class0_output_dir
    else:
        class_output_dir = class1_output_dir

    subj_out_dir = os.path.join(class_output_dir, subject)
    create_dir(subj_out_dir)

    seg_path = os.path.join(subj_in_dir, subject + "-Segmentation.nii.gz")
    segment = load_nii(seg_path)
    seg_shape = list(segment.shape)

    idx = []
    for i in range(seg_shape[2]):
        if np.sum(segment[:, :, i]) > 0:
            idx.append(i)

    if len(idx) > 3:
        idx = idx[:3]

    for file in os.listdir(subj_in_dir):
        if "Segmentation" in file:
            continue

        if "T1" in file:
            group = "T1c"
            weight = 0.2
        else:
            group = "T2"
            weight = 0.5

        mask = np.copy(segment)
        mask[np.where(mask == 0)] = weight
        volume = load_nii(os.path.join(subj_in_dir, file))

        out_file_dir = os.path.join(subj_out_dir, group)
        create_dir(out_file_dir)
        for i in range(len(idx)):
            data = np.multiply(volume[:, :, idx[i]], mask[:, :, idx[i]])
            data = (data - np.mean(data)) / np.std(data)

            if subject not in no_ud:
                data = np.flipud(data)

            data_shape = list(data.shape)
            if data_shape != target_size:
                factor = [n / s for n, s in zip(target_size, data_shape)]
                data = zoom(data, zoom=factor, order=1, prefilter=False)

            out_file_path = os.path.join(out_file_dir, subject + "-" + group + "-" + str(i))
            np.save(out_file_path, data)
            out_image_path = os.path.join(out_file_dir, subject + "-" + group + "-" + str(i) + ".png")
            scipy.misc.imsave(out_image_path, data)
