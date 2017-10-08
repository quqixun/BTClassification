import os
import numpy as np
import pandas as pd
from btc_settings import *


# Path settings
parent_dir = os.path.dirname(os.getcwd())

input_dir = os.path.join(parent_dir, DATA_FOLDER, AUGMENT_FOLDER)
label_file = os.path.join(parent_dir, DATA_FOLDER, LABEL_FILE)

# Load labels and cases' names
labels = pd.read_csv(label_file)
case_names = os.listdir(input_dir)


#
# # Plot one case
# import matplotlib.pyplot as plt
#
# idx = 7
# case_no = case_names[idx]
# case_grade = labels[GRADE_LABEL][labels[CASE_NO] == case_no].values[0]

# case_path = os.path.join(input_dir, case_no)
# volume_names = os.listdir(case_path)

# # Plot the middle slice of each volume

# slice_no = int((PARTIAL_SIZE + 1) / 2)
# channel = 3

# volumes, vmins, vmaxs = [], [], []
# for vn in volume_names:
#     volume = np.load(os.path.join(case_path, vn))
#     volumes.append(volume)
#     vmins.append(np.min(volume[..., channel]))
#     vmaxs.append(np.max(volume[..., channel]))

# vmin = np.min(vmins)
# vmax = np.max(vmaxs)

# column_num = 10
# volumes_num = len(volumes)
# row_num = np.ceil(volumes_num / column_num)

# plt.figure()
# for i in range(volumes_num):
#     vc = volumes[i][..., channel]
#     plt.subplot(row_num, column_num, i + 1)
#     plt.axis("off")
#     plt.imshow(vc[:, :, slice_no], cmap="gray", vmin=vmin, vmax=vmax)
# mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
# plt.show()


# Obtain amount of each grade type
grade2_num = 0
grade3_num = 0
grade4_num = 0

for cn in case_names:
    case_grade = labels[GRADE_LABEL][labels[CASE_NO] == cn].values[0]
    volumes_num = len(os.listdir(os.path.join(input_dir, cn)))
    if case_grade == GRADE_II:
        grade2_num += volumes_num
    elif case_grade == GRADE_III:
        grade3_num += volumes_num
    elif case_grade == GRADE_IV:
        grade4_num += volumes_num
    else:
        # print("The grade of " + cn + " is unknown.")
        continue

print("Total number of patches: ", str(grade2_num + grade3_num + grade4_num))
print("The number of Grade II patches: ", str(grade2_num))
print("The number of Grade III patches: ", str(grade3_num))
print("The number of Grade IV patches: ", str(grade4_num))


if os.path.isfile("train_cases.txt"):
    os.remove("train_cases.txt")
open("train_cases.txt", "a").close()

if os.path.isfile("validate_cases.txt"):
    os.remove("validate_cases.txt")
open("validate_cases.txt", "a").close()


def generate_cases_set(labels, grade):
    # Set cases for training and validating
    cases = labels[CASE_NO][labels[GRADE_LABEL] == grade].values.tolist()

    # Generate cases' names for training set and validating set
    cases_num = len(cases)
    train_num = int(cases_num * PROPORTION)
    index = list(np.arange(cases_num))
    train_index = list(np.random.choice(cases_num, train_num, replace=False))
    validate_index = [i for i in index if i not in train_index]
    train = [cases[i] for i in train_index]
    validate = [cases[i] for i in validate_index]

    return train, validate


# Write cases' names into text files
def save_cases_names(file, cases):
    txt = open(file, "a")
    for case in cases:
        txt.write("{}\n".format(case))


grade2_train, grade2_validate = generate_cases_set(labels, GRADE_II)
save_cases_names("train_cases.txt", grade2_train)
save_cases_names("validate_cases.txt", grade2_validate)

grade3_train, grade3_validate = generate_cases_set(labels, GRADE_III)
save_cases_names("train_cases.txt", grade3_train)
save_cases_names("validate_cases.txt", grade3_validate)

grade4_train, grade4_validate = generate_cases_set(labels, GRADE_IV)
save_cases_names("train_cases.txt", grade4_train)
save_cases_names("validate_cases.txt", grade4_validate)
