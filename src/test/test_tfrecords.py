import os
import numpy as np
from tqdm import *
import pandas as pd
import tensorflow as tf
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
    np.random.seed(RANDOM_SEED)
    train_index = list(np.random.choice(cases_num, train_num, replace=False))
    validate_index = [i for i in index if i not in train_index]
    train = [[cases[i], grade] for i in train_index]
    validate = [[cases[i], grade] for i in validate_index]

    return train, validate


# Write cases' names into text files
def save_cases_names(file, cases):
    txt = open(file, "a")
    for case in cases:
        txt.write("{}\n".format(case))


grade2_train, grade2_validate = generate_cases_set(labels, GRADE_II)
# save_cases_names("train_cases.txt", grade2_train)
# save_cases_names("validate_cases.txt", grade2_validate)

grade3_train, grade3_validate = generate_cases_set(labels, GRADE_III)
# save_cases_names("train_cases.txt", grade3_train)
# save_cases_names("validate_cases.txt", grade3_validate)

grade4_train, grade4_validate = generate_cases_set(labels, GRADE_IV)
# save_cases_names("train_cases.txt", grade4_train)
# save_cases_names("validate_cases.txt", grade4_validate)


# Count patches in training set of each grade group


def count_patches(input_dir, dataset):
    num = 0
    for d in dataset:
        num += len(os.listdir(os.path.join(input_dir, d[0])))
    return num


grade2_train_num = count_patches(input_dir, grade2_train)
grade3_train_num = count_patches(input_dir, grade3_train)
grade4_train_num = count_patches(input_dir, grade4_train)

print("Total of training set: ", str(grade2_train_num + grade3_train_num + grade4_train_num))
print("The number of Grade II patches: ", grade2_train_num)
print("The number of Grade III patches: ", grade3_train_num)
print("The number of Grade IV patches: ", grade4_train_num)


grade2_validate_num = count_patches(input_dir, grade2_validate)
grade3_validate_num = count_patches(input_dir, grade3_validate)
grade4_validate_num = count_patches(input_dir, grade4_validate)

print("Total of validating set: ", str(grade2_validate_num + grade3_validate_num + grade4_validate_num))
print("The number of Grade II patches: ", grade2_validate_num)
print("The number of Grade III patches: ", grade3_validate_num)
print("The number of Grade IV patches: ", grade4_validate_num)

train_cases = grade2_train + grade3_train + grade4_train
validate_cases = grade2_validate + grade3_validate + grade4_validate
save_cases_names("train_cases.txt", train_cases)
save_cases_names("validate_cases.txt", validate_cases)

# print(train_cases)
# print(validate_cases)


# size49_num = 0
# case_num = len(case_names)
# for idx in tqdm(range(case_num)):
#     # idx = 9
#     case_no = case_names[idx]
#     case_path = os.path.join(input_dir, case_no)
#     volumes_path = os.listdir(case_path)
#     for vp in volumes_path:
#         volume_path = os.path.join(case_path, vp)
#         # print(volume_path)

#         volume = np.load(volume_path)
#         # print(volume.shape, np.mean(volume), np.std(volume))
#         if list(volume.shape)[0] == 49:
#             size49_num += 1

# print(size49_num)


partial_train_set = [["TCGA-CS-4942", 0], ["TCGA-CS-4944", 1], ["TCGA-02-0006", 2],
                     ["TCGA-02-0009", 2], ["TCGA-02-0011", 2], ["TCGA-02-0027", 2]]
partial_validate_set = [["TCGA-CS-6668", 0], ["TCGA-CS-5393", 1], ["TCGA-02-0033", 2],
                        ["TCGA-02-0034", 2], ["TCGA-02-0037", 2], ["TCGA-02-0046", 2]]

partial_input_dir = os.path.join(parent_dir, DATA_FOLDER, "Augmented_Partial")
output_dir = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER)
partial_train_set_path = os.path.join(output_dir, "partial_train.tfrecord")
partial_validate_set_path = os.path.join(output_dir, "partial_validate.tfrecord")


def create_tfrecord(input_dir, tfrecord_path, cases, mode):
    '''_CREATE_TFRECORD
    '''

    def normalize(volume):
        return (volume - np.min(volume)) / np.std(volume)

    print("Create TFRecord to " + mode)

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for case in tqdm(cases):
        case_path = os.path.join(input_dir, case[0])
        volumes_path = [os.path.join(case_path, p) for p in os.listdir(case_path)]
        for vp in volumes_path:
            if not os.path.isfile(vp):
                continue
            volume = np.load(vp)
            volume = normalize(volume)
            volume_raw = volume.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[case[1]])),
                "volume": tf.train.Feature(bytes_list=tf.train.BytesList(value=[volume_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()

    return


create_tfrecord(partial_input_dir, partial_train_set_path, partial_train_set, "train")
create_tfrecord(partial_input_dir, partial_validate_set_path, partial_validate_set, "validate")
