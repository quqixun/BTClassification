import os
import numpy as np
from tqdm import *
from random import seed, shuffle
import tensorflow as tf
import pandas as pd


def get_data_path(dir_path, label):
    subjects_name = os.listdir(dir_path)
    seed(SEED)
    shuffle(subjects_name)
    subjects = [[os.path.join(dir_path, n), label] for n in subjects_name]
    return subjects


def get_dataset(subjects):
    subj_num = len(subjects)
    train_idx = round(subj_num * TRAIN_PROP)
    valid_idx = round(subj_num * (TRAIN_PROP + VALID_PROP))

    train_set = subjects[:train_idx]
    valid_set = subjects[train_idx:valid_idx]
    test_set = subjects[valid_idx:]

    return train_set, valid_set, test_set


def write_tfrecord(subjects, tfrecord_path):

    def normalize(data):
        temp = np.copy(data)
        temp = (temp - np.mean(temp)) / np.std(temp)
        return temp.astype(np.float32)

    print("Create TFRecord to: " + tfrecord_path)
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    # For each case in list
    for subject in tqdm(subjects):
        # Generate paths for all data in one case
        subj_path = subject[0]
        subj_label = subject[1]
        subj_case_dir = [os.path.join(subj_path, n) for n in os.listdir(subj_path)]

        for scd in subj_case_dir:
            # Read, normalize and convert data to binary
            cor = normalize(np.load(os.path.join(scd, "cor.npy")))
            sag = normalize(np.load(os.path.join(scd, "sag.npy")))
            ax = normalize(np.load(os.path.join(scd, "ax.npy")))

            # Form an example of a data with its grade
            cor_raw = cor.tobytes()
            sag_raw = sag.tobytes()
            ax_raw = ax.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[subj_label])),
                "cor": tf.train.Feature(bytes_list=tf.train.BytesList(value=[cor_raw])),
                "sag": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sag_raw])),
                "ax": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ax_raw]))
            }))

            # Write the example into tfrecord file
            writer.write(example.SerializeToString())

    # Close writer
    writer.close()

    return


def save_to_csv(subjects, csv_path):
    subj_paths = [subj[0] for subj in subjects]
    subj_labels = [subj[1] for subj in subjects]

    df = pd.DataFrame(data={"subject": subj_paths, "label": subj_labels})
    df = df[["subject", "label"]]
    df.to_csv(csv_path, index=False)

    return


SEED = 7
TRAIN_PROP = 0.6
VALID_PROP = 0.2

parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data", "Original", "BraTS")
hgg_dir = os.path.join(data_dir, "HGGViewsVolume")
lgg_dir = os.path.join(data_dir, "LGGViewsVolume")

hgg_subjects = get_data_path(hgg_dir, 1)
lgg_subjects = get_data_path(lgg_dir, 0)

hgg_train, hgg_valid, hgg_test = get_dataset(hgg_subjects)
lgg_train, lgg_valid, lgg_test = get_dataset(lgg_subjects)

train = hgg_train + lgg_train
valid = hgg_valid + lgg_valid
test = hgg_test + lgg_test

# print(len(hgg_train), len(hgg_valid), len(hgg_test))
# print(len(lgg_train), len(lgg_valid), len(lgg_test))

tfrecords_dir = os.path.join(parent_dir, "data", "TFRecords", "MultiViews")
if not os.path.isdir(tfrecords_dir):
    os.makedirs(tfrecords_dir)

train_tfr_path = os.path.join(tfrecords_dir, "train.tfrecord")
valid_tfr_path = os.path.join(tfrecords_dir, "valid.tfrecord")

write_tfrecord(train, train_tfr_path)
write_tfrecord(valid, valid_tfr_path)

save_to_csv(train, os.path.join(tfrecords_dir, "train.csv"))
save_to_csv(valid, os.path.join(tfrecords_dir, "valid.csv"))
save_to_csv(test, os.path.join(tfrecords_dir, "test.csv"))
