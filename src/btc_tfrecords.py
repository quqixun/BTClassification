# Brain Tumor Classification
# Script for Creating TFRecords
# Author: Qixun Qu
# Create on: 2017/10/09
# Modify on: 2017/10/10

#     ,,,         ,,,
#   ;"   ';     ;'   ",
#   ;  @.ss$$$$$$s.@  ;
#   `s$$$$$$$$$$$$$$$'
#   $$$$$$$$$$$$$$$$$$
#  $$$$P""Y$$$Y""W$$$$$
#  $$$$  p"$$$"q  $$$$$
#  $$$$  .$$$$$.  $$$$'
#   $$$DaU$$O$$DaU$$$'
#    '$$$$'.^.'$$$$'
#       '&$$$$$&'


import os
import numpy as np
from tqdm import *
import pandas as pd
import tensorflow as tf
from btc_settings import *


class BTCTFRecords():

    def __init__(self):
        '''__INIT__
        '''

        return

    def create_tfrecord(self, input_dir, output_dir, temp_dir, label_file):
        '''CREATE_TFRECORD
        '''

        # Check whether the input folder is exist
        if not os.path.isdir(input_dir):
            print("Input directory is not exist.")
            raise

        # Check whether the label file is exist
        if not os.path.isfile(label_file):
            print("The label file is not exist.")
            raise

        # Create folder to save outputs
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.train_tfrecord = os.path.join(output_dir, TFRECORD_TRAIN)
        self.validate_tfrecord = os.path.join(output_dir, TFRECORD_VALIDATE)

        # Obtain serial numbers of cases
        self.case_no = os.listdir(input_dir)

        # Read labels of all cases from label file
        self.labels = pd.read_csv(label_file)

        # TFRecords creation pipline
        self._check_case_no()
        self._create_temp_files(temp_dir)
        train_set, validate_set = self._generate_cases_set()
        self._write_tfrecord(input_dir, self.train_tfrecord, train_set, "train")
        self._write_tfrecord(input_dir, self.validate_tfrecord, validate_set, "validate")

        return

    def _check_case_no(self):
        '''_CHECK_CASE_NO

            If cases cannot be found in label file, the
            process will be stopped.

        '''

        # Put unfound cases into list
        not_found_cases = []
        all_cases_no = self.labels[CASE_NO].values.tolist()
        for cv in self.case_no:
            if cv not in all_cases_no:
                not_found_cases.append(cv)

        # If the list is not empty, quit program
        if len(not_found_cases) != 0:
            print("Cannot find case in label file.")
            raise

        return

    def _create_temp_files(self, temp_dir):
        '''_CREATE_TEMP_FILES
        '''

        def create_txt_file(path):
            if os.path.isfile(path):
                os.remove(path)
            open(path, "a").close()

        # Create folder to save temporary files
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)

        self.train_set_file = os.path.join(temp_dir, TRAIN_SET_FILE)
        self.validate_set_file = os.path.join(temp_dir, VALIDATE_SET_FILE)

        create_txt_file(self.train_set_file)
        create_txt_file(self.validate_set_file)

        return

    def _generate_cases_set(self):
        '''_GENERATE_CASES_SET
        '''

        def save_cases_names(file, cases):
            txt = open(file, "a")
            for case in cases:
                txt.write(case[0] + CASES_FILE_SPLIT)

        train_set, validate_set = [], []
        for grade in GRADES_LIST:
            cases = self.labels[CASE_NO][self.labels[GRADE_LABEL] == grade].values.tolist()
            cases_num = len(cases)
            train_num = int(cases_num * PROPORTION)
            index = list(np.arange(cases_num))
            np.random.seed(RANDOM_SEED)
            train_index = list(np.random.choice(cases_num, train_num, replace=False))
            validate_index = [i for i in index if i not in train_index]
            train_set += [[cases[i], grade] for i in train_index]
            validate_set += [[cases[i], grade] for i in validate_index]

        save_cases_names(self.train_set_file, train_set)
        save_cases_names(self.validate_set_file, validate_set)

        return train_set, validate_set

    def _write_tfrecord(self, input_dir, tfrecord_path, cases, mode):
        '''_WRITE_TFRECORD
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

    def decode_tfrecord(self, path, batch_size, num_epoches, patch_shape,
                        min_after_dequeue, capacity):
        if not num_epoches:
            num_epoches = None

        with tf.name_scope("input"):
            queue = tf.train.string_input_producer([path], num_epochs=num_epoches)
            volume, label = self._decode_example(queue, patch_shape)

            volumes, labels = tf.train.shuffle_batch([volume, label],
                                                     batch_size=batch_size,
                                                     num_threads=4,
                                                     capacity=capacity,
                                                     min_after_dequeue=min_after_dequeue)
        return volumes, labels

    def _decode_example(self, queue, patch_shape):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                "label": tf.FixedLenFeature([], tf.int64),
                "volume": tf.FixedLenFeature([], tf.string)
            })

        volume = tf.decode_raw(features["volume"], tf.float32)
        volume = tf.reshape(volume, patch_shape)
        label = tf.cast(features["label"], tf.uint8)

        return volume, label


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())

    input_dir = os.path.join(parent_dir, DATA_FOLDER, AUGMENT_FOLDER)
    output_dir = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER)
    temp_dir = os.path.join(TEMP_FOLDER, TFRECORDS_FOLDER)
    label_file = os.path.join(parent_dir, DATA_FOLDER, LABEL_FILE)

    tfr = BTCTFRecords()
    tfr.create_tfrecord(input_dir, output_dir, temp_dir, label_file)
