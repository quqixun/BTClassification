# Brain Tumor Classification
# Script for Creating and Loading TFRecords
# Author: Qixun Qu
# Create on: 2017/10/09
# Modify on: 2017/11/28

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

'''

Class BTCTFRecords

-1- Write all examples with their labels into two tfrecord
    files respectively.
    (1) Check whether all cases can be found in labels file;
    (2) Create three empty temporary filts, which are:
        a text file to save cases' names of dataset1;
        a text file to save cases' names of dataset2;
        a jason file to save the number of data of
        dataset1 and dataset2;
    (3) Generate cases' names of two datasets
        respectively according to the label file;
    (4) Extract relevant data to write TFRecords.

-2- Load batches and labels for training and validating from
    tfrecords.

'''


from __future__ import print_function

import os
import json
import argparse
import numpy as np
from tqdm import *
import pandas as pd
import tensorflow as tf
from btc_settings import *


class BTCTFRecords():

    def __init__(self, data_mode=None):
        '''__INIT__

            Initialization.

            Input:
            ------
            - data_mode: string, "patch", "volume" or "slice"

            Usage examples:
            ---------------
            - Create instance first:
              tfr = BTCTFRecords()
            - Create TFRecords:
              tfr.create_tfrecord(paras, ...)
            - Load TFRecords:
              outputs = tfr.decode_tfrecord(paras, ...)

        '''

        # Leads to different normalization methods
        self.data_mode = data_mode

        return

    def create_tfrecord(self, input_dir, output_dir, temp_dir, label_file):
        '''CREATE_TFRECORD

            Initialize variables to create tfrecoeds and carry out
            pipline to write tfrecords for two datasets.

            Inputs:
            -------
            - input_dir: the path of directory where patches are saved in
            - output_dir: the path of directory to write tfrecord files
            - temp_dir: the path of directory to save temporary files
            - label_file: the path of label file

        '''

        # Check whether the input folder is exist
        if not os.path.isdir(input_dir):
            raise IOError("Input directory is not exist.")

        # Check whether the label file is exist
        if not os.path.isfile(label_file):
            raise IOError("The label file is not exist.")

        # Create folder to save outputs
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # Generate paths for tfrecords
        self.dataset1_tfrecord = os.path.join(output_dir, TFRECORD1)
        self.dataset2_tfrecord = os.path.join(output_dir, TFRECORD2)

        # Read labels of all cases from label file
        self.labels = pd.read_csv(label_file)

        # Initialize an empty dictionary to keep amount of
        # patches in two datasets
        self.data_num = {}

        # TFRecords creation pipline
        self._check_case_no(input_dir)
        self._create_temp_files(temp_dir)
        dataset1, dataset2 = self._generate_cases_set()
        self._write_tfrecord(input_dir, self.dataset1_tfrecord, dataset1, TFRECORD_MODE1)
        self._write_tfrecord(input_dir, self.dataset2_tfrecord, dataset2, TFRECORD_MODE2)

        # Save dictionary into json file
        with open(self.data_num_file, "w") as json_file:
            json.dump(self.data_num, json_file)

        return

    def _check_case_no(self, input_dir):
        '''_CHECK_CASE_NO

            If cases cannot be found in label file, the
            process will be stopped.

        '''

        # Obtain serial numbers of cases
        case_no = os.listdir(input_dir)

        # Put unfound cases into list
        not_found_cases = []
        all_cases_no = self.labels[CASE_NO].values.tolist()
        for cv in case_no:
            if cv not in all_cases_no:
                not_found_cases.append(cv)

        # If the list is not empty, quit program
        if len(not_found_cases) != 0:
            print("Cannot find case in label file.")
            raise

        return

    def _create_temp_files(self, temp_dir):
        '''_CREATE_TEMP_FILES

            Create three empty temporary files in given
            temporary folder.

            Input:
            ------
            - temp_dir: the path of directory where temporary
                        files will be saved in

        '''

        # Function to create empty files
        def create_temp_file(path):
            if os.path.isfile(path):
                os.remove(path)
            open(path, "w").close()

        # Create folder to save temporary files
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)

        # A text file to save cases' names of dataset 1
        self.dataset1_file = os.path.join(temp_dir, DATASET1_FILE)
        create_temp_file(self.dataset1_file)

        # A text file to save cases' names of dataset 2
        self.dataset2_file = os.path.join(temp_dir, DATASET2_FILE)
        create_temp_file(self.dataset2_file)

        # A jason file to save the number of data of two datasets
        self.data_num_file = os.path.join(temp_dir, DATA_NUM_FILE)
        create_temp_file(self.data_num_file)

        return

    def _generate_cases_set(self):
        '''_GENERATE_CASES_SET

            Generate cases' names for two datasets
            according to the label file.

            For each grade group:
            - Compute the number of cases in two datasets.
              Number of cases in dataset1 = all cases * PROPORTION
              Number of cases in dataset2 = all cases - cases in dataset1
              PROPORTION can be found in btc_settings.py.
            - Randomle select cases with respect to dataset1 and dataset2.
            - Put cases' names with their grades into list.

            outputs:
            --------
            - dataset1: cases' names and their grades in dataset1
            - dataset2: cases' names and their grades in dataset2

        '''

        # Function to write cases list into file
        def save_cases_names(file, cases):
            txt = open(file, "a")
            for case in cases:
                txt.write(case[0] + CASES_FILE_SPLIT)

        # Initialize empty list for two datasets
        dataset1, dataset2 = [], []

        # Generates set for each grade group
        for grade in GRADES_LIST:
            # Get all cases' names of a certain grade
            cases = self.labels[CASE_NO][self.labels[GRADE_LABEL] == grade].values.tolist()

            # Compute number of cases in dataset1
            cases_num = len(cases)
            dataset1_num = int(cases_num * PROPORTION)

            # Randomly get cases' names for dataset1
            index = list(np.arange(cases_num))
            np.random.seed(RANDOM_SEED)
            dataset1_index = list(np.random.choice(cases_num, dataset1_num, replace=False))

            # Other cases are regarded as validating set
            dataset2_index = [i for i in index if i not in dataset1_index]

            # Put cases' names with grades into list
            dataset1 += [[cases[i], grade] for i in dataset1_index]
            dataset2 += [[cases[i], grade] for i in dataset2_index]

        # Save list into file
        save_cases_names(self.dataset1_file, dataset1)
        save_cases_names(self.dataset2_file, dataset2)

        return dataset1, dataset2

    def _write_tfrecord(self, input_dir, tfrecord_path, cases, mode):
        '''_WRITE_TFRECORD

            Write data into tfrecord file.
            For each case in two datasets:
                For each data in a certain case:
                    Mormalize the data
                    Write the data into tfrecord file

            Inputs:
            -------
            - input_dir: the path of directory where keeps all data
            - tfrecord_path: the path to save tfrecord file
            - cases: a list consists of cases' names, such as:
                     [["case1", grade_of_case1], ["case2", grade_of_case2]]

        '''

        # normalize the data
        def normalize(data):
            temp = np.copy(data)
            for c in range(CHANNELS):
                channel = data[..., c]
                if self.data_mode == "patch":
                    temp[..., c] = (channel - np.mean(channel)) / np.std(channel)
                else:  # self.data_mode is "volume" or "slice"
                    if np.max(channel) == 0:
                        return None
                    temp[..., c] = channel / np.max(channel)
            return temp.astype(np.float32)

        print("Create TFRecord of " + mode)

        # Variable to count number in dataset1 or dataset2
        data_num = 0

        # Create writer
        writer = tf.python_io.TFRecordWriter(tfrecord_path)

        grade2_num = 0
        grade3_num = 0
        grade4_num = 0

        # For each case in list
        for case in tqdm(cases):
            # Generate paths for all data in one case
            case_path = os.path.join(input_dir, case[0])
            data_path = [os.path.join(case_path, p) for p in os.listdir(case_path)]

            for dp in data_path:

                # If the data can not be found, skip to next iteration
                if not os.path.isfile(dp):
                    continue

                # Read, normalize and convert data to binary
                data = np.load(dp)
                data = normalize(data)

                # Use one channel of data
                # data = data[..., 0]
                # print(data.shape)

                if data is None:
                    continue

                # Form an example of a data with its grade
                data_raw = data.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[case[1]])),
                    "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))
                }))

                # Write the example into tfrecord file
                writer.write(example.SerializeToString())

                # Count
                data_num += 1

                if case[1] == GRADE_II:
                    grade2_num += 1
                elif case[1] == GRADE_III:
                    grade3_num += 1
                elif case[1] == GRADE_IV:
                    grade4_num += 1

        # Close writer
        writer.close()

        print("Grade 2: {0}, Grade 3: {1}, Grade 4: {2}".format(grade2_num, grade3_num, grade4_num))

        # Save number of data into dictionary
        # {mode1: xxxx, mode2: xxxx}
        self.data_num[mode] = data_num

        return

    def decode_tfrecord(self, path, batch_size,
                        num_epoches, patch_shape,
                        capacity, min_after_dequeue):
        '''DECODE_TFRECORD

            Decode batches from tfrecords according to given settings.
            Global settings can be found in btc_settings.py.

            Inputs:
            -------
            - path: the path of tfrecord file
            - batch_size: the number of data in one batch
            - num_epoches: the number of training epoches
            - patch_shape: the shape of each data
            - capacity: the maximum number of elements in the queue
            - min_after_dequeue: minimum number elements in the queue after
                                 a dequeue, used to ensure a level of mixing
                                 of elements

            Outputs:
            --------
            - data: shuffled data batch for dataset1 or dataset2
            - labels: grade labels for data

        '''

        # Set default value of number of epoches
        if not num_epoches:
            num_epoches = None

        with tf.name_scope("input"):
            # Generate queue and load example
            queue = tf.train.string_input_producer([path], num_epochs=num_epoches)
            data, label = self._decode_example(queue, patch_shape)

            # Shuffle data
            datas, labels = tf.train.shuffle_batch([data, label],
                                                   batch_size=batch_size,
                                                   num_threads=NUM_THREADS,
                                                   capacity=capacity,
                                                   min_after_dequeue=min_after_dequeue)
        return datas, labels

    def _decode_example(self, queue, patch_shape):
        '''_DECORD_EXAMPLE

            Decode one example from tfrecord file,
            including its dat and label.

            Inputs:
            -------
            - queue: a queue of input filenames
            - patch_shape: shape of one data

            Outputs:
            --------
            - data: one data in given shape
            - label: the grade of the data

        '''

        # Load features for one example, in this case,
        # they are data and its label
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                "label": tf.FixedLenFeature([], tf.int64),
                "data": tf.FixedLenFeature([], tf.string)
            })

        # Load and reshape data
        data = tf.decode_raw(features["data"], tf.float32)
        data = tf.reshape(data, patch_shape)

        # Extract its label
        label = features["label"]

        return data, label


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    help_str = "Select a data in 'patch', 'volume' or 'slice'."
    parser.add_argument("--data", action="store", dest="data", help=help_str)
    args = parser.parse_args()

    parent_dir = os.path.dirname(os.getcwd())
    label_file = os.path.join(parent_dir, DATA_FOLDER, LABEL_FILE)

    if args.data == "patch":
        input_dir = os.path.join(parent_dir, DATA_FOLDER, AUGMENT_FOLDER)
        output_dir = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER, PATCHES_FOLDER)
        temp_dir = os.path.join(TEMP_FOLDER, TFRECORDS_FOLDER, PATCHES_FOLDER)
    elif args.data == "volume":
        input_dir = os.path.join(parent_dir, DATA_FOLDER, VOLUMES_FOLDER)
        output_dir = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER, VOLUMES_FOLDER)
        temp_dir = os.path.join(TEMP_FOLDER, TFRECORDS_FOLDER, VOLUMES_FOLDER)
    elif args.data == "slice":
        input_dir = os.path.join(parent_dir, DATA_FOLDER, SLICES_FOLDER)
        output_dir = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER, SLICES_FOLDER)
        temp_dir = os.path.join(TEMP_FOLDER, TFRECORDS_FOLDER, SLICES_FOLDER)
    else:
        raise ValueError("Cannot find data in 'patch', 'volume' or 'slice'.")

    tfr = BTCTFRecords(args.data)
    tfr.create_tfrecord(input_dir, output_dir, temp_dir, label_file)
