# Brain Tumor Classification
# Load and Split dataset into training set,
# validation set and testing set.
# Author: Qixun QU
# Copyleft: MIT Licience

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


from __future__ import print_function


import os
import numpy as np
import pandas as pd
import nibabel as nib
from random import seed, shuffle
from keras.utils import to_categorical


class BTCDataset(object):

    def __init__(self,
                 hgg_dir, lgg_dir,
                 volume_type="t1ce",
                 train_prop=0.6,
                 valid_prop=0.2,
                 random_state=0,
                 is_augment=True,
                 pre_trainset_path=None,
                 pre_validset_path=None,
                 pre_testset_path=None,
                 data_format=".nii.gz"):
        '''__INIT__

            Intialize configurations for loading
            and partitioning dataset.

            Important variables:
            - train_x, train_y
            - valid_x, valid_y
            - test_x, test_y
            (x: brain images, y: labels)

            Inputs:
            -------

            - hgg_dir: string, path of directory contains HGG subjects.
            - lgg_dir: string, path of directory contains LGG subjects.
            - subj_separated: boolean, True: partition scans according to
                              subjects or False: randomly partition all scans.
                              Default is True.
            - volume_type: string, type of brain tissue, "t1ce", "flair",
                           "t1" or "t2". Default is "t1ce".
            - train_prop: float between 0 and 1, proportion of training
                          data to whole dataset. Default is 0.6.
            - valid_prop: float between 0 and 1, proportion of validation
                          data to whole dataset. Default is 0.2.
            - random_state: int, seed for reproducibly partition dataset.
            - is_augment: boolean, if True, do augmentation by flipping
                          image from left to right. Defalut is False.
            - pre_trainset_path, pre_validset_path, ore_testset_path:
              string, path of csv file, gives information of subjects (IDs
              and labels) in training set, validation set and testing set.
            - data_format: string, format of brain images, defalut is ".nii.gz".

        '''

        self.hgg_dir = hgg_dir
        self.lgg_dir = lgg_dir
        self.volume_type = volume_type

        self.train_prop = train_prop
        self.valid_prop = valid_prop
        self.random_state = int(random_state)
        self.is_augment = is_augment

        self.pre_trainset = pre_trainset_path
        self.pre_validset = pre_validset_path
        self.pre_testset = pre_testset_path
        self.data_format = data_format

        self.train_x, self.train_y = None, None
        self.valid_x, self.valid_y = None, None
        self.test_x, self.test_y = None, None

        return

    def run(self, pre_split=True,
            save_split=False,
            save_split_dir=None):
        '''RUN

            Partition dataset.

            Inputs:
            -------

            - pre_split: boolean, if True, read csv files to get information
                         of partitions that have been split. Default is True.
            - save_split: boolean, if True, save partition to csv files.
                          Default is False.
            - save_split_dir: string, path of directory to save partition
                              information. It is useful if save_split is True.
                              Default is None.

        '''

        print("\nSplitting dataset to train, valide and test.\n")

        # Load partition's information from csv file
        # or generate new partitions
        trainset, validset, testset = \
            self._get_pre_datasplit() if pre_split else \
            self._get_new_datasplit()

        # Load images acording to partition information
        self._load_dataset(trainset, validset, testset)

        if save_split and (not pre_split):
            # Save new partitions into csv files
            self.save_split_dir = save_split_dir
            self._save_dataset(trainset, validset, testset)

        return

    def _get_pre_datasplit(self):
        '''_GET_PRE_DATASPLIT

            Load partition inforamtion from csv files for
            training set, validation set and testing set.
            In each csv file, information includes:
            - ID: subject's ID.
            - label: subject's label, 1 for HGG and 0 for LGG.

            Outputs:
            --------

            - trainset, validset, testset: list of information,
              each element is [subject_path, label].

        '''

        # Parameters for function to load csv
        paras = {"hgg_dir": self.hgg_dir,
                 "lgg_dir": self.lgg_dir,
                 "data_format": self.data_format,
                 "csv_path": None}

        # Load partition of training set
        paras["csv_path"] = self.pre_trainset
        trainset = self.load_datasplit(**paras)

        # Load partition of validation set
        paras["csv_path"] = self.pre_validset
        validset = self.load_datasplit(**paras)

        # Load partition of testing set
        paras["csv_path"] = self.pre_testset
        testset = self.load_datasplit(**paras)

        return trainset, validset, testset

    def _get_new_datasplit(self):
        '''_GET_NEW_DATASPLIT

            Obtain new partition of dataset.
            -1- Generate paths of all subjects.
            -2- Randomly reoarrange the path list.
            -3- Partition dataset according to proportions.
            -4- Merge HGG and LGG subjects.

            Outputs:
            --------

            - trainset, validset, testset: list of information,
              each element is [subject_path, label].

        '''

        # Parameters for function to load subject's paths
        paras = {"label": None,
                 "dir_path": None,
                 "volume_type": self.volume_type,
                 "random_state": self.random_state}

        # Load HGG subjects' paths
        paras["label"], paras["dir_path"] = 1, self.hgg_dir
        hgg_subjects = self.get_subjects_path(**paras)

        # Load LGG subjects' paths
        paras["label"], paras["dir_path"] = 0, self.lgg_dir
        lgg_subjects = self.get_subjects_path(**paras)

        # Parameters for function to partition dataset
        paras = {"subjects": None,
                 "train_prop": self.train_prop,
                 "valid_prop": self.valid_prop}

        # Partition HGG subjects into three sets
        paras["subjects"] = hgg_subjects
        hgg_train, hgg_valid, hgg_test = self.split_dataset(**paras)

        # Partition LGG subjects into three sets
        paras["subjects"] = lgg_subjects
        lgg_train, lgg_valid, lgg_test = self.split_dataset(**paras)

        # Merge HGG and LGG subjects
        trainset = hgg_train + lgg_train
        validset = hgg_valid + lgg_valid
        testset = hgg_test + lgg_test

        return trainset, validset, testset

    def _load_dataset(self, trainset, validset, testset):
        '''_LOAD_DATASET

            Load images and labels for three partitions:
            training set, validation set and testing set.

        '''

        # Load images and labels of subjects in testing set
        self.test_x, test_y = self.load_data(testset, "test set")
        self.test_y = to_categorical(test_y, num_classes=2)

        # Load images and labels of subjects in validation set
        self.valid_x, valid_y = self.load_data(validset, "valid set")
        self.valid_y = to_categorical(valid_y, num_classes=2)

        # Load images and labels of subjects in training set
        train_x, train_y = self.load_data(trainset, "train set")

        if self.is_augment:
            # Augmentation on LGG subjects
            train_x, train_y = self.augment(train_x, train_y)

        self.train_x = train_x
        self.train_y = to_categorical(train_y, num_classes=2)

        return

    def _save_dataset(self, trainset, validset, testset):
        '''_SAVE_DATASET

            Save partition informatio into csv files.

            Outputs:
            --------

            - trainset_[random_state].csv
            - validset_[random_state].csv
            - testset_[random_state].csv

        '''

        # Generate paths for output csv files
        ap = str(self.random_state) + ".csv"
        trainset_path = os.path.join(self.save_split_dir, "trainset_" + ap)
        validset_path = os.path.join(self.save_split_dir, "validset_" + ap)
        testset_path = os.path.join(self.save_split_dir, "testset_" + ap)

        # Save information into csv files
        self.save_datasplit(trainset, trainset_path)
        self.save_datasplit(validset, validset_path)
        self.save_datasplit(testset, testset_path)

        return

    @staticmethod
    def load_datasplit(hgg_dir, lgg_dir, csv_path,
                       data_format=".nii.gz"):
        '''LOAD_DATASPLIT

            Load partition information from given csv file.

            Inputs:
            -------

            - hgg_dir: string, directory path of HGG subjects.
            - lgg_dir: string, directory path of LGG subjects.
            - csv_path: string, path of csv file which contains
                        partition information.
            - data_format: string, dormat of input images,
                           default is ".nii.gz".

            Output:
            -------

            - info: list of partition information, each element is
                    [subject_path, label].

        '''

        # Load IDs and labels form csv file
        df = pd.read_csv(csv_path)
        IDs = df["ID"].values.tolist()
        labels = df["label"].values.tolist()

        info = []
        for ID, label in zip(IDs, labels):
            # Generate directopy path of each subject
            target_dir = hgg_dir if label else lgg_dir
            path = os.path.join(target_dir, ID[:-5],
                                ID + data_format)
            info.append([path, label])
        return info

    @staticmethod
    def save_datasplit(dataset, to_path):
        '''SAVE_DATASPLIT

            Save partition information into csv file.

            Inputs:
            -------

            - dataset: list, information of partition, each element
                       is [subject_path, label].
            - to_path: string, the path of csv file to be saved.

            Output:
            -------

            - A csv table with two columns, "ID" and "label".

        '''

        IDs, labels = [], []
        for i in dataset:
            # Extract ID from subject's path
            IDs.append(i[0].split("/")[-1].split(".")[0])
            # Extract label
            labels.append(i[1])

        # Create pandas DataFrame and save it into csv file
        df = pd.DataFrame(data={"ID": IDs, "label": labels})
        df.to_csv(to_path, index=False)

        return

    @staticmethod
    def get_subjects_path(dir_path, volume_type, label,
                          random_state=0):
        '''GET_SUBJECTS_PATH

            Obtain subjects' paths of HGG or LGG.

            Inputs:
            -------

            - dir_path: string, directory path of HGG or LGG subjects.
            - volume_type: string, type of brain tissue, "t1ce", "flair",
                           "t1" or "t2".
            - label: int, 1 for HGG and o for LGG.
            - random_state: int, seed for shuffle paths list.

            Output:
            -------

            - subjects_paths: list with two columns, each element is
                              [subject_path, label].

        '''

        # Obtain all subjects' names
        subjects = os.listdir(dir_path)

        # Set seed and shuffle list
        # Different seed leads to different shuffled list
        # to change subjects in partitions
        seed(random_state)
        shuffle(subjects)

        subjects_paths = []
        for subject in subjects:
            subject_dir = os.path.join(dir_path, subject)
            for scan_name in os.listdir(subject_dir):
                if volume_type not in scan_name:
                    # Not target volume
                    continue

                # Element [subject_dir, label]
                scan_path = os.path.join(subject_dir, scan_name)
                subjects_paths.append([scan_path, label])

        return subjects_paths

    @staticmethod
    def split_dataset(subjects, train_prop=0.6, valid_prop=0.2):
        '''SPLIT_DATASET

            Partition dataset into three parts according
            to proportions.

            Inputs:
            -------

            - subjects: list with two columns, information of all
                        subjects, each element is [subject_path, label].
            - train_prop: float between 0 and 1, proportion of training
                          data to whole dataset. Default is 0.6.
            - valid_prop: float between 0 and 1, proportion of validation
                          data to whole dataset. Default is 0.2.

            Outputs:

            - trainset, validset, testset: partition information,including
              subjects' paths and labels.

        '''

        subj_num = len(subjects)

        # Extract subjects for testing set
        train_valid_num = subj_num * (train_prop + valid_prop)
        train_valid_idx = int(round(train_valid_num))
        testset = subjects[train_valid_idx:]

        # Extract subjects validation set
        valid_idx = int(round(subj_num * valid_prop))
        validset = subjects[:valid_idx]
        # Extract subjects for training set
        trainset = subjects[valid_idx:train_valid_idx]

        return trainset, validset, testset

    @staticmethod
    def load_data(dataset, mode):
        '''LOAD_DATA

            Load images from partition information.

            Inputs:
            -------

            - dataset: list with two columns, [subject_path, label].
            - mode: string, indicates which partition, "train set",
                    "valid set" or "test set".

            Outputs:
            --------

            - x: numpy ndarray in shape [n, 112, 96, 96, 1], n is the
                 number of scans in one partition. Input images.
            - y: numpy ndarray in shape [n, 1]. Labels of subjects.

        '''

        x, y = [], []
        print("Loading {} data ...".format(mode))
        for subject in dataset:
            volume_path, label = subject[0], subject[1]
            # Load image and rotate it to standard space
            volume = nib.load(volume_path).get_data()
            volume = np.transpose(volume, axes=[1, 0, 2])
            volume = np.flipud(volume)

            # Extract mean and std from brain object
            volume_obj = volume[volume > 0]
            obj_mean = np.mean(volume_obj)
            obj_std = np.std(volume_obj)
            # Normalize whole image
            volume = (volume - obj_mean) / obj_std

            volume = np.expand_dims(volume, axis=3)
            x.append(volume.astype(np.float32))
            y.append(label)

        x = np.array(x)
        y = np.array(y).reshape((-1, 1))

        return x, y

    @staticmethod
    def augment(train_x, train_y):
        '''AUGMENT

              Do augmentation of LGG subjects in training set
              by flipping each image from left to right.

              Inputs:
              -------

              - train_x: numpy ndarray, images array of training set.
              - train_y: numpy ndarray, labels of training set.

              Outputs:
              --------

              - train_x: augmented training images, which are double as original.
              - train_y: augmented labels of training set.

        '''

        print("Do Augmentation on LGG Samples ...")
        train_x_aug, train_y_aug = [], []
        for i in range(len(train_y)):
            train_x_aug.append(train_x[i])
            train_y_aug.append(train_y[i])
            if train_y[i] == 0:
                # Flip image if it is LGG
                train_x_aug.append(np.fliplr(train_x[i]))
                train_y_aug.append(np.array([0]))
        train_x = np.array(train_x_aug)
        train_y = np.array(train_y_aug).reshape((-1, 1))

        return train_x, train_y


if __name__ == "__main__":

    import gc

    parent_dir = os.path.dirname(os.getcwd())

    # Set dirctory for input images (separated subjects)
    data_dir = os.path.join(parent_dir, "data")
    hgg_dir = os.path.join(data_dir, "HGGSegTrimmed")
    lgg_dir = os.path.join(data_dir, "LGGSegTrimmed")

    # Test 1
    # Load and split dataset
    data = BTCDataset(hgg_dir, lgg_dir,
                      volume_type="t1ce",
                      train_prop=0.6,
                      valid_prop=0.2,
                      random_state=0)
    data.run(pre_split=False,
             save_split=True,
             save_split_dir="DataSplit")
    print(data.train_x.shape, data.train_y.shape)
    del data
    gc.collect()

    # Test 2
    # Load dataset which has been split
    data = BTCDataset(hgg_dir, lgg_dir,
                      volume_type="t1ce",
                      pre_trainset_path="DataSplit/trainset.csv",
                      pre_validset_path="DataSplit/validset.csv",
                      pre_testset_path="DataSplit/testset.csv")
    data.run(pre_split=True)
    print(data.train_x.shape, data.train_y.shape)
    del data
    gc.collect()
