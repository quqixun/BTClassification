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
                 save_split=False,
                 save_dir=None,
                 pre_split=False,
                 pre_trainset_path=None,
                 pre_validset_path=None,
                 pre_testset_path=None,
                 data_format=".nii.gz"):
        '''__INIT__
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

        trainset, validset, testset = \
            self._get_pre_datasplit() if pre_split else \
            self._get_new_datasplit()

        self._load_dataset(trainset, validset, testset)

        if save_split and (not pre_split):
            self.save_dir = save_dir
            self._save_dataset(trainset, validset, testset)

        return

    def _get_pre_datasplit(self):
        paras = {"hgg_dir": self.hgg_dir,
                 "lgg_dir": self.lgg_dir,
                 "data_format": self.data_format,
                 "csv_path": None}

        paras["csv_path"] = self.pre_trainset
        trainset = self.load_datasplit(**paras)

        paras["csv_path"] = self.pre_validset
        validset = self.load_datasplit(**paras)

        paras["csv_path"] = self.pre_testset
        testset = self.load_datasplit(**paras)

        return trainset, validset, testset

    def _get_new_datasplit(self):
        paras = {"label": None,
                 "dir_path": None,
                 "volume_type": self.volume_type,
                 "random_state": self.random_state}

        paras["label"], paras["dir_path"] = 1, self.hgg_dir
        hgg_subjects = self.get_subjects_path(**paras)

        paras["label"], paras["dir_path"] = 0, self.lgg_dir
        lgg_subjects = self.get_subjects_path(**paras)

        paras = {"subjects": None,
                 "train_prop": self.train_prop,
                 "valid_prop": self.valid_prop}

        paras["subjects"] = hgg_subjects
        hgg_train, hgg_valid, hgg_test = self.split_dataset(**paras)

        paras["subjects"] = lgg_subjects
        lgg_train, lgg_valid, lgg_test = self.split_dataset(**paras)

        trainset = hgg_train + lgg_train
        validset = hgg_valid + lgg_valid
        testset = hgg_test + lgg_test

        return trainset, validset, testset

    def _load_dataset(self, trainset, validset, testset):

        self.test_x, test_y = self.load_data(testset, "testset")
        self.test_y = to_categorical(test_y, num_classes=2)

        self.valid_x, valid_y = self.load_data(validset, "validset")
        self.valid_y = to_categorical(valid_y, num_classes=2)

        train_x, train_y = self.load_data(trainset, "trainset")
        if self.is_augment:
            train_x, train_y = self.augment(train_x, train_y)
        self.train_x = train_x
        self.train_y = to_categorical(train_y, num_classes=2)

        return

    def _save_dataset(self, trainset, validset, testset):
        ap = str(self.random_state) + ".csv"
        trainset_path = os.path.join(self.save_dir, "trainset_" + ap)
        validset_path = os.path.join(self.save_dir, "validset_" + ap)
        testset_path = os.path.join(self.save_dir, "testset_" + ap)

        self.save_datasplit(trainset, trainset_path)
        self.save_datasplit(validset, validset_path)
        self.save_datasplit(testset, testset_path)

        return

    @staticmethod
    def load_datasplit(hgg_dir, lgg_dir, csv_path,
                       data_format=".nii.gz"):
        '''LOAD_DATASPLIT
        '''
        df = pd.read_csv(csv_path)
        IDs = df["ID"].values.tolist()
        labels = df["label"].values.tolist()
        info = []
        for ID, label in zip(IDs, labels):
            target_dir = hgg_dir if label else lgg_dir
            path = os.path.join(target_dir, ID[:-5],
                                ID + data_format)
            info.append([path, label])
        return info

    @staticmethod
    def save_datasplit(dataset, to_path):
        IDs, labels = [], []
        for i in dataset:
            IDs.append(i[0].split("/")[-1].split(".")[0])
            labels.append(i[1])

        df = pd.DataFrame(data={"ID": IDs, "label": labels})
        df.to_csv(to_path, index=False)
        return

    @staticmethod
    def get_subjects_path(dir_path, volume_type, label,
                          random_state=0):
        subjects = os.listdir(dir_path)
        seed(random_state)
        shuffle(subjects)
        subjects_paths = []
        for subject in subjects:
            subject_dir = os.path.join(dir_path, subject)
            for scan_name in os.listdir(subject_dir):
                if volume_type not in scan_name:
                    continue
                scan_path = os.path.join(subject_dir, scan_name)
                subjects_paths.append([scan_path, label])
        return subjects_paths

    @staticmethod
    def split_dataset(subjects, train_prop=0.6, valid_prop=0.2):
        subj_num = len(subjects)
        train_valid_num = subj_num * (train_prop + valid_prop)
        train_valid_idx = int(round(train_valid_num))
        testset = subjects[train_valid_idx:]

        valid_idx = int(round(subj_num * valid_prop))
        validset = subjects[:valid_idx]
        trainset = subjects[valid_idx:train_valid_idx]
        return trainset, validset, testset

    @staticmethod
    def load_data(dataset, mode):
        x, y = [], []
        print("Loading {} data ...".format(mode))
        for subject in dataset:
            volume_path, label = subject[0], subject[1]
            volume = nib.load(volume_path).get_data()
            volume = np.transpose(volume, axes=[1, 0, 2])
            volume = np.flipud(volume)

            volume_obj = volume[volume > 0]
            obj_mean = np.mean(volume_obj)
            obj_std = np.std(volume_obj)
            volume = (volume - obj_mean) / obj_std

            volume = np.expand_dims(volume, axis=3)
            x.append(volume.astype(np.float32))
            y.append(label)

        x = np.array(x)
        y = np.array(y).reshape((-1, 1))

        return x, y

    @staticmethod
    def augment(train_x, train_y):
        print("Do Augmentation on LGG Samples ...")
        train_x_aug, train_y_aug = [], []
        for i in range(len(train_y)):
            train_x_aug.append(train_x[i])
            train_y_aug.append(train_y[i])
            if train_y[i] == 0:
                train_x_aug.append(np.fliplr(train_x[i]))
                train_y_aug.append(np.array([0]))
        train_x = np.array(train_x_aug)
        train_y = np.array(train_y_aug).reshape((-1, 1))

        return train_x, train_y


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data", "BraTS")
    hgg_dir = os.path.join(data_dir, "HGGSegTrimmed")
    lgg_dir = os.path.join(data_dir, "LGGSegTrimmed")

    # Load and split dataset
    data = BTCDataset(hgg_dir, lgg_dir,
                      volume_type="t1ce",
                      train_prop=0.6,
                      valid_prop=0.2,
                      random_state=0,
                      save_split=True,
                      save_dir="DataSplit")

    # Load dataset which has been splitted
    data = BTCDataset(hgg_dir, lgg_dir,
                      volume_type="t1ce",
                      pre_split=True,
                      pre_trainset_path="DataSplit/trainset.csv",
                      pre_validset_path="DataSplit/validset.csv",
                      pre_testset_path="DataSplit/testset.csv")
