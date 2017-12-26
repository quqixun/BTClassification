import os
import shutil
import numpy as np
from tqdm import *
from models import *
import pandas as pd
import nibabel as nib
import tensorflow as tf
from random import seed, shuffle

from keras.layers import *
from keras.callbacks import CSVLogger
from keras.optimizers import SGD, Adam, Adagrad
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (ModelCheckpoint,
                             LearningRateScheduler,
                             ReduceLROnPlateau,
                             TensorBoard)


SEED = 77
TRAIN_PROP = 0.8
TEST_PROP = 0.2
VOLUME_SIZE = [112, 112, 112, 1]

BATCH_SIZE = 8
EPOCHS_NUM = 50
SPLITS_NUM = 4


def get_data_path(dir_path, volume_type, label):
    subjects = os.listdir(dir_path)
    seed(SEED)
    shuffle(subjects)
    subjects_paths = []
    for subject in subjects:
        subject_dir = os.path.join(dir_path, subject)
        for scan_name in os.listdir(subject_dir):
            if volume_type in scan_name:
                subjects_paths.append([os.path.join(subject_dir, scan_name), label])
    return subjects_paths


def get_dataset(subjects):
    subj_num = len(subjects)
    train_idx = round(subj_num * TRAIN_PROP)

    train_set = subjects[:train_idx]
    test_set = subjects[train_idx:]

    return train_set, test_set


def save_to_csv(subjects, csv_path):
    subj_paths = [subj[0] for subj in subjects]
    subj_labels = [subj[1] for subj in subjects]

    df = pd.DataFrame(data={"subject": subj_paths, "label": subj_labels})
    df = df[["subject", "label"]]
    df.to_csv(csv_path, index=False)

    return


def load_data(info):
    x, y = [], []
    print("Loading data ...")
    for subject in tqdm(info):
        volume_path, label = subject[0], subject[1]
        volume = nib.load(volume_path).get_data()
        volume = np.rot90(volume, 3)
        volume_obj = volume[volume > 0]
        volume = (volume - np.mean(volume_obj)) / np.std(volume_obj)
        # volume = volume / np.max(volume_obj) - 0.5
        volume = np.reshape(volume, VOLUME_SIZE)
        x.append(volume.astype(np.float32))
        y.append(label)

    x = np.array(x)
    y = np.array(y).reshape((-1, 1))

    return x, y


def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return


def lr_schedule(epoch):

    lrs = [1e-4] * 20 + [5e-5] * 20 + [1e-5] * 30 + [5e-6] * 30
    lr = lrs[epoch]
    print("Learning rate: ", lr)

    return lr


def train(trainset_info, model_type, models_dir,
          logs_dir, optimizer, model_name, augment=False):

    x, y = load_data(trainset_info)
    kfold = StratifiedKFold(n_splits=SPLITS_NUM, shuffle=True)
    kfold_no = 0

    for tidx, vidx in kfold.split(x, y):
        print("KFold: ", kfold_no)

        x_train, y_train = [], []
        for idx in tidx:
            x_train.append(x[idx])
            y_train.append(y[idx])

            if y[idx] == 0:
                aug_volume = np.fliplr(x[idx])
                x_train.append(aug_volume.astype(np.float32))
                y_train.append(y[idx])

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        y_train = to_categorical(y_train, num_classes=2)
        x_valid = x[vidx]
        y_valid = to_categorical(y[vidx], num_classes=2)

        if model_type == "vggish":
            model = vggish()
        elif model_type == "pyramid":
            model = pyramid()

        if optimizer == "adam":
            opt = Adam(lr=lr_schedule(0))
        elif optimizer == "adagrade":
            opt = Adagrad(lr=lr_schedule(0))
        elif optimizer == "sgd":
            opt = SGD(lr=lr_schedule(0))
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])

        model.summary()
        # model_name = model_type + "_" + optimizer
        print(model_name)

        model_dir = os.path.join(models_dir, model_name, "kfold" + str(kfold_no))
        create_dir(model_dir)

        log_dir = os.path.join(logs_dir, model_name, "kfold" + str(kfold_no))
        create_dir(log_dir)

        best_model_path = os.path.join(model_dir, "best.h5")
        last_model_path = os.path.join(model_dir, "last.h5")
        logs_path = os.path.join(model_dir, "learning_curv.csv")
        csv_logger = CSVLogger(logs_path, append=True, separator=';')

        checkpoint = ModelCheckpoint(filepath=best_model_path,
                                     monitor="val_loss",
                                     verbose=1,
                                     save_best_only=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
        #                                cooldown=0,
        #                                patience=5,
        #                                min_lr=1e-6)
        tb = TensorBoard(log_dir=log_dir, batch_size=BATCH_SIZE)
        callbacks = [checkpoint, lr_scheduler, csv_logger, tb]

        class_weight = {0: 1., 1: 1.}
        if not augment:
            print('Not using data augmentation.')
            model.fit(x_train, y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS_NUM,
                      validation_data=(x_valid, y_valid),
                      shuffle=True,
                      callbacks=callbacks,
                      class_weight=class_weight)
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=False)

            datagen.fit(x_train, augment=True, rounds=10)
            model.fit_generator(
                datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
                steps_per_epoch=len(x_train) / BATCH_SIZE, callbacks=callbacks,
                validation_data=(x_valid, y_valid),
                epochs=EPOCHS_NUM, verbose=1, workers=4)

        model.save(last_model_path)
        kfold_no += 1
        # break

    return


if __name__ == "__main__":

    # volume_type = "flair"
    volume_type = "t1ce"
    # volume_type = "t2"

    # model_type = "vggish"
    model_type = "pyramid"

    # opt_type = "sgd"
    opt_type = "adam"
    # opt_type = "adagrade"

    model_name = "_".join([volume_type, model_type, opt_type])

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data", "Original", "BraTS")
    hgg_dir = os.path.join(data_dir, "HGGTrimmed")
    lgg_dir = os.path.join(data_dir, "LGGTrimmed")

    hgg_subjects = get_data_path(hgg_dir, volume_type, 1)
    lgg_subjects = get_data_path(lgg_dir, volume_type, 0)

    hgg_train, hgg_test = get_dataset(hgg_subjects)
    lgg_train, lgg_test = get_dataset(lgg_subjects)

    trainset_info = hgg_train + lgg_train
    testset_info = hgg_test + lgg_test

    save_to_csv(trainset_info, "train.csv")
    save_to_csv(testset_info, "test.csv")

    models_dir = os.path.join(parent_dir, "models")
    logs_dir = os.path.join(parent_dir, "logs")

    train(trainset_info, model_type, models_dir,
          logs_dir, opt_type, model_name, False)
