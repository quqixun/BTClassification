from __future__ import print_function


import numpy as np
import os

import math
import json
import shutil
import argparse
from tqdm import *
from models import *
import pandas as pd
import nibabel as nib
# import matplotlib.pyplot as plt
from random import seed, shuffle

from keras.layers import *
from keras.callbacks import CSVLogger
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from sklearn.metrics import (log_loss, recall_score,
                             precision_score, roc_auc_score,
                             roc_curve, confusion_matrix)
from keras.utils import to_categorical
from keras.callbacks import (ModelCheckpoint,
                             LearningRateScheduler,
                             TensorBoard)


TRAIN_VALID_PROP = 0.85
VALID_PROP = 0.15
TEST_PROP = 0.15
VOLUME_SIZE = [112, 96, 96, 1]


def get_data_path(dir_path, label, SEED):
    subjects = os.listdir(dir_path)
    seed(SEED)
    shuffle(subjects)
    subjects_paths = []
    for subject in subjects:
        subjects_paths.append([os.path.join(dir_path, subject), label])
    return subjects_paths


def get_dataset(subjects, valid=False):
    subj_num = len(subjects)
    train_idx = int(round(subj_num * TRAIN_VALID_PROP))

    test_set = subjects[train_idx:]
    if valid:
        valid_idx = int(round(subj_num * VALID_PROP))
        valid_set = subjects[:valid_idx]
        train_set = subjects[valid_idx:train_idx]
        return train_set, valid_set, test_set
    else:
        train_set = subjects[:train_idx]
        return train_set, test_set


def save_to_csv(subjects, csv_path):
    subj_paths = [subj[0] for subj in subjects]
    subj_labels = [subj[1] for subj in subjects]

    df = pd.DataFrame(data={"subject": subj_paths, "label": subj_labels})
    df = df[["subject", "label"]]
    df.to_csv(csv_path, index=False)
    return


def load_data(info, mode, norm=1):
    x, y = [], []
    print("Loading {} data ...".format(mode))
    for subject in info:
        volume_path, label = subject[0], subject[1]
        vpaths = []
        if os.path.isdir(volume_path):
            for scan in os.listdir(volume_path):
                vpaths.append(os.path.join(volume_path, scan))
        else:
            vpaths.append(volume_path)
        for vpath in vpaths:
            volume = nib.load(vpath).get_data()
            volume = np.transpose(volume, [2, 0, 1])
            volume = np.rot90(volume, 2)
            if norm:
                if norm == 1:
                    obj_idx = np.where(volume > 0)
                    obj = volume[obj_idx]
                    obj = (obj - np.mean(obj)) / np.std(obj)
                    volume[obj_idx] = obj
                elif norm == 2:
                    volume = (volume - np.mean(volume)) / np.std(volume)
                elif norm == 3:
                    volume = volume / np.max(volume)
            volume = np.expand_dims(volume, axis=3)
            x.append(volume.astype(np.float32))
            y.append(label)

    x = np.array(x)
    y = np.array(y).reshape((-1, 1))

    return x, y


def create_dir(path, rm=True):
    if os.path.isdir(path):
        if rm:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
    return


def lr_schedule(epoch):

    global epochs_num, lr_start, lr_end

    def dylr(epochs_num, lr_start, lr_end):
        lrs = [lr_start]

        if epochs_num == 1:
            return lrs

        diff = (lr_start - lr_end) / (epochs_num - 1)
        for i in range(1, epochs_num - 1):
            lrs.append(lr_start - i * diff)
        lrs.append(lr_end)

        # lrs = [1e-5] * 30 + [1e-6] * 70

        return lrs

    lrs = dylr(epochs_num, lr_start, lr_end)
    lr = lrs[epoch]
    print("\n------------------------------------------------")
    print("Learning rate: ", lr)

    return lr


def step_decay(epoch):
    global epochs_num, lr_start, lr_end
    initial_lrate = lr_start
    if initial_lrate <= 1e-5:
        drop = 1.0
    elif initial_lrate == 1e-4:
        drop = 1.0 / 3.0
    else:
        drop = 0.1
    epochs_drop = 30.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print("\n------------------------------------------------")
    print("Learning rate: ", lrate)
    return lrate


def two_step_decay(epoch):
    global epochs_num, lr_start
    # num1 = int(epochs_num * 0.3)
    # num2 = int(epochs_num * 0.4)
    # lrs = [lr_start] * num1 + [lr_start * 0.1] * num1 + [lr_start * 0.01] * num2
    lrs = [lr_start] * 40 + [lr_start * 0.5] * 30 + [lr_start * 0.1] * 30
    lr = lrs[epoch]
    print("\n------------------------------------------------")
    print("Learning rate: ", lr)
    return lr


def test(SEED, x, y,
         model_name, model_type,
         models_dir, test_logs_dir, mode, pool, scale):
    print(mode)
    y_category = to_categorical(y, num_classes=2)

    ad_idx = np.where(y == 1)[0]
    nc_idx = np.where(y == 0)[0]

    model_dir = os.path.join(models_dir, model_name)

    if not os.path.isdir(model_dir):
        raise IOError("Model directory is not exist.")

    if model_type == "pyramid":
        model_fn = pyramid
    elif model_type == "pyramid2":
        model_fn = pyramid2
    elif model_type == "pyramid3":
        model_fn = pyramid3
    elif model_type == "pyramid4":
        model_fn = pyramid4
    elif model_type == "pyramid5":
        model_fn = pyramid5
    elif model_type == "pyramid6":
        model_fn = pyramid6
    elif model_type == "pyramid_bba":
        model_fn = pyramid_bba
    elif model_type == "vggish":
        model_fn = vggish
    elif model_type == "vggish2":
        model_fn = vggish2

    model = model_fn(pool=pool, scale=scale)

    for style in ["last", "best"]:
        model_path = os.path.join(model_dir, style + ".h5")
        model.load_weights(model_path)
        prediction = model.predict(x)

        arg_prediction = np.reshape(np.argmax(prediction, axis=1).astype(np.int), (-1, 1))

        total_accuracy = (y == arg_prediction).all(axis=1).mean()
        ad_accuracy = (y[ad_idx] == arg_prediction[ad_idx]).all(axis=1).mean()
        nc_accuracy = (y[nc_idx] == arg_prediction[nc_idx]).all(axis=1).mean()

        total_loss = log_loss(y_category, prediction, normalize=True)
        ad_loss = log_loss(y_category[ad_idx], prediction[ad_idx], normalize=True)
        nc_loss = log_loss(y_category[nc_idx], prediction[nc_idx], normalize=True)

        ad_precision = precision_score(y, arg_prediction, pos_label=1)
        ad_recall = recall_score(y, arg_prediction, pos_label=1)

        nc_precision = precision_score(y, arg_prediction, pos_label=0)
        nc_recall = recall_score(y, arg_prediction, pos_label=0)

        roc_auc = roc_auc_score(y, prediction[:, 1])
        roc_line = roc_curve(y, prediction[:, 1], pos_label=1)

        tn, fp, fn, tp = confusion_matrix(y, arg_prediction).ravel()

        df = pd.DataFrame(data={"seed": SEED,
                                "acc": total_accuracy,
                                "ad_acc": ad_accuracy,
                                "nc_acc": nc_accuracy,
                                "loss": total_loss,
                                "ad_loss": ad_loss,
                                "nc_loss": nc_loss,
                                "ad_precision": ad_precision,
                                "nc_precision": nc_precision,
                                "ad_recall": ad_recall,
                                "nc_recall": nc_recall,
                                "roc_auc": roc_auc,
                                "TN": tn,
                                "FP": fp,
                                "FN": fn,
                                "TP": tp}, index=[0])
        df = df[["seed", "acc", "ad_acc", "nc_acc",
                 "loss", "ad_loss", "nc_loss",
                 "ad_precision", "ad_recall",
                 "nc_precision", "nc_recall",
                 "roc_auc", "TN", "FP", "FN", "TP"]]

        subject_log_dir = os.path.join(test_logs_dir, model_name)
        create_dir(subject_log_dir, False)

        df_path = os.path.join(subject_log_dir, mode + "_" + style + "_metrics.csv")
        df.to_csv(df_path, index=False)

        np.save(os.path.join(subject_log_dir, mode + "_" + style + "_roc_curve.npy"), roc_line)
    return


def data_augment(x_train, y_train):
    print("Do Augmentation on Training Set ...")
    aug_x_train, aug_y_train = [], []
    for i in range(len(y_train)):
        aug_x_train.append(x_train[i])
        aug_y_train.append(y_train[i])
        aug_x_train.append(np.fliplr(x_train[i]))
        aug_y_train.append(np.array([0]))
    x_train = np.array(aug_x_train)
    y_train = np.array(aug_y_train).reshape((-1, 1))

    return x_train, y_train


def train(trainset_info, validset_info, testset_info,
          paras, models_dir, logs_dir, test_logs_dir):

    # Load parameters
    model_name = paras["model_name"]
    model_type = paras["model_type"]
    SEED = paras["seed"]

    optimizer = paras["optimizer"]
    batch_size = paras["batch_size"]
    l2_coeff = paras["l2_coeff"]
    bn_momentum = paras["bn_momentum"]
    initializer = paras["initializer"]
    drop_rate = paras["drop_rate"]

    global epochs_num, lr_start, lr_end
    epochs_num = paras["epochs_num"]
    lr_start = paras["lr_start"]
    lr_end = paras["lr_end"]
    lr_style = paras["lr_style"]

    augment = paras["augment"]
    norm = paras["norm"]
    pool = paras["pool"]
    scale = paras["scale"]

    # Load dataset
    x_train, y_train = load_data(trainset_info, "trainset", norm)
    train_1_num = len(np.where(y_train == 1)[0])
    train_0_num = len(y_train) - train_1_num
    print("Train 1:", train_1_num, "Train 0:", train_0_num)
    if augment:
        x_train, y_train = data_augment(x_train, y_train)
    y_train_category = to_categorical(y_train, num_classes=2)

    x_test, y_test = load_data(testset_info, "testset", norm)
    y_test_category = to_categorical(y_test, num_classes=2)
    test_1_num = len(np.where(y_test == 1)[0])
    test_0_num = len(y_test) - test_1_num
    print("Test 1:", test_1_num, "Test 0:", test_0_num)

    x_valid, y_valid = load_data(validset_info, "validset", norm)
    y_valid_category = to_categorical(y_valid, num_classes=2)
    valid_1_num = len(np.where(y_valid == 1)[0])
    valid_0_num = len(y_valid) - valid_1_num
    print("Valid 1:", valid_1_num, "Valid 0:", valid_0_num)

    # x_train_mean = np.mean(x_train, axis=0)
    # x_train -= x_train_mean
    # x_test -= x_train_mean
    # x_valid -= x_train_mean

    if model_type == "pyramid":
        model_fn = pyramid
    elif model_type == "pyramid2":
        model_fn = pyramid2
    elif model_type == "pyramid3":
        model_fn = pyramid3
    elif model_type == "pyramid_bba":
        model_fn = pyramid_bba
    elif model_type == "pyramid4":
        model_fn = pyramid4
    elif model_type == "pyramid5":
        model_fn = pyramid5
    elif model_type == "pyramid6":
        model_fn = pyramid6

    model = model_fn(l2_coeff, bn_momentum,
                     initializer, drop_rate, pool, scale)

    time_decay = lr_start / epochs_num
    if optimizer == "adam":
        opt = Adam(lr=lr_start, epsilon=1e-8, decay=time_decay, amsgrad=True)
    elif optimizer == "adagrad":
        opt = Adagrad(lr=lr_start, epsilon=1e-8, decay=time_decay)
    elif optimizer == "sgd":
        opt = SGD(lr=lr_start, decay=time_decay, momentum=0.9, nesterov=True)
    elif optimizer == "adadelta":
        opt = Adadelta(lr=lr_start)
    elif optimizer == "rmsprop":
        opt = RMSprop(lr=lr_start)

    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    model.summary()

    print("Model: ", model_name)
    # print("Parameters: ", paras)

    model_dir = os.path.join(models_dir, model_name)
    create_dir(model_dir)

    log_dir = os.path.join(logs_dir, model_name)
    create_dir(log_dir)

    best_model_path = os.path.join(model_dir, "best.h5")
    last_model_path = os.path.join(model_dir, "last.h5")
    logs_path = os.path.join(model_dir, "learning_curve.csv")
    csv_logger = CSVLogger(logs_path, append=True, separator=",")

    checkpoint = ModelCheckpoint(filepath=best_model_path,
                                 monitor="val_loss",
                                 verbose=0,
                                 save_best_only=True)

    if lr_style == 1:
        lr_scheduler = LearningRateScheduler(lr_schedule)
    elif lr_style == 2:
        lr_scheduler = LearningRateScheduler(step_decay)
    elif lr_style == 3:
        lr_scheduler = LearningRateScheduler(two_step_decay)

    tb = TensorBoard(log_dir=log_dir, batch_size=batch_size)
    # callbacks = [checkpoint, lr_scheduler, csv_logger, tb]
    callbacks = [checkpoint, csv_logger, tb]

    class_weight = [1., 1.]
    model.fit(x_train, y_train_category,
              batch_size=batch_size,
              epochs=epochs_num,
              validation_data=(x_valid, y_valid_category),
              shuffle=True,
              callbacks=callbacks,
              class_weight=class_weight)

    model.save(last_model_path)
    train_score = model.evaluate(x_train, y_train_category, batch_size=batch_size, verbose=0)
    valid_score = model.evaluate(x_valid, y_valid_category, batch_size=batch_size, verbose=0)
    test_score = model.evaluate(x_test, y_test_category, batch_size=batch_size, verbose=0)

    print("Training Set: Loss: {0:.4f}, Accuracy: {1:.4f}".format(train_score[0], train_score[1]))
    print("Validation Set: Loss: {0:.4f}, Accuracy: {1:.4f}".format(valid_score[0], valid_score[1]))
    print("Testing Set: Loss: {0:.4f}, Accuracy: {1:.4f}".format(test_score[0], test_score[1]))

    # test(SEED, x_train, y_train, model_name, model_type,
    #      models_dir, test_logs_dir, "train", pool)
    test(SEED, x_valid, y_valid, model_name, model_type,
         models_dir, test_logs_dir, "valid", pool, scale)
    test(SEED, x_test, y_test, model_name, model_type,
         models_dir, test_logs_dir, "test", pool, scale)

    # K.clear_session()

    return


def load_paras(file_path, model):
    paras = json.load(open(file_path))
    return paras[model]


def get_sepdata_path(data_dir):
    groups = os.listdir(data_dir)
    data_info = []
    for group in groups:
        if group == "AD":
            label = 1
        else:
            label = 0

        group_dir = os.path.join(data_dir, group)
        for data_name in os.listdir(group_dir):
            data_path = os.path.join(group_dir, data_name)
            data_info.append([data_path, label])

    shuffle(data_info)
    return data_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    model_help_str = "Select a model."
    parser.add_argument("--model", action="store", default="model0",
                        dest="model", help=model_help_str)

    args = parser.parse_args()
    model = args.model

    parent_dir = os.path.dirname(os.getcwd())

    if "best" in model:
        paras_path = os.path.join(os.getcwd(), "models_best.json")
        paras = load_paras(paras_path, model)
        models_dir = os.path.join(parent_dir, "models_best")
        logs_dir = os.path.join(parent_dir, "logs_best")
        test_logs_dir = os.path.join(parent_dir, "test_logs_best")
    else:
        paras_path = os.path.join(os.getcwd(), "models.json")
        paras = load_paras(paras_path, model)
        models_dir = os.path.join(parent_dir, "models")
        logs_dir = os.path.join(parent_dir, "logs")
        test_logs_dir = os.path.join(parent_dir, "test_logs")

    volume_type = paras["volume_type"]
    data_dir = os.path.join(parent_dir, "data", volume_type)

    if volume_type == "sepsubj":
        trainset_info = get_sepdata_path(os.path.join(data_dir, "train"))
        validset_info = get_sepdata_path(os.path.join(data_dir, "valid"))
        testset_info = get_sepdata_path(os.path.join(data_dir, "test"))
    else:
        ad_dir = os.path.join(data_dir, "AD")
        nc_dir = os.path.join(data_dir, "NC")

        SEED = paras["seed"]
        ad_subjects = get_data_path(ad_dir, 1, int(SEED))
        nc_subjects = get_data_path(nc_dir, 0, int(SEED))

        ad_train, ad_valid, ad_test = get_dataset(ad_subjects, True)
        nc_train, nc_valid, nc_test = get_dataset(nc_subjects, True)

        print("AD train:", len(ad_train))
        print("AD valid:", len(ad_valid))
        print("AD test:", len(ad_test))
        print("NC train:", len(nc_train))
        print("NC valid:", len(nc_valid))
        print("NC test:", len(nc_test))

        trainset_info = ad_train + nc_train
        validset_info = ad_valid + nc_valid
        testset_info = ad_test + nc_test

    train(trainset_info, validset_info, testset_info,
          paras, models_dir, logs_dir, test_logs_dir)
