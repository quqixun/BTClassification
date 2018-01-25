from __future__ import print_function

import os
import json
import shutil
import argparse
import numpy as np
from tqdm import *
from models import *
import pandas as pd
import nibabel as nib
from random import seed, shuffle

from keras.layers import *
from keras.callbacks import CSVLogger
from keras.optimizers import SGD, Adam, Adagrad
from sklearn.metrics import (log_loss, recall_score,
                             precision_score, roc_auc_score,
                             roc_curve, confusion_matrix)
from keras.utils import to_categorical
from keras.callbacks import (ModelCheckpoint,
                             LearningRateScheduler,
                             TensorBoard)


TRAIN_VALID_PROP = 0.8
VALID_PROP = 0.2
TEST_PROP = 0.2
VOLUME_SIZE = [112, 112, 96, 1]

BATCH_SIZE = 8
EPOCHS_NUM = 60
SPLITS_NUM = 4


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


def load_data(info, mode):
    x, y = [], []
    print("Loading {} data ...".format(mode))
    for subject in info:
        volume_path, label = subject[0], subject[1]
        volume = nib.load(volume_path).get_data()
        volume = np.rot90(np.transpose(volume, [0, 2, 1]), 1)
        # volume = np.rot90(volume, 3)
        volume_obj = volume[volume > 0]
        volume = (volume - np.mean(volume_obj)) / np.std(volume_obj)
        volume = np.reshape(volume, VOLUME_SIZE)
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
        return lrs

    lrs = dylr(epochs_num, lr_start, lr_end)
    lr = lrs[epoch]
    print("\n------------------------------------------------")
    print("Learning rate: ", lr)

    return lr


def test(SEED, x, y,
         model_name, model_type,
         models_dir, test_logs_dir, mode):
    print(mode)
    y_category = to_categorical(y, num_classes=2)

    ad_idx = np.where(y == 1)[0]
    nc_idx = np.where(y == 0)[0]

    model_dir = os.path.join(models_dir, model_name)

    if not os.path.isdir(model_dir):
        raise IOError("Model directory is not exist.")

    if model_type == "pyramid":
        model = pyramid()

    model_path = os.path.join(model_dir, "last.h5")
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

    df_path = os.path.join(subject_log_dir, mode + "_metrics.csv")
    df.to_csv(df_path, index=False)

    np.save(os.path.join(subject_log_dir, mode + "_roc_curve.npy"), roc_line)
    return


def augment(x_train, y_train):
    print("Do Augmentation on nc Samples ...")
    aug_x_train, aug_y_train = [], []
    for i in range(len(y_train)):
        aug_x_train.append(x_train[i])
        aug_y_train.append(y_train[i])
        if y_train[i] == 0:
            aug_x_train.append(np.fliplr(x_train[i]))
            aug_y_train.append(np.array([0]))
    x_train = np.array(aug_x_train)
    y_train = np.array(aug_y_train).reshape((-1, 1))

    return x_train, y_train


def train(trainset_info, validset_info, testset_info,
          paras, models_dir, logs_dir, test_logs_dir):
    # Load dataset
    x_test, y_test = load_data(testset_info, "testset")
    y_test_category = to_categorical(y_test, num_classes=2)

    x_valid, y_valid = load_data(validset_info, "validset")
    y_valid_category = to_categorical(y_valid, num_classes=2)

    x_train, y_train = load_data(trainset_info, "trainset")
    # x_train, y_train = augment(x_train, y_train)
    y_train_category = to_categorical(y_train, num_classes=2)

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

    if model_type == "pyramid":
        model_fn = pyramid

    model = model_fn(l2_coeff, bn_momentum, initializer, drop_rate)

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
                                 verbose=1,
                                 save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    tb = TensorBoard(log_dir=log_dir, batch_size=batch_size)
    callbacks = [checkpoint, lr_scheduler, csv_logger, tb]

    model.fit(x_train, y_train_category,
              batch_size=batch_size,
              epochs=epochs_num,
              validation_data=(x_valid, y_valid_category),
              shuffle=True,
              callbacks=callbacks)

    model.save(last_model_path)
    # score[0]: loss, score[1]: accuracy
    train_score = model.evaluate(x_train, y_train_category, batch_size=batch_size, verbose=0)
    valid_score = model.evaluate(x_valid, y_valid_category, batch_size=batch_size, verbose=0)
    test_score = model.evaluate(x_test, y_test_category, batch_size=batch_size, verbose=0)

    print("Training Set: Loss: {0:.4f}, Accuracy: {1:.4f}".format(train_score[0], train_score[1]))
    print("Validation Set: Loss: {0:.4f}, Accuracy: {1:.4f}".format(valid_score[0], valid_score[1]))
    print("Testing Set: Loss: {0:.4f}, Accuracy: {1:.4f}".format(test_score[0], test_score[1]))

    test(SEED, x_train, y_train, model_name, model_type,
         models_dir, test_logs_dir, "train")
    test(SEED, x_valid, y_valid, model_name, model_type,
         models_dir, test_logs_dir, "valid")
    test(SEED, x_test, y_test, model_name, model_type,
         models_dir, test_logs_dir, "test")

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

    if "-" in model:
        paras_path = os.path.join(os.getcwd(), "models_best2.json")
        paras = load_paras(paras_path, model)
        models_dir = os.path.join(parent_dir, "models_best2")
        logs_dir = os.path.join(parent_dir, "logs_best2")
        test_logs_dir = os.path.join(parent_dir, "test_logs_best2")
    else:
        paras_path = os.path.join(os.getcwd(), "models.json")
        paras = load_paras(paras_path, model)
        models_dir = os.path.join(parent_dir, "models")
        logs_dir = os.path.join(parent_dir, "logs")
        test_logs_dir = os.path.join(parent_dir, "test_logs")

    # ----------------------------------------------------------------
    # data_dir = os.path.join(parent_dir, "data", "Original", "BraTS")
    # ad_dir = os.path.join(data_dir, "adTrimmed")
    # nc_dir = os.path.join(data_dir, "ncTrimmed")

    # volume_type = paras["volume_type"]
    # SEED = paras["seed"]
    # ad_subjects = get_data_path(ad_dir, volume_type, 1, int(SEED))
    # nc_subjects = get_data_path(nc_dir, volume_type, 0, int(SEED))

    # ad_train, ad_valid, ad_test = get_dataset(ad_subjects, True)
    # nc_train, nc_valid, nc_test = get_dataset(nc_subjects, True)

    # trainset_info = ad_train + nc_train
    # validset_info = ad_valid + nc_valid
    # testset_info = ad_test + nc_test

    # with open("trainset_info", "rb") as fp:
    #     trainset_info = pickle.load(fp)

    # with open("validset_info", "rb") as fp:
    #     validset_info = pickle.load(fp)

    # with open("testset_info", "rb") as fp:
    #     testset_info = pickle.load(fp)

    # with open("trainset_info", "wb") as fp:
    #     pickle.dump(trainset_info, fp)

    # with open("validset_info", "wb") as fp:
    #     pickle.dump(validset_info, fp)

    # with open("testset_info", "wb") as fp:
    #     pickle.dump(testset_info, fp)

    # train(trainset_info, validset_info, testset_info,
    #       paras, models_dir, logs_dir, test_logs_dir)
    # ----------------------------------------------------------------

    # For separate subjects
    # data_dir = os.path.join(parent_dir, "data", "sepsubj")
    # trainset_info = get_sepdata_path(os.path.join(data_dir, "train"))
    # validset_info = get_sepdata_path(os.path.join(data_dir, "val"))
    # testset_info = get_sepdata_path(os.path.join(data_dir, "test"))

    # For none-separate subjects
    data_dir = os.path.join(parent_dir, "data", "full_processed")
    ad_dir = os.path.join(data_dir, "AD")
    nc_dir = os.path.join(data_dir, "NC")

    SEED = paras["seed"]
    ad_subjects = get_data_path(ad_dir, 1, int(SEED))
    nc_subjects = get_data_path(nc_dir, 0, int(SEED))

    ad_train, ad_valid, ad_test = get_dataset(ad_subjects, True)
    nc_train, nc_valid, nc_test = get_dataset(nc_subjects, True)

    trainset_info = ad_train + nc_train
    validset_info = ad_valid + nc_valid
    testset_info = ad_test + nc_test

    train(trainset_info, validset_info, testset_info,
          paras, models_dir, logs_dir, test_logs_dir)
