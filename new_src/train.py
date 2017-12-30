import os
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (log_loss, recall_score,
                             precision_score, roc_auc_score,
                             roc_curve)
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (ModelCheckpoint,
                             LearningRateScheduler,
                             # ReduceLROnPlateau,
                             TensorBoard)


# SEED = 77
TRAIN_PROP = 0.8
TEST_PROP = 0.2
VOLUME_SIZE = [112, 112, 96, 1]

BATCH_SIZE = 8
EPOCHS_NUM = 60
SPLITS_NUM = 4


def get_data_path(dir_path, volume_type, label, SEED):
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


def load_data(info, mode):
    x, y = [], []
    print("Loading {} data ...".format(mode))
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

    def dylr(start, end):
        lrs = [start]
        diff = (start - end) / (EPOCHS_NUM - 1)
        for i in range(1, EPOCHS_NUM - 1):
            lrs.append(start - i * diff)
        lrs.append(end)
        return lrs

    # lrs = [1e-3] * 10 + [5e-4] * 10 + [1e-4] * 10 + [5e-5] * 10 + [1e-5] * 0
    lrs = dylr(1e-4, 1e-6)
    lr = lrs[epoch]
    print("\n------------------------------------------------")
    print("Learning rate: ", lr)

    return lr


def load_model(model_type):
    if model_type == "vggish":
        model = vggish()
    elif model_type == "pyramid":
        model = pyramid()
    return model


def train(trainset_info, testset_info, model_type, models_dir,
          logs_dir, optimizer, model_name, augment=False):

    x_test, y_test = load_data(testset_info, "testset")
    y_test = to_categorical(y_test, num_classes=2)

    x, y = load_data(trainset_info, "trainset")
    kfold = StratifiedKFold(n_splits=SPLITS_NUM, shuffle=True)
    kfold_no = 0
    cvlosses, cvaccs = [], []

    for tidx, vidx in kfold.split(x, y):
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
        print("Model: ", model_name)
        print("KFold: ", kfold_no, "\n")

        model_dir = os.path.join(models_dir, model_name)
        kmodel_dir = os.path.join(model_dir, "kfold" + str(kfold_no))
        create_dir(kmodel_dir)

        log_dir = os.path.join(logs_dir, model_name)
        klog_dir = os.path.join(log_dir, "kfold" + str(kfold_no))
        create_dir(klog_dir)

        best_model_path = os.path.join(kmodel_dir, "best.h5")
        last_model_path = os.path.join(kmodel_dir, "last.h5")
        logs_path = os.path.join(kmodel_dir, "learning_curv.csv")
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
        tb = TensorBoard(log_dir=klog_dir, batch_size=BATCH_SIZE)
        callbacks = [checkpoint, lr_scheduler, csv_logger, tb]

        class_weight = {0: 1., 1: 1.}
        if not augment:
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
        score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
        cvlosses.append(score[0])
        cvaccs.append(score[1])
        kfold_no += 1
        # break

    test_loss = "\nLoss of testset: {0:.3f} (+/- {1:.3f})".format(np.mean(cvlosses), np.std(cvlosses))
    test_acc = "Accuracy of testset: {0:.3f}% (+/- {1:.3f}%)\n".format(np.mean(cvaccs), np.std(cvaccs))
    test_log = test_loss + "\n" + test_acc
    test_logs_path = os.path.join(model_dir, "test_log.txt")
    print(test_log)
    with open(test_logs_path, "w") as file:
        file.write(test_log)

    return


def test(SEED, testset_info, model_type, models_dir, model_name, test_logs_dir):
    x_test, y_test = load_data(testset_info, "testset")
    y_test_category = to_categorical(y_test, num_classes=2)

    hgg_idx = np.where(y_test == 1)[0]
    lgg_idx = np.where(y_test == 0)[0]

    model_dir = os.path.join(models_dir, model_name)
    kfold_dirs = [os.path.join(model_dir, k) for k in os.listdir(model_dir)]

    predictions = []
    for kfold_dir in kfold_dirs:
        if not os.path.isdir(kfold_dir):
            continue

        model_path = os.path.join(kfold_dir, "last.h5")

        if model_type == "vggish":
            model = vggish()
        elif model_type == "pyramid":
            model = pyramid()

        model.load_weights(model_path)
        prediction = model.predict(x_test)
        predictions.append(prediction)

    mean_prediction = np.zeros(y_test_category.shape)
    for prediction in predictions:
        mean_prediction += prediction
    mean_prediction = mean_prediction / 4

    arg_prediction = np.reshape(np.argmax(mean_prediction, axis=1).astype(np.int), (-1, 1))

    total_accuracy = (y_test == arg_prediction).all(axis=1).mean()
    hgg_accuracy = (y_test[hgg_idx] == arg_prediction[hgg_idx]).all(axis=1).mean()
    lgg_accuracy = (y_test[lgg_idx] == arg_prediction[lgg_idx]).all(axis=1).mean()

    total_loss = log_loss(y_test_category, mean_prediction, normalize=True)
    hgg_loss = log_loss(y_test_category[hgg_idx], mean_prediction[hgg_idx], normalize=True)
    lgg_loss = log_loss(y_test_category[lgg_idx], mean_prediction[lgg_idx], normalize=True)

    hgg_precision = precision_score(y_test, arg_prediction, pos_label=1)
    hgg_recall = recall_score(y_test, arg_prediction, pos_label=1)

    lgg_precision = precision_score(y_test, arg_prediction, pos_label=0)
    lgg_recall = recall_score(y_test, arg_prediction, pos_label=0)

    roc_auc = roc_auc_score(y_test, mean_prediction[:, 1])
    roc_line = roc_curve(y_test, mean_prediction[:, 1], pos_label=1)

    df = pd.DataFrame(data={"seed": SEED,
                            "acc": total_accuracy,
                            "hgg_acc": hgg_accuracy,
                            "lgg_acc": lgg_accuracy,
                            "loss": total_loss,
                            "hgg_loss": hgg_loss,
                            "lgg_loss": lgg_loss,
                            "hgg_precision": hgg_precision,
                            "lgg_precision": lgg_precision,
                            "hgg_recall": hgg_recall,
                            "lgg_recall": lgg_recall,
                            "roc_auc": roc_auc}, index=[0])
    df = df[["seed", "acc", "hgg_acc", "lgg_acc",
             "loss", "hgg_loss", "lgg_loss",
             "hgg_precision", "hgg_recall",
             "lgg_precision", "lgg_recall",
             "roc_auc"]]

    subject_log_dir = os.path.join(test_logs_dir, model_name)
    create_dir(subject_log_dir)

    df_path = os.path.join(subject_log_dir, "metrics.csv")
    df.to_csv(df_path, index=False)

    np.save(os.path.join(subject_log_dir, "roc_curve.npy"), roc_line)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    mode_help_str = "Select a mode type in 'train', or 'test'."
    parser.add_argument("--mode", action="store", default="train",
                        dest="mode", help=mode_help_str)

    volume_help_str = "Select a volume type in 'flair', 't1ce', or 't2'."
    parser.add_argument("--volume", action="store", default="t1ce",
                        dest="volume", help=volume_help_str)

    model_help_str = "Select a model type in 'pyramid', or 'vggish'."
    parser.add_argument("--model", action="store", default="pyramid",
                        dest="model", help=model_help_str)

    opt_help_str = "Select a optimizer type in 'adam', 'sgd', or 'adagrade'."
    parser.add_argument("--opt", action="store", default="adam",
                        dest="opt", help=opt_help_str)

    seed_help_str = "Set a seed."
    parser.add_argument("--seed", action="store", default="0",
                        dest="seed", help=seed_help_str)

    args = parser.parse_args()

    # volume_type = "flair"
    # volume_type = "t1ce"
    # volume_type = "t2"

    # model_type = "vggish"
    # model_type = "pyramid"

    # opt_type = "sgd"
    # opt_type = "adam"
    # opt_type = "adagrade"

    mode = args.mode
    volume_type = args.volume
    model_type = args.model
    opt_type = args.opt
    SEED = args.seed

    model_name = "_".join([volume_type, model_type, opt_type, SEED])

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data", "Original", "BraTS")
    hgg_dir = os.path.join(data_dir, "HGGTrimmed")
    lgg_dir = os.path.join(data_dir, "LGGTrimmed")

    hgg_subjects = get_data_path(hgg_dir, volume_type, 1, int(SEED))
    lgg_subjects = get_data_path(lgg_dir, volume_type, 0, int(SEED))

    hgg_train, hgg_test = get_dataset(hgg_subjects)
    lgg_train, lgg_test = get_dataset(lgg_subjects)

    trainset_info = hgg_train + lgg_train
    testset_info = hgg_test + lgg_test

    # save_to_csv(trainset_info, "train.csv")
    # save_to_csv(testset_info, "test.csv")

    models_dir = os.path.join(parent_dir, "models")
    logs_dir = os.path.join(parent_dir, "logs")
    test_logs_dir = os.path.join(parent_dir, "test_logs")

    if mode == "train":
        train(trainset_info, testset_info,
              model_type, models_dir,
              logs_dir, opt_type, model_name, False)
    else:
        test(SEED, testset_info, model_type, models_dir, model_name, test_logs_dir)
