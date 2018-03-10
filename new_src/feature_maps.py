from __future__ import print_function

import os
import shutil
import numpy as np
import nibabel as nib
from models import *
import matplotlib.pyplot as plt

from random import seed, shuffle


TRAIN_VALID_PROP = 0.85
VALID_PROP = 0.15
TEST_PROP = 0.15
VOLUME_SIZE = [1, 112, 96, 96, 1]


def create_dir(path, rm=True):
    if os.path.isdir(path):
        if rm:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
    return


def load_nii(path):
    volume = nib.load(path).get_data()
    volume = np.transpose(volume, axes=[1, 0, 2])
    volume = np.flipud(volume)
    # plt.figure()
    # plt.imshow(volume[..., volume.shape[-1] // 2], cmap="gray")
    # plt.show()
    volume_obj = volume[volume > 0]
    volume = (volume - np.mean(volume_obj)) / np.std(volume_obj)
    return volume


def save2nii(to_path, data):
    data = np.rot90(data, 3)
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, to_path)
    return


def extract_features(info, weight_path, feature_dir):
    feats_dir = os.path.join(feature_dir)
    create_dir(feats_dir, rm=False)

    model = pyramid5(scale=1)
    model.summary()
    model.load_weights(weight_path)

    # conv0_layer = Model(inputs=model.input,
    #                     outputs=model.get_layer("conv3d_1").output)
    # conv1_layer = Model(inputs=model.input,
    #                     outputs=model.get_layer("conv3d_2").output)
    # conv2_layer = Model(inputs=model.input,
    #                     outputs=model.get_layer("conv3d_3").output)
    # conv3_layer = Model(inputs=model.input,
    #                     outputs=model.get_layer("conv3d_4").output)

    # conv4_layer = Model(inputs=model.input,
    #                     outputs=model.get_layer("conv4").output)
    # conv5_layer = Model(inputs=model.input,
    #                     outputs=model.get_layer("conv5").output)
    # conv6_layer = Model(inputs=model.input,
    #                     outputs=model.get_layer("conv6").output)
    # conv7_layer = Model(inputs=model.input,
    #                     outputs=model.get_layer("conv7").output)

    # conv_layers = [conv0_layer, conv1_layer,
    #                conv2_layer, conv3_layer,
    #                conv4_layer, conv5_layer,
    #                conv6_layer, conv7_layer]
    # conv_dirs = ["conv0", "conv1", "conv2", "conv3",
    #              "conv4", "conv5", "conv6", "conv7"]

    conv_layer = Model(inputs=model.input,
                       outputs=model.get_layer("scale1").output)
    conv_layers = [conv_layer]
    conv_dirs = ["scale1"]

    subj = info[0]
    volume_path = subj[0]
    print(volume_path)

    for out_dir, out_layer in zip(conv_dirs, conv_layers):
        out_dir = os.path.join(feats_dir, out_dir)
        create_dir(out_dir, rm=False)

        volume = load_nii(volume_path)
        volume = np.expand_dims(volume, axis=0)
        volume = np.expand_dims(volume, axis=4)
        out = out_layer.predict(volume)

        num = out.shape[-1]
        for i in range(num):
            feature_map = out[..., i][0, ...]
            out_path = os.path.join(out_dir, str(i) + ".nii.gz")
            save2nii(out_path, feature_map)

    return


def get_data_path(dir_path, label, SEED):
    subjects = os.listdir(dir_path)
    seed(SEED)
    shuffle(subjects)
    subjects_paths = []
    for subject in subjects:
        subject_dir = os.path.join(dir_path, subject)
        for scan_name in os.listdir(subject_dir):
            if "t1ce" in scan_name:
                subjects_paths.append([os.path.join(subject_dir, scan_name), label])
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


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    models_dir = os.path.join(parent_dir, "models")

    data_dir = os.path.join(parent_dir, "data")
    hgg_dir = os.path.join(data_dir, "HGGTrimmed2")
    lgg_dir = os.path.join(data_dir, "LGGTrimmed2")

    hgg_subjects = get_data_path(hgg_dir, 1, 67)
    lgg_subjects = get_data_path(lgg_dir, 0, 67)

    hgg_train, hgg_valid, hgg_test = get_dataset(hgg_subjects, True)
    lgg_train, lgg_valid, lgg_test = get_dataset(lgg_subjects, True)

    trainset_info = hgg_train + lgg_train
    validset_info = hgg_valid + lgg_valid
    testset_info = hgg_test + lgg_test

    weight_path = os.path.join(models_dir, "model-fm-s1", "last.h5")
    feature_dir = os.path.join(parent_dir, "feature_maps")

    extract_features(trainset_info, weight_path, feature_dir)
