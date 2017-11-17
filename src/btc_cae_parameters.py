# Brain Tumor Classification
# Script for CAEs' Hyper-Parameters
# Author: Qixun Qu
# Create on: 2017/11/06
# Modify on: 2017/11/17

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

Hyper-parameters for training pipeline

-1- Basic Settings:
    - dims: string, dimentions of input
    - train_path: string, the path of tfrecord for training
    - validate_path: string, the path of tfrecord for validating
    - train_num: int, the number of patches in training set
    - validate_num: int, the number of patches in validating set
    - classes_num: int, the number of grading groups
    - patch_shape: int list, each patch's shape
    - capacity: int, the maximum number of elements in the queue
    - min_after_dequeue: int, minimum number elements in the queue
                         after a dequeue, used to ensure a level
                         of mixing of elements

-2- Parameters for Training:
    - batch_size: int, the number of patches in one batch
   - num_epoches: int or list of ints, the number of epoches
    - learning_rates: list of floats, gives the learning rates for
                      different training epoches
    - learning_rate_first: float, the learning rate for first epoch
    - learning_rate_last: float, the learning rate for last epoch
    - l2_loss_coeff: float, coeddicient of l2 regularization item
    - kl_coeff: float, coefficient of sparse penalty term
    - sparse_level: float, sparsity parameter
    - winner_nums: the number of winners in Winner-Take-All autoencoder
    - lifetime_rate: the percentage of winners to be kept in
                     Winner-Take-All autoencoder

-3- Parameters for Constructing Model
    - activation: string, indicates the activation method by either
                  "relu" or "sigmoid" for autoencoder
    - bn_momentum: float, momentum for removing average in batch
                   normalization, typically values are 0.999, 0.99, 0.9, etc
    - drop_rate: float, rate of dropout of input units, which is
                 between 0 and 1
    - cae_pool: pooling method in autoencoder
    - sparse_type: sparse constraint methods wither in "kl" or "wta"

'''


from __future__ import print_function

import os
import json
import math
from btc_settings import *


def get_parameters(mode="cae", data="volume", sparse="kl"):
    '''GET_PARAMETERS

        Return parameters for training autoencoder and classifier.

        Inputs:
        -------
        - mode: string, "cae" for autoencoder, "clf" for classifier
        - data: string, "volume" for 3D data, "slices" for 2D data
        - sparse: string, "kl" for KL-divergence sparcity constraint,
                  "wta" for Winner-Take-All constraint

        Output:
        - a dictionary of parameters

    '''

    # Check value of "data"
    if data == "volume":
        data_folder = VOLUMES_FOLDER
        data_shape = VOLUME_SHAPE
        data_dims = "3D"
        batch_size = 1
    elif data == "slice":
        data_folder = SLICES_FOLDER
        data_shape = SLICE_SHAPE
        data_dims = "2D"
        batch_size = 16
    else:
        raise ValueError("Cannot found data type in 'volume' or 'slice'.")

    # Check value of "sparse"
    if sparse == "kl":
        activation = "sigmoid"
        kl_coeff = 0.001
        sparse_level = 0.05
        winner_nums = None
        lifetime_rate = None
    elif sparse == "wta":
        activation = "relu"
        kl_coeff = None
        sparse_level = None
        winner_nums = 10
        lifetime_rate = 0.5
    else:
        raise ValueError("Cannot found sparse type in 'kl' or 'wta'.")

    # Set path of the folder in where tfrecords are save in
    parent_dir = os.path.dirname(os.getcwd())
    tfrecords_dir = os.path.join(parent_dir, DATA_FOLDER,
                                 TFRECORDS_FOLDER, data_folder)

    # Create paths for training and validating tfrecords
    tpath = os.path.join(tfrecords_dir, "train.tfrecord")
    vpath = os.path.join(tfrecords_dir, "validate.tfrecord")

    # Load dict from json file in which the number of
    # training and valdating set can be found
    json_path = os.path.join(TEMP_FOLDER, TFRECORDS_FOLDER,
                             data_folder, DATA_NUM_FILE)

    if not os.path.isfile(json_path):
        raise IOError("The json file is not exist.")

    # Load json file to read the number of data
    with open(json_path) as json_file:
        data_num = json.load(json_file)

    train_num = data_num["train"]
    validate_num = data_num["validate"]

    # Settings for decodeing tfrecords
    min_after_dequeue = max([train_num, validate_num])
    capacity = math.ceil(min_after_dequeue * 1.05)

    # Form parameters for autoencoder
    cae_paras = {"dims": data_dims,
                 "train_path": tpath,
                 "validate_path": vpath,
                 "train_num": train_num,
                 "validate_num": validate_num,
                 "classes_num": 3,
                 "patch_shape": data_shape,
                 "capacity": capacity,
                 "min_after_dequeue": min_after_dequeue,
                 "batch_size": batch_size,
                 "num_epoches": [1],
                 "learning_rates": [1e-3],
                 # "learning_rate_first": 1e-3,
                 # "learning_rate_last": 1e-4,
                 "l2_loss_coeff": 0.001,
                 "activation": activation,
                 "bn_momentum": 0.99,
                 "drop_rate": 0.5,
                 "cae_pool": "stride",
                 "sparse_type": sparse,
                 "kl_coeff": kl_coeff,
                 "sparse_level": sparse_level,
                 "winner_nums": winner_nums,
                 "lifetime_rate": lifetime_rate}

    # Form parameters for classifier
    clf_paras = {"dims": data_dims,
                 "train_path": tpath,
                 "validate_path": vpath,
                 "train_num": train_num,
                 "validate_num": validate_num,
                 "classes_num": 3,
                 "patch_shape": data_shape,
                 "capacity": capacity,
                 "min_after_dequeue": min_after_dequeue,
                 "batch_size": 16,
                 "num_epoches": [10],
                 "learning_rates": [1e-3],
                 # "learning_rate_first": 1e-3,
                 # "learning_rate_last": 1e-4,
                 "l2_loss_coeff": 0.001,
                 "activation": "relu",  # "lrelu"
                 "alpha": None,
                 "bn_momentum": 0.99,
                 "drop_rate": 0.0,
                 "cae_pool": "stride"}

    # Check "mode" and return parameters
    if mode == "cae":
        return cae_paras
    elif mode == "clf":
        return clf_paras
    else:
        raise ValueError("Select parameters in 'cae' or 'clf'.")

    return


if __name__ == "__main__":

    print(get_parameters("volume"))
    print(get_parameters("slice"))
