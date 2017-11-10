# Brain Tumor Classification
# Script for CAEs' Hyper-Parameters
# Author: Qixun Qu
# Create on: 2017/11/06
# Modify on: 2017/11/10

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
    - sparse_penalty_coeff: float, coefficient of sparse penalty term
    - sparse_level: float, sparsity parameter
    - more parameters to be added

-3- Parameters for Constructing Model
    - activation: string, indicates the activation method by either
                  "relu" or "sigmoid" for autoencoder
    - bn_momentum: float, momentum for removing average in batch
                   normalization, typically values are 0.999, 0.99, 0.9, etc
    - drop_rate: float, rate of dropout of input units, which is
                 between 0 and 1

'''


import os
import json
import math
from btc_settings import *


'''
Parameters for Autoencoder Models
'''

# Set path of the folder in where tfrecords are save in
parent_dir = os.path.dirname(os.getcwd())
tfrecords_dir = os.path.join(parent_dir, DATA_FOLDER,
                             TFRECORDS_FOLDER, VOLUMES_FOLDER)

# Create paths for training and validating tfrecords
tpath = os.path.join(tfrecords_dir, "train.tfrecord")
vpath = os.path.join(tfrecords_dir, "validate.tfrecord")

# Load dict from json file in which the number of
# training and valdating set can be found
json_path = os.path.join(TEMP_FOLDER, TFRECORDS_FOLDER,
                         VOLUMES_FOLDER, DATA_NUM_FILE)
with open(json_path) as json_file:
    data_num = json.load(json_file)

train_num = data_num["train"]
validate_num = data_num["validate"]

# Settings for partial dataset to test
# tpath = os.path.join(tfrecords_dir, "partial_train.tfrecord")
# vpath = os.path.join(tfrecords_dir, "partial_validate.tfrecord")
# train_num = 3
# validate_num = 3

# Settings for decodeing tfrecords
min_after_dequeue = max([train_num, validate_num])
capacity = math.ceil(min_after_dequeue * 1.1)

parameters = {
    # Basic settings
    "dims": "3D",
    "train_path": tpath,
    "validate_path": vpath,
    "train_num": train_num,
    "validate_num": validate_num,
    "classes_num": 3,
    "patch_shape": VOLUME_SHAPE,
    "capacity": capacity,
    "min_after_dequeue": min_after_dequeue,
    # Parameters for training
    "batch_size": 1,
    "num_epoches": [10, 10, 10],
    "learning_rates": [1e-3, 1e-4, 1e-5],
    # "learning_rate_first": 1e-3,
    # "learning_rate_last": 1e-4,
    "l2_loss_coeff": 0.001,
    "sparse_penalty_coeff": 0.001,
    "sparse_level": 0.05,
    # Parameter for model's structure
    "activation": "relu",  # "sigmoid"
    "bn_momentum": 0.99,
    "drop_rate": 0.5
}
