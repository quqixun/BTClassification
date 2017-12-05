# Brain Tumor Classification
# Script for CNNs' Hyper-Parameters
# Author: Qixun Qu
# Create on: 2017/10/14
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
    - l2_loss_coeff: float, coeddicient of le regularization item

-3- Parameters for Constructing Model
    - activation: string, indicates the activation method by either
                  "relu" or "lrelu" (leaky relu) for general cnn models
    - alpha: float, slope of the leaky relu at x < 0
    - bn_momentum: float, momentum for removing average in batch
                   normalization, typically values are 0.999, 0.99, 0.9 etc
    - drop_rate: float, rate of dropout of input units, which is
                 between 0 and 1

'''


import os
import json
import math
from btc_settings import *


'''
Parameters for General CNN Models
'''

# Set path of the folder in where tfrecords are save in
parent_dir = os.path.dirname(os.getcwd())
tfrecords_dir = os.path.join(parent_dir, DATA_FOLDER,
                             TFRECORDS_FOLDER, PATCHES_FOLDER)

# Create paths for training and validating tfrecords
tpath = os.path.join(tfrecords_dir, "dataset1.tfrecord")
vpath = os.path.join(tfrecords_dir, "dataset2.tfrecord")

# Load dict from json file in which the number of
# training and valdating set can be found
json_path = os.path.join(TEMP_FOLDER, TFRECORDS_FOLDER,
                         PATCHES_FOLDER, DATA_NUM_FILE)
with open(json_path) as json_file:
    data_num = json.load(json_file)

train_num = data_num["dataset1"]
validate_num = data_num["dataset2"]

# Settings for decodeing tfrecords
min_after_dequeue = max([train_num, validate_num])
capacity = math.ceil(min_after_dequeue * 1.1)

cnn_parameters = {
    # Basic settings
    "dims": "3D",
    "train_path": tpath,
    "validate_path": vpath,
    "train_num": train_num,
    "validate_num": validate_num,
    "classes_num": 3,
    "patch_shape": PATCH_SHAPE,
    "capacity": capacity,
    "min_after_dequeue": min_after_dequeue,
    # Parameters for training
    "batch_size": 32,
    "num_epoches": 100,  # [40, 30, 20, 10],
    "learning_rates": [1e-3, 1e-4, 1e-5, 1e-6],
    "learning_rate_first": 1e-4,
    "learning_rate_last": 1e-6,
    "l2_loss_coeff": 0.001,
    # Parameter for model's structure
    "activation": "lrelu",  # "lrelu",
    "alpha": 0.333,  # "lrelu"
    "bn_momentum": 0.99,
    "drop_rate": 0.8
}
