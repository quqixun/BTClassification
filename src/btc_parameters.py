# Brain Tumor Classification
# Script for Hyper-Parameters
# Author: Qixun Qu
# Create on: 2017/10/14
# Modify on: 2017/10/28

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
    - num_epoches: int, the number of epoches
    - learning_rate_first: float, the learning rate for first epoch
    - learning_rate_last: float, the learning rate for last epoch
    - more parameters to be added

-3- Parameters for Constructing Model
    - activation: string, indicates the activation method by either
      "relu" or "lrelu" (leaky relu) for general cnn models
    - alpha: float, slope of the leaky relu at x < 0
    - bn_momentum: float, momentum for removing average in batch
      normalization, typically values are 0.999, 0.99, 0.9, etc
    - drop_rate: float, rate of dropout of input units, which is
      between 0 and 1

'''


import os
import json
from btc_settings import *


'''
Parameters for General CNN Models
'''

# Set path of the folder in where tfrecords are save in
parent_dir = os.path.dirname(os.getcwd())
tfrecords_dir = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER)

# Create paths for training and validating tfrecords
tpath = os.path.join(tfrecords_dir, "partial_train.tfrecord")
vpath = os.path.join(tfrecords_dir, "partial_validate.tfrecord")

# Whole dataset
# tpath = os.path.join(tfrecords_dir, "train.tfrecord")
# vpath = os.path.join(tfrecords_dir, "validate.tfrecord")

# Load dict from json file in which the number of
# training and valdating set can be found
json_path = os.path.join(TEMP_FOLDER, TFRECORDS_FOLDER, VOLUMES_NUM_FILE)
with open(json_path) as json_file:
    volumes_num = json.load(json_file)

train_num = 236
validate_num = 224

# train_num = volumes_num["train"]
# validate_num = volumes_num["validate"]

cnn_parameters = {
    # Basic settings
    "train_path": tpath,
    "validate_path": vpath,
    "train_num": train_num,
    "validate_num": validate_num,
    "classes_num": 3,
    "patch_shape": PATCH_SHAPE,
    "capacity": 350,
    "min_after_dequeue": 300,
    # Parameters for training
    "batch_size": 10,
    "num_epoches": 1,
    "learning_rate_first": 1e-3,
    "learning_rate_last": 1e-4,
    "l2_loss_coeff": 0.001,
    # Parameter for model's structure
    "activation": "relu",  # "lrelu",
    "alpha": None,  # "lrelu"
    "bn_momentum": 0.99,
    "drop_rate": 0.5
}


'''
Parameters for Autoencoder Models
'''

cae_parameters = {
    # Basic settings
    "train_path": tpath,
    "validate_path": vpath,
    "train_num": train_num,
    "validate_num": validate_num,
    "classes_num": 3,
    "patch_shape": PATCH_SHAPE,
    "capacity": 350,
    "min_after_dequeue": 300,
    # Parameters for training
    "batch_size": 10,
    "num_epoches": 1,
    "learning_rate_first": 1e-3,
    "learning_rate_last": 1e-4,
    "l2_loss_coeff": 0.001,
    "sparse_penalty_coeff": 0.001,
    "sparse_level": 0.05,
    # Parameter for model's structure
    "activation": "relu",
    "bn_momentum": 0.99,
    "drop_rate": 0.5
}
