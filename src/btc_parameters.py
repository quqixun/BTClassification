# Brain Tumor Classification
# Script for Hyper-Parameters
# Author: Qixun Qu
# Create on: 2017/10/14
# Modify on: 2017/10/20

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

-2- Paraneters for Training:
    - batch_size: int, the number of patches in one batch
    - num_epoches: int, the number of epoches
    - more parameters to be added

-3- Parameter for Constructing Model
    - activation: string, indicate the activation method by either
      "relu" or "lrelu" (leaky relu)
    - alpha: float, slope of the leaky relu at x < 0
    - bn_momentum: float, momentum for removing average in batch
      normalization, typically values are 0.999, 0.99, 0.9, etc
    - drop_rate: float, rate of dropout of input units, which is
      between 0 and 1

'''


import os
import json
from btc_settings import *


# Set path of the folder in where tfrecords are save in
parent_dir = os.path.dirname(os.getcwd())
tfrecords_dir = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER)

# Create paths for training and validating tfrecords
tpath = os.path.join(tfrecords_dir, "partial_train.tfrecord")
vpath = os.path.join(tfrecords_dir, "partial_validate.tfrecord")

# Load dict from json file in which the number of
# training and valdating set can be found
json_path = os.path.join(TEMP_FOLDER, TFRECORDS_FOLDER, VOLUMES_NUM_FILE)
with open(json_path) as json_file:
    volumes_num = json.load(json_file)

parameters = {
    # Basic settings
    "train_path": tpath,
    "validate_path": vpath,
    "train_num": 236,  # volumes_num["train"],
    "validate_num": 224,  # volumes_num["validate"],
    "classes_num": 3,
    "patch_shape": PATCH_SHAPE,
    "capacity": 350,
    "min_after_dequeue": 300,
    # Parameters for training
    "batch_size": 10,
    "num_epoches": 1,
    # Parameter for model's structure
    "activation": "relu",  # "lrelu",
    "alpha": 0.2,  # "lrelu"
    "drop_rate": 0.5,
    "bn_momentum": 0.99,
    "drop_rate": 0.5
}
