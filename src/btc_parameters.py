# Brain Tumor Classification
# Script for Hyper Parameters
# Author: Qixun Qu
# Create on: 2017/10/14
# Modify on: 2017/10/18

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


import os
import json
from btc_settings import *


parent_dir = os.path.dirname(os.getcwd())
tfrecords_dir = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER)

tpath = os.path.join(tfrecords_dir, "partial_train.tfrecord")
vpath = os.path.join(tfrecords_dir, "partial_validate.tfrecord")

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
    # Hyper-parameters
    "batch_size": 10,
    "num_epoches": 1,
    "activation": "relu",  # "lrelu",
    "alpha": 0.2  # for "lrelu"
}
