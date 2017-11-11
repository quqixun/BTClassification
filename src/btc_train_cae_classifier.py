# Brain Tumor Classification
# Script for Training Classifier
# for Autoencoders
# Author: Qixun Qu
# Create on: 2017/11/11
# Modify on: 2017/11/11

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

Class BTCTrainCAEClassifier

-1- Models for classifier are defined in class BTCModels.
-2- Hyper-parameters for classofoer can be set in btc_cae_parameters.py.
-3- Loading tfrecords for training and validating by
    functions in class BTCTFRecords.

'''


from __future__ import print_function

import os
import json
import shutil
import argparse
import numpy as np
import tensorflow as tf
from btc_settings import *
from btc_models import BTCModels
from btc_tfrecords import BTCTFRecords
from btc_cae_parameters import get_parameters


class BTCTrainCAEClassier():

    def __init__(self, paras, save_path, logs_path):
        '''__INIT__
        '''

        pool = paras["cae_pool"]
        self.net = "cae_" + pool
        if (self.net != CAE_STRIDE) and (self.net != CAE_POOL):
            raise ValueError("Pool method should be 'stride' or 'pool'.")

        # Initialize BTCTFRecords to load data
        self.tfr = BTCTFRecords()

        # Create folders to keep models
        # if the folder is not exist
        self.model_path = os.path.join(save_path, self.net)
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        # Create folders to keep models
        # if the folder is not exist
        self.logs_path = os.path.join(logs_path, self.net)
        if os.path.isdir(self.logs_path):
            shutil.rmtree(self.logs_path)
        os.makedirs(self.logs_path)

        # Basic settings
        self.train_path = paras["train_path"]
        self.validate_path = paras["validate_path"]
        self.classes_num = paras["classes_num"]
        self.patch_shape = paras["patch_shape"]
        self.capacity = paras["capacity"]
        self.min_after_dequeue = paras["min_after_dequeue"]

        # For training process
        self.batch_size = paras["batch_size"]
        self.num_epoches = np.sum(paras["num_epoches"])
        self.learning_rates = self._get_learning_rates(
            paras["num_epoches"], paras["learning_rates"])
        # self.learning_rates = self._get_learning_rates_decay(
        #     paras["learning_rate_first"], paras["learning_rate_last"])
        self.l2_loss_coeff = paras["l2_loss_coeff"]

        # For models' structure
        act = paras["activation"]
        alpha = paras["alpha"]
        bn_momentum = paras["bn_momentum"]
        drop_rate = paras["drop_rate"]
        dims = paras["dims"]

        # Initialize BTCModels to set general settings
        self.models = BTCModels(self.classes_num, act, alpha,
                                bn_momentum, drop_rate, dims, pool)
        self.network = self.models.autoencoder_classier

        # Computer the number of batches in each epoch for
        # both training and validating respectively
        self.tepoch_iters = self._get_epoch_iters(paras["train_num"])
        self.vepoch_iters = self._get_epoch_iters(paras["validate_num"])

        # Create empty lists to save loss and accuracy
        self.train_metrics, self.validate_metrics = [], []

        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    help_str = "Select a data in 'volume' or 'slice'."
    parser.add_argument("--data", action="store", dest="data", help=help_str)
    args = parser.parse_args()

    parent_dir = os.path.dirname(os.getcwd())
    save_path = os.path.join(parent_dir, "models")
    logs_path = os.path.join(parent_dir, "logs")

    parameters = get_parameters(args.data, "clf")

    btc = BTCTrainCAEClassifier(parameters, save_path, logs_path)
    btc.train()
