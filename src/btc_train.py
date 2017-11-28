# Brain Tumor Classification
# Script for Abstract Class for Training
# Author: Qixun Qu
# Create on: 2017/11/13
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

Class BTCTrain

The parent class for:
- BTCTrainCNN
- BTCTrainCAE
- BTCTrainCAEClassifier

In this class, functions are provided to train
models with different structures, including:
- load_data: load data for training and validating
             from tfrecords files
- inputs: declear placeholders
- get_softmax_loss: compute loss for general cnn models
- get_sparsity_loss: compute loss for sparsity autoencoder
                     with KL-dicengence constraint
- get_mean_square_loss: compute reconstruction loss for
                        autoencoder classifier
- get_accuracy: compute classification accuracy
- create_optimizer: create optimizer to minimize loss
- initialize_variables: initialize variables of filters

'''


from __future__ import print_function

import os
import json
import time
import shutil
import numpy as np
import tensorflow as tf
from btc_settings import *
from btc_models import BTCModels
from btc_tfrecords import BTCTFRecords


class BTCTrain(object):

    def __init__(self, paras):
        '''__INIT__

            Initialize parameters from given dictionary.

            Input:
            ------
            - paras: dict, can be found in btc_cnn_parameters.py
                     and btc_cae_parameters.py

        '''

        # Initialize BTCTFRecords to load data
        self.tfr = BTCTFRecords()

        # Setting for input
        # Dimentions of input
        self.dims = paras["dims"]
        # Path for tfrecord of training set
        self.train_path = paras["train_path"]
        # Path for tfrecords of validating set
        self.validate_path = paras["validate_path"]
        # The number of classes
        self.classes_num = paras["classes_num"]
        # Input tensor's shape
        self.patch_shape = paras["patch_shape"]
        # Parameters for loading tfrecords
        self.capacity = paras["capacity"]
        self.min_after_dequeue = paras["min_after_dequeue"]

        # Training settings
        self.batch_size = paras["batch_size"]
        self.num_epoches = np.sum(paras["num_epoches"])
        self.learning_rates = self._set_learning_rates(
            paras["num_epoches"], paras["learning_rates"])
        self.l2_loss_coeff = paras["l2_loss_coeff"]

        # Settings for constructing models
        self.act = paras["activation"]
        self.alpha = self._get_parameter(paras, "alpha")
        self.bn_momentum = paras["bn_momentum"]
        self.drop_rate = paras["drop_rate"]

        # Settings for autoencoder
        self.cae_pool = self._get_parameter(paras, "cae_pool")
        self.sparse_type = self._get_parameter(paras, "sparse_type")
        # KL constraint
        self.kl_coeff = self._get_parameter(paras, "kl_coeff")
        self.p = self._get_parameter(paras, "sparse_level")
        # Winner-Take-All constraint
        self.k = self._get_parameter(paras, "winner_nums")
        self.lifetime_rate = self._get_parameter(paras, "lifetime_rate")

        # Initialize BTCModels to set general settings
        self.models = BTCModels(self.classes_num, self.act, self.alpha,
                                self.bn_momentum, self.drop_rate,
                                self.dims, self.cae_pool, self.lifetime_rate)

        # Computer the number of batches in each epoch for
        # both training and validating respectively
        self.tepoch_iters = self._get_epoch_iters(paras["train_num"])
        self.vepoch_iters = self._get_epoch_iters(paras["validate_num"])

        # Create empty lists to save loss and accuracy
        self.train_metrics, self.validate_metrics = [], []

        return

    def _get_parameter(self, paras, name):
        '''_GET_PARAMETER

            Extract value from dictionary according to the
            given name. If the name is not exist, return None.

            Inputs:
            -------
            - paras: dict, parameters
            - name: string, attribute's name you want to get

            Output:
            - value of attribute or None

        '''

        return paras[name] if name in paras.keys() else None

    def set_net_name(self, net):
        '''SET_NET_NAME

            Concatenate string to set net's name.

            Input:
            -----
            - net: string, the original net's name

            Output:
            -------
            - new name of the net witch indicates
              the dimentions of input data

        '''

        net_name = net + "_" + self.dims

        if self.cae_pool is not None:
            net_name += "_" + self.cae_pool

        if self.sparse_type is not None:
            net_name += "_" + self.sparse_type

        return net_name

    def set_dir_path(self, path, net_name):
        '''SET_DIR_NAME

            Generate folders to save models and logs.

            Inputs:
            -------
            - path: string, the path of parent directory
                    of models and logs
            - net_name: string, net's name

            Output:
            -------
            - the path of directory

        '''

        dir_path = os.path.join(path, net_name)

        # If the directory is exist, remove all contents
        # and create a new empty folder
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        return dir_path

    def _set_learning_rates(self, num_epoches, learning_rates):
        '''_GET_LEARNING_RATES

            Compute learning rate for each epoch according to
            the given learning rates.

            Inputs:
            -------
            - num_epoches: list of ints, indicates the number of epoches
                           that share the same learning rate
            - learning_rates: list of floats, gives the learning rates for
                           different training epoches

            Outputs:
            --------
            - a list of learning rates

            Example:
            --------
            - num_epoches: [2, 2]
            - learning_rates: [1e-3, 1e-4]
            - return: [1e-3, 1e-3, 1e-4, 1e-4]

        '''

        if len(num_epoches) != len(learning_rates):
            raise ValueError("len(num_epoches) should equal to len(learning_rates).")

        learning_rate_per_epoch = []
        for n, l in zip(num_epoches, learning_rates):
            learning_rate_per_epoch += [l] * n

        return learning_rate_per_epoch

    def _set_learning_rates_decay(self, first_rate, last_rate):
        '''_GET_LEARNING_RATES_DECAY

            Compute learning rate for each epoch according to
            the start and the end point.

            Inputs:
            -------
            - first_rate: float, the learning rate for first epoch
            - last_rate: float, the learning rate for last epoch

            Outputs:
            --------
            - a list of learning rates

        '''

        learning_rates = [first_rate]

        if self.num_epoches == 1:
            return learning_rates

        decay_step = (first_rate - last_rate) / (self.num_epoches - 1)
        for i in range(1, self.num_epoches - 1):
            learning_rates.append(first_rate - decay_step * i)
        learning_rates.append(last_rate)

        return learning_rates

    def _get_epoch_iters(self, data_num):
        '''_GET_EPOCH_ITERS

            The helper funtion to compute the number of iterations
            of each epoch.

            Input:
            -------
            - data_num: int, the number of patches in dataset

            Output:
            -------
            - a list consists of iterations in each epoch

        '''

        index = np.arange(1, self.num_epoches + 1)
        iters_per_epoch = np.floor(index * (data_num / self.batch_size))

        return iters_per_epoch.astype(np.int64)

    def _load_tfrecord(self, tfrecord_path):
        '''_LOAD_DATA

            The helper funtion to load patches from tfrecord files.
            All patches are suffled abd returned in batch size.

            Input:
            ------
            - tfrecord_path: string, the path fo tfrecord file

            Output:
            -------
            - suffled patches in batch size

        '''

        return self.tfr.decode_tfrecord(path=tfrecord_path,
                                        batch_size=self.batch_size,
                                        num_epoches=self.num_epoches,
                                        patch_shape=self.patch_shape,
                                        capacity=self.capacity,
                                        min_after_dequeue=self.min_after_dequeue)

    def load_data(self):
        '''LOAD_DATA

            Load training data and validating data
            from tfrecord files.

            Outputs:
            - tra_data, val_data: shuffled data
            - tra_labels, val_labels: data labels

        '''

        with tf.name_scope("tfrecords"):
            tra_data, tra_labels = self._load_tfrecord(self.train_path)
            val_data, val_labels = self._load_tfrecord(self.validate_path)

        return tra_data, tra_labels, val_data, val_labels

    def inputs(self):
        '''INPUTS

            Create placeholders that will be input into the model.
            - x: data, 5D tensor, shape in [batch_size, height, width, depth, channels],
                 or 4D tensor, shape in [batch_size, height, width, channels]
            - y_input: labels for x
            - is_training: a symbol to describe mode, True for training mode,
                           False for validating and testing mode
            - learning_rate: the learning rate for one epoch

        '''

        with tf.name_scope("inputs"):
            x = tf.placeholder(tf.float32, [self.batch_size] + self.patch_shape, "volumes")
            y_input = tf.placeholder(tf.int64, [None], "labels")
            is_training = tf.placeholder(tf.bool, [], "mode")
            learning_rate = tf.placeholder_with_default(0.0, [], "learning_rate")

        # Add learning rate into observation
        tf.summary.scalar("learning rate", learning_rate)

        return x, y_input, is_training, learning_rate

    def _get_l2_loss(self, variables=None):
        '''_GET_L2_LOSS

            Compute l2 regularization term.

            Input:
            ------
            - variables: list of tensors, indicates l2 loss
                         computed from which variables

            Output:
            - le regularization term

        '''

        # Use all trainable variables
        if variables is None:
            variables = tf.trainable_variables()

        return tf.add_n([tf.nn.l2_loss(v) for v in variables if "kernel" in v.name])

    def get_softmax_loss(self, y_in, y_out, variables=None):
        '''GET_SOFTMAX_LOSS

            Compute loss, which consists of softmax cross entropy
            and l2 regularization term.

            Inputs:
            -------
            - y_in: tensor, input labels
            - y_out: tensor, model outputs
            - variables: list of tensors, indicates l2 loss
                         computed from which variables

            Output:
            -------
            - softmax cross entropy + l2 loss

        '''

        # Compute softmax cross entropy
        def softmax_loss(y_in, y_out):
            # Convert labels to onehot array first, such as:
            # [0, 1, 2] ==> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            # y_in_onehot = tf.one_hot(indices=y_in, depth=self.classes_num)
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_in,
                                                                                 logits=y_out))

        with tf.name_scope("loss"):
            loss = softmax_loss(y_in, y_out)
            # Regularization term to reduce overfitting
            loss += self._get_l2_loss(variables) * self.l2_loss_coeff

        # Add loss into summary
        tf.summary.scalar("loss", loss)

        return loss

    def get_mean_square_loss(self, y_in, y_out):
        '''GET_MEAN_SQUARE_LOSS

            Compute mean square loss between
            input and the reconstruction.

            Inputs:
            -------
            - y_in: tensor, original input
            - y_out: tensor, reconstruction

            Output:
            - mean squqre loss

        '''

        loss = self._get_l2_loss() * self.l2_loss_coeff
        loss += tf.div(tf.reduce_mean(tf.square(y_out - y_in)), 2)

        return loss

    def get_sparsity_loss(self, y_in, y_out, code):
        '''GET_SPARSITY_LOSS

            Compute loss, which consists of mean square loss,
            l2 regularization term and sparsity penalty term.

            Inputs:
            -------
            - y_in: tensor, original input to train the model
            - y_out: tensor, the reconstruction of input
            - code: tensor, compress presentation of input

            Output:
            -------
            - sparcity loss + reconstruction loss + l2 loss

        '''

        # Compute sparsity penalty term
        def sparse_penalty(p, p_hat):
            sp1 = p * (tf.log(p) - tf.log(p_hat))
            sp2 = (1 - p) * (tf.log(1 - p) - tf.log(1 - p_hat))
            return sp1 + sp2

        with tf.name_scope("loss"):
            loss = self.get_mean_square_loss(y_in, y_out)
            # Average value of activated code
            p_hat = tf.reduce_mean(code, axis=[1, 2, 3]) + 1e-8
            loss += tf.reduce_sum(sparse_penalty(self.p, p_hat)) * self.kl_coeff

        # Add loss into summary
        tf.summary.scalar("loss", loss)

        return loss

    def get_accuracy(self, y_in_labels, y_out):
        '''GET_ACCURACY

            Compute accuracy of classification.

            Inputs:
            -------
            - y_in_labels: tensor, labels for input cases
            - y_out: tensor, output from model

            Output:
            -------
            - classification accuracy

        '''

        with tf.name_scope("accuracy"):
            # Obtain the predicted labels for each input example first
            y_out_labels = tf.argmax(input=y_out, axis=1)
            correction_prediction = tf.equal(y_out_labels, y_in_labels)
            accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

        # Add accuracy into summary
        tf.summary.scalar("accuracy", accuracy)

        return accuracy

    def create_optimizer(self, learning_rate, loss, var_list=None):
        '''CREATE_OPTIMIZER

            Create optimizer to minimize loss:

            inputs:
            -------
            - learning_rate: float, learning rate of one training epoch
            - loss: the loss needs to be minimized
            - var_list: variables to be updated

            Ouput:
            ------
            - the optimizer

        '''

        # Optimize loss
        with tf.name_scope("train"):
            # Update moving_mean and moving_variance of
            # batch normalization in training process
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate)
                train_op = train_op.minimize(loss, var_list=var_list)

        return train_op

    def initialize_variables(self):
        '''INITIALIZE_VARIABLES

            Initialize global and local variables.

        '''

        with tf.name_scope("init"):
            init = tf.group(tf.local_variables_initializer(),
                            tf.global_variables_initializer())

        return init

    def create_writers(self, logs_path, graph):
        '''CREATE_WRITERS

            Create writers to save logs in given path.

            Inputs:
            -------
            - logs_path: string, the path of directory to save logs
            - graph: a computation graph

            Outputs:
            - writers for training process and validating process

        '''

        # Create writers to write logs in file
        tra_writer = tf.summary.FileWriter(os.path.join(logs_path, "train"), graph)
        val_writer = tf.summary.FileWriter(os.path.join(logs_path, "validate"), graph)

        return tra_writer, val_writer

    def print_metrics(self, stage, epoch_no, iters, rtime, loss, accuracy=None):
        '''PRINT_METRICS

            Print metrics of each training and validating step.

            Inputs:
            -------
            - stage: string, "Train" or "Validate"
            - epoch_no: int, epoch number
            - iters: int, step number
            - rtime: string, time cost of one step
            - loss: float, loss
            - accuracy: float, classification accuracy

        '''

        log_str = "[Epoch {}] ".format(epoch_no)
        log_str += (stage + " Step {}: ").format(iters)
        log_str += "Loss: {0:.6f}".format(loss)

        if accuracy is not None:
            log_str += ", Accuracy: {0:.6f}".format(accuracy)

        log_str += ", Time Cost: " + rtime

        self.green_print(log_str)

        return

    def print_mean_metrics(self, stage, epoch_no, loss_list, accuracy_list=None):
        '''PRINT_MEAN_METRICS

            Print mean metrics after each training and validating epoch.

            Inputs:
            -------
            - stage: string, "Train" or "Validate"
            - epoch_no: int, epoch number
            - loss_list: list of floats, which keeps loss of each step
                         inner one training or validating epoch
            - accuracy_list: list of floats, which keeps accuracy of each
                             step inner one training or validating epoch

        '''

        loss_mean = np.mean(loss_list)

        log_str = "[Epoch {}] ".format(epoch_no)
        log_str += stage + " Stage: "
        log_str += "Mean Loss: {0:.6f}".format(loss_mean)

        if accuracy_list is not None:
            accuracy_mean = np.mean(accuracy_list)
            log_str += ", Mean Accuracy: {0:.6f}".format(accuracy_mean)

        self.yellow_print(log_str)

        return loss_mean

    def print_time(self, epoch_no, epoch_time):
        '''PRINT_TIME

            Print time cost of one epoch, insluding
            training and validating steps.

            Inputs:
            -------
            - epoch_no: int, epoch number
            - epoch_time: string, time cost of one epoch

        '''

        time_str = "[Epoch {}] ".format(epoch_no)
        time_str += "Time Cost: " + epoch_time

        self.yellow_print(time_str)

        return

    def save_model_per_epoch(self, sess, saver, epoch_no, mode):
        '''SAVE_MODEL_PER_EPOCH

            Save model into checkpoint.

            Inputs:
            -------
            - sess: the session of training
            - saver: the saver created before training
            - epoch_no: int, epoch number
            - mode: string, "best" or "last"

        '''

        # Create directory to save model
        save_dir = os.path.join(self.model_path, mode)
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        # Save model's graph and variables of each epoch into folder
        save_path = os.path.join(save_dir, "model")
        saver.save(sess, save_path, global_step=None)

        log_str = "[Epoch {}] ".format(epoch_no)
        log_str += mode.capitalize() + " Model was saved in: " + save_dir
        self.cyan_print(log_str)

        return

    def save_metrics(self, filename, data):
        '''SAVE_METRICS

            Save metrics (loss and accuracy) into pickle files.
            Durectiry has been set as self.logs_path.

            Inputs:
            -------
            - filename: string, the name of file, not the full path
            - data: 2D list, including loss and accuracy for either
                    training results or validating results

        '''

        metrics_num = len(np.shape(data))

        if metrics_num == 2:  # Input metrics have loss and accuracy
            loss = ["{0:.6f}".format(d[0]) for d in data]
            accuracy = ["{0:.6f}".format(d[1]) for d in data]
            metrics = {"loss": loss, "accuracy": accuracy}
        elif metrics_num == 1:  # Input metrics only have loss
            loss = ["{0:.6f}".format(d) for d in data]
            metrics = {"loss": loss}
        else:
            raise ValueError("Too many metrics to record.")

        # Save metrics into json file
        json_path = os.path.join(self.logs_path, filename)
        if os.path.isfile(json_path):
            os.remove(txt_path)

        with open(json_path, "w") as json_file:
            json.dump(metrics, json_file)

        return

    #
    # Helper functions to print information in color
    #

    def green_print(self, log_str):
        print(PCG + log_str + PCW)

    def yellow_print(self, log_str):
        print(PCY + log_str + PCW)

    def blue_print(self, log_str):
        print(PCB + log_str + PCW)

    def cyan_print(self, log_str):
        print(PCC + log_str + PCW)

    #
    # Helper function of timer
    #

    def get_time(self, start_time):
        '''GET_TIME

            Obtain the time interval between
            the given start time and now.

            Inputs:
            -------
            - start_time: float, start time

            Output:
            -------
            - a string of time interval with unit

        '''

        # Default unit is "seconds"
        time_num = time.time() - start_time
        time_str = "{0:.3f}s".format(time_num)

        # Change unit to "minutes"
        if time_num >= 60:
            time_str = "{0:.3f}m".format(time_num / 60)

        # Change unit to "hours"
        if time_num >= 3600:
            time_str = "{0:.3f}h".format(time_num / 3600)

        return time_str
