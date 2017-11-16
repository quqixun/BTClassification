# Brain Tumor Classification
# Script for Abstract Class for Training
# Author: Qixun Qu
# Create on: 2017/11/13
# Modify on: 2017/11/16

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

'''


from __future__ import print_function

import os
import json
import shutil
import numpy as np
import tensorflow as tf
from btc_settings import *
from btc_models import BTCModels
from btc_tfrecords import BTCTFRecords


class BTCTrain(object):

    def __init__(self, paras):
        '''__INIT__
        '''

        # Initialize BTCTFRecords to load data
        self.tfr = BTCTFRecords()

        self.dims = paras["dims"]
        self.train_path = paras["train_path"]
        self.validate_path = paras["validate_path"]
        self.classes_num = paras["classes_num"]
        self.patch_shape = paras["patch_shape"]
        self.capacity = paras["capacity"]
        self.min_after_dequeue = paras["min_after_dequeue"]

        self.batch_size = paras["batch_size"]
        self.num_epoches = np.sum(paras["num_epoches"])
        self.learning_rates = self._set_learning_rates(
            paras["num_epoches"], paras["learning_rates"])
        self.l2_loss_coeff = paras["l2_loss_coeff"]

        self.act = paras["activation"]
        self.alpha = self._get_parameter(paras, "alpha")
        self.bn_momentum = paras["bn_momentum"]
        self.drop_rate = paras["drop_rate"]

        self.cae_pool = self._get_parameter(paras, "cae_pool")
        self.sparse_type = self._get_parameter(paras, "sparse_type")
        self.kl_coeff = self._get_parameter(paras, "kl_coeff")
        self.p = self._get_parameter(paras, "sparse_level")
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
        return paras[name] if name in paras.keys() else None

    def set_net_name(self, net):
        return net + self.dims

    def set_dir_path(self, path, net_name):

        dir_path = os.path.join(path, net_name)

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
        # Load data from tfrecord files
        with tf.name_scope("tfrecords"):
            tra_data, tra_labels = self._load_tfrecord(self.train_path)
            val_data, val_labels = self._load_tfrecord(self.validate_path)

        return tra_data, tra_labels, val_data, val_labels

    def inputs(self):
        # Define inputs for model:
        # - features: 5D volume, shape in [batch_size, height, width, depth, channels]
        # - labels: 1D list, shape in [batch_size]
        # - training symbol: boolean
        with tf.name_scope("inputs"):
            x = tf.placeholder(tf.float32, [self.batch_size] + self.patch_shape, "volumes")
            y_input = tf.placeholder(tf.int64, [None], "labels")
            is_training = tf.placeholder(tf.bool, [], "mode")
            learning_rate = tf.placeholder_with_default(0.0, [], "learning_rate")

        tf.summary.scalar("learning rate", learning_rate)

        return x, y_input, is_training, learning_rate

    def _get_l2_loss(self, variables=None):
        '''_GET_L2_LOSS

            Compute l2 regularization term

        '''

        if variables is None:
            variables = tf.trainable_variables()

        return tf.add_n([tf.nn.l2_loss(v) for v in variables if "kernel" in v.name])

    def get_softmax_loss(self, y_in, y_out, variables=None):
        '''_GET_SOFTMAX_LOSS

            Compute loss, which consists of softmax cross entropy
            and l2 regularization term.

            Inputs:
            -------
            - y_in: tensor, input labels
            - y_out: tensor, model outputs

            Output:
            -------
            - śoftmax cross entropy + l2 loss

        '''

        # Compute softmax cross entropy
        def softmax_loss(y_in, y_out):
            # Convert labels to onehot array first, such as:
            # [0, 1, 2] ==> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            y_in_onehot = tf.one_hot(indices=y_in, depth=self.classes_num)
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in_onehot,
                                                                          logits=y_out))

        with tf.name_scope("loss"):
            # Regularization term to reduce overfitting
            loss = softmax_loss(y_in, y_out)
            loss += self._get_l2_loss(variables) * self.l2_loss_coeff

        # Add loss into summary
        tf.summary.scalar("loss", loss)

        return loss

    def get_mean_square_loss(self, y_in, y_out):
        '''GET_RECONSTRUCTION_LOSS

            Compute mean square loss between
            input and the reconstruction

        '''

        loss = self._get_l2_loss() * self.l2_loss_coeff
        loss += tf.div(tf.reduce_mean(tf.square(y_out - y_in)), 2)

        return loss

    def get_sparsity_loss(self, y_in, y_out, code):
        '''_GET_LOSS

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
            p_hat = tf.reduce_mean(code, axis=[1, 2, 3]) + 1e-8
            loss += tf.reduce_sum(sparse_penalty(self.p, p_hat)) * self.kl_coeff

        # Add loss into summary
        tf.summary.scalar("loss", loss)

        return loss

    def get_accuracy(self, y_in_labels, y_out):
        '''_GET_ACCURACY

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
        # Define initialization of graph
        with tf.name_scope("init"):
            init = tf.group(tf.local_variables_initializer(),
                            tf.global_variables_initializer())

        return init

    def create_writers(self, logs_path, graph):
        # Create writers to write logs in file
        tra_writer = tf.summary.FileWriter(os.path.join(logs_path, "train"), graph)
        val_writer = tf.summary.FileWriter(os.path.join(logs_path, "validate"), graph)

        return tra_writer, val_writer

    def print_metrics(self, stage, epoch_no, iters, loss, accuracy=None):
        '''_PRINT_METRICS

            Print metrics of each training and validating step.

            Inputs:
            -------
            - stage: string, "Train" or "Validate"
            - epoch_no: int, epoch number
            - iters: int, step number
            - loss: float, loss
            - accuracy: float, classification accuracy

        '''

        log_str = "[Epoch {}] ".format(epoch_no)
        log_str += (stage + " Step {}: ").format(iters)
        log_str += "Loss: {0:.10f}".format(loss)

        if accuracy is not None:
            log_str += ", Accuracy: {0:.10f}".format(accuracy)

        self.green_print(log_str)

        return

    def print_mean_metrics(self, stage, epoch_no, loss_list, accuracy_list=None):
        '''_PRINT_MEAN_METRICS

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
        log_str += "Mean Loss: {0:.10f}".format(loss_mean)

        if accuracy_list is not None:
            accuracy_mean = np.mean(accuracy_list)
            log_str += ", Mean Accuracy: {0:.10f}".format(accuracy_mean)

        self.yellow_print(log_str)

        return loss_mean

    def save_model_per_epoch(self, sess, saver, epoch_no):
        '''_SAVE_MODEL_PER_EPOCH
        '''

        # ckpt_dir = os.path.join(self.model_path, "epoch-" + str(epoch_no))
        # if os.path.isdir(ckpt_dir):
        #     shutil.rmtree(ckpt_dir)
        # os.makedirs(ckpt_dir)

        # Save model's graph and variables of each epoch into folder
        save_path = os.path.join(self.model_path, "model")
        saver.save(sess, save_path, global_step=None)

        log_str = "[Epoch {}] ".format(epoch_no)
        log_str += "Model was saved in: {}".format(self.model_path)
        self.cyan_print(log_str)

        return

    def save_metrics(self, filename, data):
        '''_SAVE_METRICS

            Save metrics (loss and accuracy) into pickle files.
            Durectiry has been set as self.logs_path.

            Inputs:
            -------
            - filename: string, the name of file, not the full path
            - data: 2D list, including loss and accuracy for either
                    training results or validating results

        '''

        metrics_num = len(np.shape(data))

        if metrics_num == 2:
            loss = ["{0:.6f}".format(d[0]) for d in data]
            accuracy = ["{0:.6f}".format(d[1]) for d in data]
            metrics = {"loss": loss, "accuracy": accuracy}
        elif metrics_num == 1:
            loss = ["{0:.6f}".format(d) for d in data]
            metrics = {"loss": loss}
        else:
            raise ValueError("Too many metrics to record.")

        json_path = os.path.join(self.logs_path, filename)
        if os.path.isfile(json_path):
            os.remove(txt_path)

        with open(json_path, "w") as json_file:
            json.dump(metrics, json_file)

        return

    def green_print(self, log_str):
        print(PCG + log_str + PCW)

    def yellow_print(self, log_str):
        print(PCY + log_str + PCW)

    def blue_print(self, log_str):
        print(PCB + log_str + PCW)

    def cyan_print(self, log_str):
        print(PCC + log_str + PCW)
