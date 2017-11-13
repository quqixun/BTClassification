# Brain Tumor Classification
# Script for Training Classifier
# for Autoencoders
# Author: Qixun Qu
# Create on: 2017/11/11
# Modify on: 2017/11/13

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


class BTCTrainCAEClassifier():

    def __init__(self, paras, save_path, logs_path):
        '''__INIT__
        '''

        pool = paras["cae_pool"]
        dims = paras["dims"]
        self.net = "cae_" + pool + dims
        self.clfier = self.net + "_clf"
        self.coder_path = os.path.join(save_path, self.net, "model")

        # Initialize BTCTFRecords to load data
        self.tfr = BTCTFRecords()

        # Create folders to keep models
        # if the folder is not exist
        self.model_path = os.path.join(save_path, self.clfier)
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        # Create folders to keep logs
        # if the folder is not exist
        self.logs_path = os.path.join(logs_path, self.clfier)
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

    def _get_epoch_iters(self, num):
        '''_GET_EPOCH_ITERS

            The helper funtion to compute the number of iterations
            of each epoch.

            Input:
            -------
            - num: int, the number of patches in dataset

            Output:
            -------
            - a list consists of iterations in each epoch

        '''

        index = np.arange(1, self.num_epoches + 1)
        iters_per_epoch = np.floor(index * (num / self.batch_size))

        return iters_per_epoch.astype(np.int64)

    def _get_learning_rates(self, num_epoches, learning_rates):
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

    def _load_data(self, tfrecord_path):
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

    def _get_loss(self, y_in, y_out):
        '''_GET_LOSS

            Compute loss, which consists of softmax cross entropy
            and l2 regularization term.

            Inputs:
            -------
            - y_in: tensor, input labels
            - y_out: tensor, model outputs

            Output:
            -------
            - loss

        '''

        # Compute softmax cross entropy
        def softmax_loss(y_in, y_out):
            # Convert labels to onehot array first, such as:
            # [0, 1, 2] ==> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            y_in_onehot = tf.one_hot(indices=y_in, depth=self.classes_num)
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in_onehot,
                                                                          logits=y_out))

        # Compute l2 regularization term
        def l2_loss():
            variables = tf.trainable_variables()
            return tf.add_n([tf.nn.l2_loss(v) for v in variables if "kernel" in v.name])

        with tf.name_scope("loss"):
            # Regularization term to reduce overfitting
            loss = softmax_loss(y_in, y_out)
            loss += l2_loss() * self.l2_loss_coeff

        # Add loss into summary
        tf.summary.scalar("loss", loss)

        return loss

    def _get_accuracy(self, y_in_labels, y_out):
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

    def _print_metrics(self, stage, epoch_no, iters, loss, accuracy):
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

        print((PCG + "[Epoch {}] ").format(epoch_no),
              (stage + " Step {}: ").format(iters),
              "Loss: {0:.10f}, ".format(loss),
              ("Accuracy: {0:.10f}" + PCW).format(accuracy))

        return

    def _print_mean_metrics(self, stage, epoch_no, loss_list, accuracy_list):
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
        accuracy_mean = np.mean(accuracy_list)

        print((PCY + "[Epoch {}] ").format(epoch_no),
              stage + " Stage: ",
              "Mean Loss: {0:.10f}, ".format(loss_mean),
              ("Mean Accuracy: {0:.10f}" + PCW).format(accuracy_mean))

        return loss_mean

    def _save_model_per_epoch(self, sess, saver, epoch_no):
        '''_SAVE_MODEL_PER_EPOCH
        '''

        # ckpt_dir = os.path.join(self.model_path, "epoch-" + str(epoch_no))
        # if os.path.isdir(ckpt_dir):
        #     shutil.rmtree(ckpt_dir)
        # os.makedirs(ckpt_dir)

        # Save model's graph and variables of each epoch into folder
        save_path = os.path.join(self.model_path, "model")
        saver.save(sess, save_path, global_step=None)
        print((PCC + "[Epoch {}] ").format(epoch_no),
              ("Model was saved in: {}" + PCW).format(self.model_path))

        return

    def train(self):
        '''TRAIN

            Train and validate the choosen model.

        '''

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

        # with tf.device("/gpu:0")
        # Obtain logits from the model
        output = self.network(x, is_training)

        # Compute loss and accuracy
        loss = self._get_loss(y_input, output)
        accuracy = self._get_accuracy(y_input, output)

        # Merge summary
        # The summary can be displayed by TensorBoard
        merged = tf.summary.merge_all()

        # Create a saver to save model while training
        variables = tf.trainable_variables()
        coder_vars = [v for v in variables if "conv" in v.name]
        logit_vars = [v for v in variables if "logits" in v.name]
        loader = tf.train.Saver(coder_vars)
        saver = tf.train.Saver(logit_vars)

        # Optimize loss
        with tf.name_scope("train"):
            # Update moving_mean and moving_variance of
            # batch normalization in training process
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=logit_vars)

        # with tf.device("/cpu:0")
        # Load data from tfrecord files
        with tf.name_scope("tfrecords"):
            tra_volumes, tra_labels = self._load_data(self.train_path)
            val_volumes, val_labels = self._load_data(self.validate_path)

        # Define initialization of graph
        with tf.name_scope("init"):
            init = tf.group(tf.local_variables_initializer(),
                            tf.global_variables_initializer())

        sess = tf.InteractiveSession()

        # Create writers to write logs in file
        tra_writer = tf.summary.FileWriter(os.path.join(self.logs_path, "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(self.logs_path, "validate"), sess.graph)

        sess.run(init)
        loader.restore(sess, self.coder_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print((PCB + "\nTraining and Validating model: {}\n" + PCW).format(self.clfier))

        # Initialize counter
        tra_iters, val_iters, epoch_no = 0, 0, 0
        one_tra_iters, one_val_iters = 0, 0
        best_val_lmean_oss = np.inf

        # Lists to save loss and accuracy of each training step
        tloss_list, taccuracy_list = [], []

        try:
            while not coord.should_stop():
                # Training step
                # Feed the graph, run optimizer and get metrics
                tx, ty = sess.run([tra_volumes, tra_labels])
                tra_fd = {x: tx, y_input: ty, is_training: True, learning_rate: self.learning_rates[epoch_no]}
                tsummary, tloss, taccuracy, _ = sess.run([merged, loss, accuracy, train_op], feed_dict=tra_fd)

                tra_iters += 1
                one_tra_iters += 1

                # Record metrics of training steps
                tloss_list.append(tloss)
                taccuracy_list.append(taccuracy)
                tra_writer.add_summary(tsummary, tra_iters)
                self.train_metrics.append([tloss, taccuracy])
                self._print_metrics("Train", epoch_no + 1, one_tra_iters, tloss, taccuracy)

                if tra_iters % self.tepoch_iters[epoch_no] == 0:
                    # Validating step
                    # Lists to save loss and accuracy of each validating step
                    vloss_list, vaccuracy_list = [], []
                    while val_iters < self.vepoch_iters[epoch_no]:
                        # Feed the graph, get metrics
                        vx, vy = sess.run([val_volumes, val_labels])
                        val_fd = {x: vx, y_input: vy, is_training: False}
                        vsummary, vloss, vaccuracy = sess.run([merged, loss, accuracy], feed_dict=val_fd)

                        val_iters += 1
                        one_val_iters += 1

                        # Record metrics of validating steps
                        vloss_list.append(vloss)
                        vaccuracy_list.append(vaccuracy)
                        val_writer.add_summary(vsummary, val_iters)
                        self.validate_metrics.append([vloss, vaccuracy])
                        self._print_metrics("Validate", epoch_no + 1, one_val_iters, vloss, vaccuracy)

                    # Compute mean loss and mean accuracy of training steps
                    # in one epoch, and empty lists for next epoch
                    self._print_mean_metrics("Train", epoch_no + 1, tloss_list, taccuracy_list)
                    tloss_list, taccuracy_list = [], []

                    # Compute mean loss and mean accuracy of validating steps in one epoch
                    val_mean_loss = self._print_mean_metrics("Validate", epoch_no + 1, vloss_list, vaccuracy_list)

                    if val_mean_loss < best_val_lmean_oss:
                        best_val_lmean_oss = val_mean_loss
                        # Save model after each epoch
                        self._save_model_per_epoch(sess, saver, epoch_no + 1)

                    print()
                    one_tra_iters = 0
                    one_val_iters = 0
                    epoch_no += 1

                    if epoch_no > self.num_epoches:
                        break

        except tf.errors.OutOfRangeError:
            # Stop training
            print(PCB + "Training has stopped." + PCW)
            # Save metrics into json files
            self._save_metrics("train_metrics.json", self.train_metrics)
            self._save_metrics("validate_metrics.json", self.validate_metrics)
            print((PCB + "Logs have been saved in: {}\n" + PCW).format(self.logs_path))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

        return

    def _save_metrics(self, filename, data):
        '''_SAVE_METRICS

            Save metrics (loss and accuracy) into pickle files.
            Durectiry has been set as self.logs_path.

            Inputs:
            -------
            - filename: string, the name of file, not the full path
            - data: 2D list, including loss and accuracy for either
                    training results or validating results

        '''

        loss = ["{0:.6f}".format(d[0]) for d in data]
        accuracy = ["{0:.6f}".format(d[1]) for d in data]
        metrics = {"loss": loss, "accuracy": accuracy}

        json_path = os.path.join(self.logs_path, filename)
        if os.path.isfile(json_path):
            os.remove(txt_path)

        with open(json_path, "w") as json_file:
            json.dump(metrics, json_file)

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
