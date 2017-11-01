# Brain Tumor Classification
# Script for Training Models
# Author: Qixun Qu
# Create on: 2017/10/14
# Modify on: 2017/10/30

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

-1- Models are defined in class BTCModels.
-2- Hyper-parameters can be set in btc_parameters.py.
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
from btc_parameters import cnn_parameters, cae_parameters


class BTCTrain():

    def __init__(self, net, paras, save_path, logs_path):
        '''__INIT__

            Initialization of class BTCTrain to set parameters
            for constructing, training and validating models.

            Inputs:
            -------
            - net: string, the name of the model applied to train
            - paras: dict, parameters for training the model, defined
                     in btc_parameters.py
            - save_path: string, the path of the folder to save models
            - logs_path: string, the path of the folder to save logs

        '''

        self.net = net
        self.cae = False
        if net == CAE_STRIDE or net == CAE_POOL:
            self.cae = True

        # Initialize BTCTFRecords to load data
        self.tfr = BTCTFRecords()

        # Create folders to keep models
        # if the folder is not exist
        self.model_path = os.path.join(save_path, net)
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        # Create folders to keep models
        # if the folder is not exist
        self.logs_path = os.path.join(logs_path, net)
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
        self.num_epoches = paras["num_epoches"]
        self.learning_rates = self._get_learning_rates(
            paras["learning_rate_first"], paras["learning_rate_last"])
        self.l2_loss_coeff = paras["l2_loss_coeff"]

        if self.cae:
            self.sparse_penalty_coeff = paras["sparse_penalty_coeff"]
            self.p = paras["sparse_level"]

        # For models' structure
        act = paras["activation"]
        alpha = paras["alpha"]
        bn_momentum = paras["bn_momentum"]
        drop_rate = paras["drop_rate"]

        # Initialize BTCModels to set general settings
        self.models = BTCModels(self.net, self.classes_num,
                                act, alpha, bn_momentum, drop_rate)
        self.network = self._get_network()

        # Computer the number of batches in each epoch for
        # both training and validating respectively
        self.tepoch_iters = self._get_epoch_iters(paras["train_num"])
        self.vepoch_iters = self._get_epoch_iters(paras["validate_num"])

        # Create empty lists to save loss and accuracy
        self.train_metrics, self.validate_metrics = [], []

        return

    def _get_network(self):
        '''_GET_NETWORK

            Return network function according to the given net's name.

        '''

        # Set models by given variable
        if self.net == CNN:
            network = self.models.cnn
        elif self.net == FULL_CNN:
            network = self.models.full_cnn
        elif self.net == RES_CNN:
            network = self.models.res_cnn
        elif self.net == DENSE_CNN:
            network = self.models.dense_cnn
        elif self.net == CAE_STRIDE:
            network = self.models.autoencoder_stride
        elif self.net == CAE_POOL:
            network = self.models.autoencoder_pool
        else:  # Raise error if model cannot be found
            raise ValueError("Could not found model.")

        return network

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

    def _get_learning_rates(self, first_rate, last_rate):
        '''_GET_LEARNING_RATES

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

    def _get_loss(self, y_in, y_out, code=None):
        '''_GET_LOSS
        '''

        def softmax_loss(y_in, y_out):
            # Convert labels to onehot array first, such as:
            # [0, 1, 2] ==> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            y_in_onehot = tf.one_hot(indices=y_in, depth=self.classes_num)
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in_onehot,
                                                                          logits=y_out))

        def mean_square_loss(y_in, y_out):
            return tf.div(tf.reduce_mean(tf.square(y_out - y_in)), 2)

        def l2_loss():
            variables = tf.trainable_variables()
            return tf.add_n([tf.nn.l2_loss(v) for v in variables if "kernel" in v.name])

        def sparse_penalty(p, p_hat):
            sp1 = p * (tf.log(p) - tf.log(p_hat))
            sp2 = (1 - p) * (tf.log(1 - p) - tf.log(1 - p_hat))
            return sp1 + sp2

        with tf.name_scope("loss"):
            # Regularization term to reduce overfitting
            loss = l2_loss() * self.l2_loss_coeff

            if not self.cae:
                loss += softmax_loss(y_in, y_out)
            else:
                loss += mean_square_loss(y_in, y_out)
                # loss += tf.reduce_sum(code) * self.sparse_penalty_coeff
                p_hat = tf.reduce_mean(code, axis=[1, 2, 3]) + 1e-8
                loss += tf.reduce_sum(sparse_penalty(self.p, p_hat)) * self.sparse_penalty_coeff

        # Add loss into summary
        tf.summary.scalar("loss", loss)

        return loss

    def _get_accuracy(self, y_in_labels, y_out):
        '''_GET_ACCURACY
        '''

        with tf.name_scope("accuracy"):
            # Obtain the predicted labels for each input example first
            y_out_labels = tf.argmax(input=y_out, axis=1)
            correction_prediction = tf.equal(y_out_labels, y_in_labels)
            accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

        if not self.cae:
            # Add accuracy into summary
            tf.summary.scalar("accuracy", accuracy)

        return accuracy

    def _print_metrics(self, stage, epoch_no, iters, loss, accuracy):
        '''_PRINT_METRICS
        '''

        print((PCG + "[Epoch {}] ").format(epoch_no),
              (stage + " Step {}: ").format(iters),
              "Loss: {0:.10f}, ".format(loss),
              ("Accuracy: {0:.10f}" + PCW).format(accuracy))

        return

    def _print_mean_metrics(self, stage, epoch_no, loss_list, accuracy_list):
        '''_PRINT_MEAN_METRICS
        '''

        loss_mean = np.mean(loss_list)
        accuracy_mean = np.mean(accuracy_list)

        print((PCY + "[Epoch {}] ").format(epoch_no),
              stage + " Stage: ",
              "Mean Loss: {0:.10f}, ".format(loss_mean),
              ("Mean Accuracy: {0:.10f}" + PCW).format(accuracy_mean))

        return

    def _save_model_per_epoch(self, sess, saver, epoch_no):
        '''_SAVE_MODEL_PER_EPOCH
        '''

        ckpt_dir = os.path.join(self.model_path, "epoch-" + str(epoch_no))
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir)

        # Save model's graph and variables of each epoch into folder
        save_path = os.path.join(ckpt_dir, self.net)
        saver.save(sess, save_path, global_step=epoch_no)
        print((PCC + "[Epoch {}] ").format(epoch_no),
              ("Model was saved in: {}\n" + PCW).format(ckpt_dir))

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
        if not self.cae:
            output = self.network(x, is_training)
        else:
            code, output = self.network(x, is_training)

        # Compute loss and accuracy
        if not self.cae:
            loss = self._get_loss(y_input, output)
            accuracy = self._get_accuracy(y_input, output)
        else:  # Autoencoder
            loss = self._get_loss(x, output, code)
            accuracy = tf.cast(0.0, tf.float32)

        # Merge summary
        # The summary can be displayed by TensorBoard
        merged = tf.summary.merge_all()

        # Optimize loss
        with tf.name_scope("train"):
            # Update moving_mean and moving_variance of
            # batch normalization in training process
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # with tf.device("/cpu:0")
        # Load data from tfrecord files
        with tf.name_scope("tfrecords"):
            tra_volumes, tra_labels = self._load_data(self.train_path)
            val_volumes, val_labels = self._load_data(self.validate_path)

        # Create a saver to save model while training
        saver = tf.train.Saver()

        # Define initialization of graph
        with tf.name_scope("init"):
            init = tf.group(tf.local_variables_initializer(),
                            tf.global_variables_initializer())

        sess = tf.InteractiveSession()

        # Create writers to write logs in file
        tra_writer = tf.summary.FileWriter(os.path.join(self.logs_path, "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(self.logs_path, "validate"), sess.graph)

        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print((PCB + "\nTraining and Validating model: {}\n" + PCW).format(self.net))

        # Initialize counter
        tra_iters, one_tra_iters, val_iters, epoch_no = 0, 0, 0, 0

        # Lists to save loss and accuracy of each training step
        tloss_list, taccuracy_list = [], []

        try:
            while not coord.should_stop():
                # Training step
                # Feed the graph, run optimizer and get metrics
                tx, ty = sess.run([tra_volumes, tra_labels])
                tra_fd = {x: tx, y_input: ty, is_training: True, learning_rate: self.learning_rates[epoch_no]}
                # tsummary, tloss, taccuracy, _ = sess.run([merged, loss, accuracy, train_op], feed_dict=tra_fd)

                # --------------------------------------------------
                out, tsummary, tloss, taccuracy, _ = sess.run([output, merged, loss, accuracy, train_op], feed_dict=tra_fd)
                vpath = os.path.join(os.path.join(TEMP_FOLDER, "CAE"), str(ty[0]) + ".npy")
                np.save(vpath, out[0, ...])
                # --------------------------------------------------

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
                        val_iters += 1
                        # Feed the graph, get metrics
                        vx, vy = sess.run([val_volumes, val_labels])
                        val_fd = {x: vx, y_input: vy, is_training: False}
                        vsummary, vloss, vaccuracy = sess.run([merged, loss, accuracy], feed_dict=val_fd)

                        # Record metrics of validating steps
                        vloss_list.append(vloss)
                        vaccuracy_list.append(vaccuracy)
                        val_writer.add_summary(vsummary, val_iters)
                        self.validate_metrics.append([vloss, vaccuracy])
                        self._print_metrics("Validate", epoch_no + 1, val_iters, vloss, vaccuracy)

                    # Compute mean loss and mean accuracy of training steps
                    # in one epoch, and empty lists for next epoch
                    self._print_mean_metrics("Train", epoch_no + 1, tloss_list, taccuracy_list)
                    tloss_list, taccuracy_list = [], []

                    # Compute mean loss and mean accuracy of validating steps in one epoch
                    self._print_mean_metrics("Validate", epoch_no + 1, vloss_list, vaccuracy_list)

                    # Save model after each epoch
                    self._save_model_per_epoch(sess, saver, epoch_no + 1)

                    one_tra_iters = 0
                    epoch_no += 1

                    if epoch_no > self.num_epoches:
                        break

        except tf.errors.OutOfRangeError:
            # Stop training
            print(PCB + "Training has stopped." + PCW)
            # Save metrics into json files
            # self._save_metrics("train_metrics.json", self.train_metrics)
            # self._save_metrics("validate_metrics.json", self.validate_metrics)
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

    '''

        Example of commandline:
        python btc_train.py --model=cnn

    '''

    parser = argparse.ArgumentParser()

    help_str = "Select a model in 'cnn', 'full_cnn', 'res_cnn', 'dense_cnn'" + \
               "'cae_stride' or 'cae_pool'."
    parser.add_argument("--model", action="store", dest="model", help=help_str)
    args = parser.parse_args()

    parent_dir = os.path.dirname(os.getcwd())
    save_path = os.path.join(parent_dir, "models")
    logs_path = os.path.join(parent_dir, "logs")

    if args.model == "cae_stride" or args.model == "cae_pool":
        parameters = cae_parameters
    else:
        parameters = cnn_parameters

    btc = BTCTrain(args.model, parameters, save_path, logs_path)
    btc.train()
