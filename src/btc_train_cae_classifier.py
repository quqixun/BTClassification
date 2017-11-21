# Brain Tumor Classification
# Script for Training Classifier
# for Autoencoders
# Author: Qixun Qu
# Create on: 2017/11/11
# Modify on: 2017/11/21

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
import time
import argparse
import numpy as np
import tensorflow as tf
from btc_train import BTCTrain
from btc_cae_parameters import get_parameters


class BTCTrainCAEClassifier(BTCTrain):

    def __init__(self, paras, save_path, logs_path):
        '''__INIT__

            Initialization of class BTCTrain to set parameters
            for constructing, training and validating models.

            Inputs:
            -------
            - paras: dict, parameters for training the model, defined
                     in btc_parameters.py
            - save_path: string, the path of the folder to save models
            - logs_path: string, the path of the folder to save logs

        '''

        super().__init__(paras)

        # The name of autoencoder
        self.net_name = self.set_net_name("cae")
        # The name of classifier
        self.clfier = self.net_name + "_clf"
        self.coder_path = os.path.join(save_path, self.net_name, "model")

        self.model_path = self.set_dir_path(save_path, self.clfier)
        self.logs_path = self.set_dir_path(logs_path, self.clfier)

        self.network = self.models.autoencoder_classier

        return

    def train(self):
        '''TRAIN

            Train and validate the choosen model.

        '''

        # with tf.device("/cpu:0")
        tra_data, tra_labels, val_data, val_labels = self.load_data()
        x, y_input, is_training, learning_rate = self.inputs()

        # with tf.device("/gpu:0")
        # Obtain logits from the model
        y_output = self.network(x, is_training)

        variables = tf.trainable_variables()
        coder_vars = [v for v in variables if "conv" in v.name]
        logit_vars = [v for v in variables if "logits" in v.name]

        # Compute loss and accuracy and merge summary
        # The summary can be displayed by TensorBoard
        loss = self.get_softmax_loss(y_input, y_output, logit_vars)
        accuracy = self.get_accuracy(y_input, y_output)
        merged = tf.summary.merge_all()

        train_op = self.create_optimizer(learning_rate, loss, logit_vars)

        loader = tf.train.Saver(coder_vars)
        saver = tf.train.Saver(logit_vars)

        sess = tf.InteractiveSession()
        sess.run(self.initialize_variables())
        loader.restore(sess, self.coder_path)
        tra_writer, val_writer = self.create_writers(self.logs_path, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        self.blue_print("\nTraining and Validating model: {}\n".format(self.clfier))

        # Initialize counter
        tra_iters, val_iters, epoch_no = 0, 0, 0
        one_tra_iters, one_val_iters = 0, 0
        best_val_lmean_oss = np.inf

        # Lists to save loss and accuracy of each training step
        tloss_list, taccuracy_list = [], []

        # Initialize the timer to count time of one epoch
        epoch_time = time.time()

        try:
            while not coord.should_stop():
                # Initialize the timer to count time of one training step
                tra_step_time = time.time()

                # Training step
                # Feed the graph, run optimizer and get metrics
                tx, ty = sess.run([tra_data, tra_labels])
                tra_fd = {x: tx, y_input: ty, is_training: True,
                          learning_rate: self.learning_rates[epoch_no]}
                tsummary, tloss, taccuracy, _ = sess.run([merged, loss, accuracy, train_op], feed_dict=tra_fd)

                # Get the time of one training step
                tstime = self.get_time(tra_step_time)

                tra_iters += 1
                one_tra_iters += 1

                # Record metrics of training steps
                tloss_list.append(tloss)
                taccuracy_list.append(taccuracy)
                tra_writer.add_summary(tsummary, tra_iters)
                self.train_metrics.append([tloss, taccuracy])
                self.print_metrics("Train", epoch_no + 1, one_tra_iters, tstime, tloss, taccuracy)

                if tra_iters % self.tepoch_iters[epoch_no] == 0:
                    # Validating step
                    # Lists to save loss and accuracy of each validating step
                    vloss_list, vaccuracy_list = [], []
                    while val_iters < self.vepoch_iters[epoch_no]:
                        # Initialize the timer to count time of one validating step
                        val_step_time = time.time()

                        # Feed the graph, get metrics
                        vx, vy = sess.run([val_data, val_labels])
                        val_fd = {x: vx, y_input: vy, is_training: False}
                        vsummary, vloss, vaccuracy = sess.run([merged, loss, accuracy], feed_dict=val_fd)

                        # Get the time of one validating step
                        vstime = self.get_time(val_step_time)

                        val_iters += 1
                        one_val_iters += 1

                        # Record metrics of validating steps
                        vloss_list.append(vloss)
                        vaccuracy_list.append(vaccuracy)
                        val_writer.add_summary(vsummary, val_iters)
                        self.validate_metrics.append([vloss, vaccuracy])
                        self.print_metrics("Validate", epoch_no + 1, one_val_iters, vstime, vloss, vaccuracy)

                    # Get the time of one epoch
                    self.print_time(epoch_no + 1, self.get_time(epoch_time))
                    epoch_time = time.time()

                    # Compute mean loss and mean accuracy of training steps
                    # in one epoch, and empty lists for next epoch
                    self.print_mean_metrics("Train", epoch_no + 1, tloss_list, taccuracy_list)
                    tloss_list, taccuracy_list = [], []

                    # Compute mean loss and mean accuracy of validating steps in one epoch
                    val_mean_loss = self.print_mean_metrics("Validate", epoch_no + 1, vloss_list, vaccuracy_list)

                    if val_mean_loss < best_val_lmean_oss:
                        best_val_lmean_oss = val_mean_loss
                        # Save model after each epoch
                        self.save_model_per_epoch(sess, saver, epoch_no + 1)

                    one_tra_iters = 0
                    one_val_iters = 0
                    epoch_no += 1
                    print()

                    if epoch_no > self.num_epoches:
                        break

        except tf.errors.OutOfRangeError:
            # Stop training
            self.blue_print("Training has stopped.")
            # Save metrics into json files
            self.save_metrics("train_metrics.json", self.train_metrics)
            self.save_metrics("validate_metrics.json", self.validate_metrics)

            self.blue_print("Logs have been saved in: {}\n".format(self.logs_path))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    data_help_str = "Select a data in 'volume' or 'slice'."
    parser.add_argument("--data", action="store", default="volume",
                        dest="data", help=data_help_str)

    sparse_help_str = "Select a sparse constraint in 'kl' and 'wta'."
    parser.add_argument("--sparse", action="store", default="kl",
                        dest="sparse", help=sparse_help_str)

    args = parser.parse_args()

    parent_dir = os.path.dirname(os.getcwd())
    save_path = os.path.join(parent_dir, "models")
    logs_path = os.path.join(parent_dir, "logs")

    parameters = get_parameters("clf", args.data, args.sparse)

    btc = BTCTrainCAEClassifier(parameters, save_path, logs_path)
    btc.train()
