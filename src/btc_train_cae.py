# Brain Tumor Classification
# Script for Training Autoencoders
# Author: Qixun Qu
# Create on: 2017/11/06
# Modify on: 2017/11/14

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

Class BTCTrainCAE

-1- Models for autoencoders are defined in class BTCModels.
-2- Hyper-parameters for autoencoders can be set in btc_cae_parameters.py.
-3- Loading tfrecords for training and validating by
    functions in class BTCTFRecords.

'''


from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf
from btc_train import BTCTrain
from btc_cae_parameters import get_parameters


class BTCTrainCAE(BTCTrain):

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

        self.net_name = self._set_net_name("cae_" + self.cae_pool)
        self.model_path = self._set_dir_path(save_path, self.net_name)
        self.logs_path = self._set_dir_path(logs_path, self.net_name)

        self.network = self.models.autoencoder

        return

    def train(self):
        '''TRAIN

            Train and validate the choosen model.

        '''

        # with tf.device("/cpu:0")
        tra_data, tra_labels, val_data, val_labels = self._load_data()
        x, y_input, is_training, learning_rate = self._inputs()

        # with tf.device("/gpu:0")
        # Obtain logits from the model
        code, y_output = self.network(x, is_training)

        # Compute loss and merge summary
        # The summary can be displayed by TensorBoard
        loss = self._get_sparsity_loss(x, y_output, code)
        merged = tf.summary.merge_all()

        train_op = self._create_optimizer(learning_rate, loss)

        # Create a saver to save model while training
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(self._initialize_variables())
        tra_writer, val_writer = self._create_writers(self.logs_path, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        self._blue_print("\nTraining and Validating model: {}\n".format(self.net_name))

        # Initialize counter
        tra_iters, val_iters, epoch_no = 0, 0, 0
        one_tra_iters, one_val_iters = 0, 0
        best_val_mean_loss = np.inf

        # Lists to save loss and accuracy of each training step
        tloss_list = []

        try:
            while not coord.should_stop():
                # Training step
                # Feed the graph, run optimizer and get metrics
                tx, ty = sess.run([tra_data, tra_labels])
                tra_fd = {x: tx, y_input: ty, is_training: True, learning_rate: self.learning_rates[epoch_no]}
                tsummary, tloss, _ = sess.run([merged, loss, train_op], feed_dict=tra_fd)

                tra_iters += 1
                one_tra_iters += 1

                # Record metrics of training steps
                tloss_list.append(tloss)
                tra_writer.add_summary(tsummary, tra_iters)
                self.train_metrics.append(tloss)
                self._print_metrics("Train", epoch_no + 1, one_tra_iters, tloss)

                if tra_iters % self.tepoch_iters[epoch_no] == 0:
                    # Validating step
                    # Lists to save loss and accuracy of each validating step
                    vloss_list = []
                    while val_iters < self.vepoch_iters[epoch_no]:
                        # Feed the graph, get metrics
                        vx, vy = sess.run([val_data, val_labels])
                        val_fd = {x: vx, y_input: vy, is_training: False}
                        vsummary, vloss = sess.run([merged, loss], feed_dict=val_fd)

                        one_val_iters += 1
                        val_iters += 1

                        # Record metrics of validating steps
                        vloss_list.append(vloss)
                        val_writer.add_summary(vsummary, val_iters)
                        self.validate_metrics.append(vloss)
                        self._print_metrics("Validate", epoch_no + 1, one_val_iters, vloss)

                    # Compute mean loss and mean accuracy of training steps
                    # in one epoch, and empty lists for next epoch
                    self._print_mean_metrics("Train", epoch_no + 1, tloss_list)
                    tloss_list = []

                    # Compute mean loss and mean accuracy of validating steps in one epoch
                    val_mean_loss = self._print_mean_metrics("Validate", epoch_no + 1, vloss_list)

                    if val_mean_loss < best_val_mean_loss:
                        best_val_mean_loss = val_mean_loss
                        # Save model after each epoch
                        self._save_model_per_epoch(sess, saver, epoch_no + 1)

                    one_tra_iters = 0
                    one_val_iters = 0
                    epoch_no += 1
                    print()

                    if epoch_no > self.num_epoches:
                        break

        except tf.errors.OutOfRangeError:
            # Stop training
            self._blue_print("Training has stopped.")
            # Save metrics into json files
            self._save_metrics("train_metrics.json", self.train_metrics)
            self._save_metrics("validate_metrics.json", self.validate_metrics)
            self._blue_print("Logs have been saved in: {}\n".format(self.logs_path))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    help_str = "Select a data in 'volume' or 'slice'."
    parser.add_argument("--data", action="store", dest="data", help=help_str)
    args = parser.parse_args()

    parent_dir = os.path.dirname(os.getcwd())
    save_path = os.path.join(parent_dir, "models")
    logs_path = os.path.join(parent_dir, "logs")

    parameters = get_parameters(args.data, "cae")

    btc = BTCTrainCAE(parameters, save_path, logs_path)
    btc.train()
