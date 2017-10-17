# Brain Tumor Classification
# Script for Training Models
# Author: Qixun Qu
# Create on: 2017/10/14
# Modify on: 2017/10/17

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
import shutil
import numpy as np
import tensorflow as tf
from btc_settings import *
from btc_models import BTCModels
from btc_parameters import parameters
from btc_tfrecords import BTCTFRecords


class BTCTrain():

    def __init__(self, net, paras, save_path, logs_path):
        '''__INIT__
        '''

        # Basic settings
        self.net = net
        self.models = BTCModels()
        self.tfr = BTCTFRecords()

        self.model_path = os.path.join(save_path, net)
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        self.logs_path = os.path.join(logs_path, net)
        if os.path.isdir(self.logs_path):
            shutil.rmtree(self.logs_path)
        os.makedirs(self.logs_path)

        self.train_path = paras["train_path"]
        self.validate_path = paras["validate_path"]

        self.classes_num = paras["classes_num"]
        self.patch_shape = paras["patch_shape"]
        self.capacity = paras["capacity"]
        self.min_after_dequeue = paras["min_after_dequeue"]

        # Hyper-parameters
        self.batch_size = paras["batch_size"]
        self.num_epoches = paras["num_epoches"]

        # Other settings
        self.tepoch_iters = self._get_epoch_iters(paras["train_num"])
        self.vepoch_iters = self._get_epoch_iters(paras["validate_num"])

        return

    def _get_epoch_iters(self, num):
        '''_GET_EPOCH_ITERS
        '''

        index = np.arange(1, self.num_epoches + 1)
        iters_per_epoch = np.floor(index * (num / self.batch_size))

        return iters_per_epoch.astype(np.int64)

    def _load_data(self, tfrecord_path):
        '''_LOAD_DATA
        '''

        return self.tfr.decode_tfrecord(path=tfrecord_path,
                                        batch_size=self.batch_size,
                                        num_epoches=self.num_epoches,
                                        patch_shape=self.patch_shape,
                                        capacity=self.capacity,
                                        min_after_dequeue=self.min_after_dequeue)

    def train(self):
        '''TRAIN
        '''

        with tf.name_scope("inputs"):
            x = tf.placeholder(tf.float32, [None] + self.patch_shape)
            y_input_classes = tf.placeholder(tf.int64, [None])
            drop_rate = tf.placeholder(tf.float32)

        if self.net == CNN:
            y_output_logits = self.models.cnn(x, self.classes_num, drop_rate)
        elif self.net == FULL_CNN:
            y_output_logits = self.models.full_cnn(x, self.classes_num, drop_rate)
        else:
            raise ValueError("Could not found model.")

        with tf.name_scope("loss"):
            y_input_onehot = tf.one_hot(indices=y_input_classes, depth=self.classes_num)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input_onehot,
                                                                          logits=y_output_logits))
        tf.summary.scalar("loss", loss)

        with tf.name_scope("accuracy"):
            y_output_classes = tf.argmax(input=y_output_logits, axis=1)
            correction_prediction = tf.equal(y_output_classes, y_input_classes)
            accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        merged = tf.summary.merge_all()

        with tf.name_scope("train"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

        with tf.name_scope("tfrecords"):
            tra_volumes, tra_labels = self._load_data(self.train_path)
            val_volumes, val_labels = self._load_data(self.validate_path)

        # Create a saver to save model while training
        saver = tf.train.Saver()

        with tf.name_scope("init"):
            init = tf.group(tf.local_variables_initializer(),
                            tf.global_variables_initializer())

        sess = tf.InteractiveSession()

        tra_writer = tf.summary.FileWriter(os.path.join(self.logs_path, "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(self.logs_path, "validate"), sess.graph)

        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tra_iters, one_tra_iters, val_iters, epoch_no = 0, 0, 0, 0

        print((PCB + "Training and Validating model: {}\n" + PCW).format(self.net))
        tloss_list, taccuracy_list = [], []
        try:
            while not coord.should_stop():
                tx, ty = sess.run([tra_volumes, tra_labels])
                tra_fd = {x: tx, y_input_classes: ty, drop_rate: 0.5}
                tsummary, tloss, taccuracy, _ = sess.run([merged, loss, accuracy, train_op],
                                                         feed_dict=tra_fd)
                tloss_list.append(tloss)
                taccuracy_list.append(taccuracy)
                tra_writer.add_summary(tsummary, tra_iters)

                tra_iters += 1
                one_tra_iters += 1
                print((PCG + "[Epoch {}] ").format(epoch_no + 1),
                      "Train Step {}: ".format(one_tra_iters),
                      "Loss: {0:.10f}, ".format(tloss),
                      ("Accuracy: {0:.10f}" + PCW).format(taccuracy))

                if tra_iters % self.tepoch_iters[epoch_no] == 0:
                    vloss_list, vaccuracy_list = [], []
                    while val_iters < self.vepoch_iters[epoch_no]:
                        val_iters += 1
                        vx, vy = sess.run([val_volumes, val_labels])
                        val_fd = {x: vx, y_input_classes: vy, drop_rate: 0.0}
                        vsummary, vloss, vaccuracy = sess.run([merged, loss, accuracy],
                                                              feed_dict=val_fd)
                        vloss_list.append(vloss)
                        vaccuracy_list.append(vaccuracy)
                        val_writer.add_summary(vsummary, tra_iters)

                    tloss_mean = np.mean(tloss_list)
                    taccuracy_mean = np.mean(taccuracy_list)
                    tloss_list, taccuracy_list = [], []

                    print((PCY + "[Epoch {}] ").format(epoch_no + 1),
                          "Train Stage: ",
                          "Mean Loss: {0:.10f}, ".format(tloss_mean),
                          ("Mean Accuracy: {0:.10f}" + PCW).format(taccuracy_mean))

                    vloss_mean = np.mean(vloss_list)
                    vaccuracy_mean = np.mean(vaccuracy_list)

                    print((PCY + "[Epoch {}] ").format(epoch_no + 1),
                          "Validate Stage: ",
                          "Mean Loss: {0:.10f}, ".format(vloss_mean),
                          ("Mean Accuracy: {0:.10f}" + PCW).format(vaccuracy_mean))

                    ckpt_dir = os.path.join(self.model_path, "epoch-" + str(epoch_no + 1))
                    if os.path.isdir(ckpt_dir):
                        shutil.rmtree(ckpt_dir)
                    os.makedirs(ckpt_dir)

                    save_path = os.path.join(ckpt_dir, self.net)
                    saver.save(sess, save_path, global_step=epoch_no + 1)
                    print((PCC + "[Epoch {}] ").format(epoch_no + 1),
                          ("Model was saved in: {}\n" + PCW).format(ckpt_dir))

                    one_tra_iters = 0
                    epoch_no += 1

        except tf.errors.OutOfRangeError:
            print(PCB + "Training has stopped." + PCW)
            print((PCB + "Logs have been saved in: {}" + PCW).format(self.logs_path))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    save_path = os.path.join(parent_dir, "models")
    logs_path = os.path.join(parent_dir, "logs")

    # model = "cnn"
    model = "full_cnn"
    # model = "res_cnn"
    # model = "dense_cnn"

    btc = BTCTrain(model, parameters, save_path, logs_path)
    btc.train()
