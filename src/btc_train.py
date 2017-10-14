# Brain Tumor Classification
# Script for Training Models
# Author: Qixun Qu
# Create on: 2017/10/14
# Modify on: 2017/10/14

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


import numpy as np
import tensorflow as tf
from btc_models import BTCModels
from btc_parameters import parameters
from btc_tfrecords import BTCTFRecords


class BTCTrain():

    def __init__(self, net, paras):
        '''__INIT__
        '''

        # Basic settings
        self.net = net
        self.mode = "train"
        self.models = BTCModels()
        self.tfr = BTCTFRecords()

        self.train_path = paras["train_path"]
        self.validate_path = paras["validate_path"]

        self.patch_shape = paras["patch_shape"]
        self.capacity = paras["capacity"]
        self.min_after_dequeue = paras["min_after_dequeue"]

        # Hyper-parameters
        self.batch_size = paras["batch_size"]
        self.num_epoches = paras["num_epoches"]

        # Other settings
        self.tepoch_iters = self._get_epoch_iters(paras["train_num"])
        self.vepoch_iters = self._get_epoch_iters(paras["validate_num"])

        print("Train iters: ", self.tepoch_iters)
        print("Validate iters: ", self.vepoch_iters)

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

        tra_volumes, tra_labels = self._load_data(self.train_path)
        val_volumes, val_labels = self._load_data(self.validate_path)

        init = tf.group(tf.local_variables_initializer(),
                        tf.global_variables_initializer())

        sess = tf.Session()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tra_iters, val_iters, epoch_no = 0, 0, 0

        try:
            while not coord.should_stop():
                tra_iters += 1
                tx, ty = sess.run([tra_volumes, tra_labels])
                print("Train: ", tra_iters, ty)

                if tra_iters % self.tepoch_iters[epoch_no] == 0:
                    while val_iters < self.vepoch_iters[epoch_no]:
                        val_iters += 1
                        vx, vy = sess.run([val_volumes, val_labels])
                        print("Validate: ", val_iters, vy)

                    epoch_no += 1
        except tf.errors.OutOfRangeError:
            print("Training has stopped.")
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

        return


if __name__ == "__main__":
    btc = BTCTrain("cnn", parameters)
    btc.train()
