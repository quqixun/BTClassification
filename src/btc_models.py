# Brain Tumor Classification
# Script for Creating Models
# Author: Qixun Qu
# Create on: 2017/10/12
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


import tensorflow as tf
from operator import mul
from functools import reduce
from tensorflow.contrib.layers import xavier_initializer


class BTCModels():

    def __init__(self):
        '''__INIT__
        '''
        return

    #
    # Helper Functions
    #

    def _conv3d(self, x, filters, kernel_size,
                padding="same", name="conv"):
        '''_CONV3D

            Full:  self._conv3d(x, 32, 3, "same", "conv")
            Short: self._conv3d(x, 32, 3)

        '''

        return tf.layers.conv3d(inputs=x,
                                filters=filters,
                                kernel_size=kernel_size,
                                padding=padding,
                                kernel_initializer=xavier_initializer(),
                                name=name)

    def _fully_connected(self, x, units, name="fcn"):
        '''_FULLy_CONNECTED

            Full:  self._fully_connected(x, 128, "fc")
            short: self._fully_connected(x, 128)

        '''

        return tf.layers.dense(inputs=x,
                               units=units,
                               kernel_initializer=xavier_initializer(),
                               name=name)

    def _batch_norm(self, x, momentum=0.9, training=True, name="bn"):
        '''_BATCH_NORM

            Full:  self._batch_norm(x, 0.9, True, "bn")
            Short: self._batch_norm(x)

        '''

        return tf.layers.batch_normalization(inputs=x,
                                             momentum=momentum,
                                             training=training,
                                             name=name)

    def _activate(self, x, act="relu", alpha=None, name="act"):
        '''_ACTIVATE

           Full:  self._activate(x, "relu", None, "act")
                  self._activate(x, "lrelu", 0.2, "act")
           Short: self._activate(x) # for relu
                  self._activate(x, "lrelu", 0.2)

        '''

        if act == "relu":
            return tf.nn.relu(x, "act")
        elif act == "lrelu":
            alpha = 0.2 if alpha is None else alpha
            return tf.nn.leaky_relu(x, alpha, "act")
        elif act is None:
            return x
        else:
            raise ValueError("Could not find act in ['relu', 'lrelu', None]")

        return

    def _conv3d_bn_act(self, x, filters, kernel_size,
                       name="cba", act="relu", alpha=None,
                       padding="same", momentum=0.9, train_bn=True):
        '''_CONV3D_BN_ACT

            Full:  self._conv3d_bn_act(x, 32, 3, "cba", "relu", None, "same", 0.9, True)
                   self._conv3d_bn_act(x, 32, 3, "cba", "lrelu", 0.2, "same", 0.9, True)
            Short: self._conv3d_bn_act(x, 32, 3, "cba") # for relu
                   self._conv3d_bn_act(x, 32, 3, "cba", "lrelu", 0.2)

        '''

        with tf.variable_scope(name):
            cba = self._conv3d(x, filters, kernel_size, padding)
            cba = self._batch_norm(cba, momentum, train_bn)
            cba = self._activate(cba, act, alpha)

            return cba

    def _fc_bn_act(self, x, units, name="fba",
                   act="relu", alpha=None,
                   momentum=0.9, train_bn=True):
        '''_FULLY_CONNECTED

            Full:  self._fc_bn_act(x, 128, "fba", "relu", None, 0.9, True)
                   self._fc_bn_act(x, 128, "fba", "lrelu", 0.2, 0.9, True)
            Short: self._fc_bn_act(x, 128, "fba") # for relu
                   self._fc_bn_act(x, 128, "fba", "lrelu", 0.2)

        '''

        with tf.variable_scope(name):
            fba = self._fully_connected(x, units)
            fba = self._batch_norm(fba, momentum, train_bn)
            fba = self._activate(fba, act, alpha)

            return fba

    def _max_pool(self, x, psize=2, stride=2, name="max_pool"):
        '''_MAX_POOL

            Full:  self._max_pool(x, 2, 1, "max_pool")
            Short: self._max_pool(x)

        '''

        return tf.layers.max_pooling3d(inputs=x,
                                       pool_size=psize,
                                       strides=stride,
                                       padding="valid",
                                       name=name)

    def _average_pool(self, x, psize=2, stride=2, name="avg_pool"):
        '''_AVERAGE_POOL

            Full:  self._average_pool(x, 2, 1, "avg_pool")
            Short: self._average_pool(x)

        '''

        return tf.layers.average_pooling3d(inputs=x,
                                           pool_size=psize,
                                           strides=stride,
                                           padding="valid",
                                           name=name)

    def _flatten(self, x, name="flatten"):
        '''_FLATTEN

            Full:  self._flatten(x, "flatten")
            Short: self._flatten(x)

        '''

        x_shape = x.get_shape().as_list()
        f_shape = reduce(mul, x_shape[1:], 1)

        return tf.reshape(tensor=x, shape=[-1, f_shape], name=name)

    def _dropout(self, x, name="dropout", drop_rate=0.5):
        '''_DROP_OUT

            Full:  self._dropout(x, "dropout", 0.5)
                   self._dropout(x, "dropout", 0)
                   self._dropout(x, "dropout", 0)
            Short: self._dropout(x, "dropout") # for training

        '''

        return tf.layers.dropout(inputs=x,
                                 rate=drop_rate,
                                 name=name)

    def _logits(self, x, classes=3, name="logits"):
        '''_OUTPUT

            Full:  self._logits(x, 3, "logits")
            Short: self._logits(x)

        '''

        return self._fully_connected(x, classes, name)

    #
    # A Simple Test Case
    #

    def _test(self):
        '''_TEST
        '''

        with tf.variable_scope("test"):
            x = tf.placeholder(tf.float32, [5, 36, 36, 36, 4], "input")
            cba1 = self._conv3d_bn_act(x, 2, 3, "layer1")
            max1 = self._max_pool(cba1, 2, 2, "max_pool1")
            cba2 = self._conv3d_bn_act(max1, 2, 3, "layer2", "lrelu", 0.2)
            avg2 = self._average_pool(cba2, 2, 2, "avg_pool2")
            cba3 = self._conv3d_bn_act(avg2, 2, 3, "layer3", "lrelu", 0.3)
            max3 = self._max_pool(cba3, 2, 2, "max_pool3")
            flat = self._flatten(max3, "flatten")
            fcn1 = self._fc_bn_act(flat, 64, "fcn1")
            drp1 = self._dropout(fcn1, "drp1", 0.5)
            fcn2 = self._fc_bn_act(drp1, 64, "fcn2", "lrelu", 0.2)
            drp2 = self._dropout(fcn2, "drp2", 0)
            outp = self._logits(drp2, 3, "logits")
            probs = tf.nn.softmax(logits=outp, name="softmax")

        print("Simple test of Class BTCModels")
        print("Input 5 volumes in 3 classes")
        print("Output probabilities' shape: ", probs.shape)

        return

    #
    # Contruct Models
    #

    def cnn(self, x, drop_rate=0.5):
        '''CNN
        '''

        # Here is a very simple case to test first
        with tf.variable_scope("cnn"):
            cba1 = self._conv3d_bn_act(x, 2, 3, "layer1")
            max1 = self._max_pool(cba1, 2, 2, "max_pool1")
            cba2 = self._conv3d_bn_act(max1, 2, 3, "layer2")
            max2 = self._max_pool(cba2, 2, 2, "max_pool2")
            cba3 = self._conv3d_bn_act(max2, 2, 3, "layer3")
            max3 = self._max_pool(cba3, 2, 2, "max_pool3")
            flat = self._flatten(max3, "flatten")
            fcn1 = self._fc_bn_act(flat, 3, "fcn1")
            drp1 = self._dropout(fcn1, "drp1", drop_rate)
            fcn2 = self._fc_bn_act(drp1, 3, "fcn2")
            drp2 = self._dropout(fcn2, "drp2", drop_rate)
            outp = self._logits(drp2, 3, "logits")

            return outp

    def full_cnn(self, x, mode="train"):
        '''FULL_CNN
        '''

        # To be finished

        return

    def res_cnn(self, x, mode="train"):
        '''RES_CNN
        '''

        # To be finished

        return

    def dense_cnn(self, x, mode="train"):
        '''DENSE_NET
        '''

        # To be finished

        return


if __name__ == "__main__":

    models = BTCModels()
    models._test()
