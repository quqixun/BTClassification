# Brain Tumor Classification
# Script for Creating Models
# Author: Qixun Qu
# Create on: 2017/10/12
# Modify on: 2017/10/20

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
from btc_settings import *
from functools import reduce
from tensorflow.contrib.layers import xavier_initializer


class BTCModels():

    def __init__(self, net, classes, act="relu", alpha=None,
                 momentum=0.99, train_bn=True, bc=True):
        '''__INIT__
        '''

        self.act = act
        self.alpha = alpha
        self.momentum = momentum
        self.train_bn = train_bn
        self.classes = classes

        if net == DENSE_CNN:
            self.bc = bc

        return

    #
    # Helper Functions
    #

    def _conv3d(self, x, filters, kernel_size, strides=1,
                padding="same", name="conv_var"):
        '''_CONV3D

            Full:  self._conv3d(x, 32, 3, 1, "same", "conv")
            Short: self._conv3d(x, 32, 3)

        '''

        return tf.layers.conv3d(inputs=x,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                kernel_initializer=xavier_initializer(),
                                name=name)

    def _fully_connected(self, x, units, name="fc_var"):
        '''_FULLy_CONNECTED

            Full:  self._fully_connected(x, 128, "fc")
            short: self._fully_connected(x, 128)

        '''

        return tf.layers.dense(inputs=x,
                               units=units,
                               kernel_initializer=xavier_initializer(),
                               name=name)

    def _batch_norm(self, x, name="bn_var"):
        '''_BATCH_NORM

            Full:  self._batch_norm(x, "bn")
            Short: self._batch_norm(x)

        '''

        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=self.momentum,
                                            is_training=self.train_bn,
                                            scope=name)

    def _activate(self, x, name="act"):
        '''_ACTIVATE

           Full:  self._activate(x, "act")
           Short: self._activate(x)

        '''

        if self.act == "relu":
            return tf.nn.relu(x, name)
        elif self.act == "lrelu":
            alpha = 0.2 if self.alpha is None else self.alpha
            return tf.nn.leaky_relu(x, alpha, name)
        else:
            raise ValueError("Could not find act in ['relu', 'lrelu']")

        return

    def _conv3d_bn_act(self, x, filters, kernel_size, strides=1,
                       name="cba", padding="same", act=True):
        '''_CONV3D_BN_ACT

            Full:  self._conv3d_bn_act(x, 32, 3, 1, "cba", same", True)
            Short: self._conv3d_bn_act(x, 32, 3, 1, "cba")

        '''

        with tf.variable_scope(name):
            with tf.name_scope("conv3d"):
                cba = self._conv3d(x, filters, kernel_size, strides, padding)
            with tf.name_scope("batch_norm"):
                cba = self._batch_norm(cba)
            if act:  # check for residual block
                with tf.name_scope("activate"):
                    cba = self._activate(cba)

            return cba

    def _fc_bn_act(self, x, units, name="fba"):
        '''_FULLY_CONNECTED

            Full:  self._fc_bn_act(x, 128, "fba")
            Short: self._fc_bn_act(x, 128)

        '''

        with tf.variable_scope(name):
            with tf.name_scope("full_connection"):
                fba = self._fully_connected(x, units)
            with tf.name_scope("batch_norm"):
                fba = self._batch_norm(fba)
            with tf.name_scope("activate"):
                fba = self._activate(fba)

            return fba

    def _max_pool(self, x, psize=2, stride=2, name="max_pool"):
        '''_MAX_POOL

            Full:  self._max_pool(x, 2, 2, "max_pool")
            Short: self._max_pool(x)

        '''

        return tf.layers.max_pooling3d(inputs=x,
                                       pool_size=psize,
                                       strides=stride,
                                       name=name)

    def _average_pool(self, x, psize=2, stride=2, name="avg_pool"):
        '''_AVERAGE_POOL

            Full:  self._average_pool(x, 2, 2, "avg_pool")
            Short: self._average_pool(x)

        '''

        return tf.layers.average_pooling3d(inputs=x,
                                           pool_size=psize,
                                           strides=stride,
                                           name=name)

    def _flatten(self, x, name="flatten"):
        '''_FLATTEN

            Full:  self._flatten(x, "flatten")
            Short: self._flatten(x)

        '''

        x_shape = x.get_shape().as_list()
        f_shape = reduce(mul, x_shape[1:], 1)

        return tf.reshape(tensor=x, shape=[-1, f_shape], name=name)

    def _dropout(self, x, name="dropout"):
        '''_DROP_OUT

            Full:  self._dropout(x, "dropout")

        '''

        return tf.layers.dropout(inputs=x, rate=self.drop_rate, name=name)

    #
    # Helper function for cnn
    #

    def _logits_fc(self, x, name="logits"):
        '''_LOGITS_FC

            Full:  self._logits_fc(x, "logits")
            Short: self._logits_fc(x)

        '''

        return self._fully_connected(x, self.classes, name)

    #
    # Helper function for full cnn
    #

    def _logits_conv(self, x, name="logits"):
        '''_LOGITS_CONV

            Full:  self._logits_conv(x, "logits")
            Short: self._logits_conv(x)

        '''

        x_shape = x.get_shape().as_list()

        with tf.variable_scope(name):
            with tf.name_scope("conv3d"):
                return self._conv3d(x, self.classes, x_shape[1:-1], 1, "valid")

    #
    # Helper function for residual cnn
    #

    def _res_block(self, x, filters, strides=1, name="res"):
        '''_RES_BLOCK

            Full:  self._res_block(x, [8, 16, 32], 1, "res")
                   self._res_block(x, [8, 16, 32])

        '''

        shortcut = False
        if (x.get_shape().as_list()[-1] != filters[2]) or strides != 1:
            shortcut = True

        res = self._conv3d_bn_act(x, filters[0], 1, strides, name + "_conv1", "valid")
        res = self._conv3d_bn_act(res, filters[1], 3, 1, name + "_conv2", "same")
        res = self._conv3d_bn_act(res, filters[2], 1, 1, name + "_conv3", "valid", False)

        if shortcut:
            x = self._conv3d_bn_act(x, filters[2], 1, strides, name + "_shortcut", "valid", False)

        with tf.name_scope(name + "_add"):
            res = tf.add(res, x)
        with tf.name_scope(name + "_activate"):
            return self._activate(res)

    #
    # Helpher functions for dense cnn
    #

    def _composite(self, x, filters, kernel_size=3, name="composite"):
        '''_COPOSITE
        '''

        with tf.name_scope("batch_norm"):
            comp = self._batch_norm(x)
        with tf.name_scope("activate"):
            comp = self._activate(comp)
        with tf.name_scope("conv3d"):
            comp = self._conv3d(comp, filters, kernel_size)
        with tf.name_scope("dropout"):
            comp = self._dropout(comp)

        return comp

    def _bottleneck(self, x, filters, name="bottleneck"):
        '''_BOTTLENECK
        '''

        with tf.name_scope("batch_norm"):
            bott = self._batch_norm(x)
        with tf.name_scope("activate"):
            bott = self._activate(bott)
        with tf.name_scope("conv3d"):
            bott = self._conv3d(bott, filters * 4, 1, padding="valid")
        with tf.name_scope("dropout"):
            bott = self._dropout(bott)

        return bott

    def _dense_internal(self, x, growth_rate, no, name):
        '''_DENSE_INTERNAL
        '''

        dint = x
        if self.bc:
            with tf.variable_scope(name + "_bott" + no):
                dint = self._bottleneck(x, growth_rate)
        with tf.variable_scope(name + "_comp" + no):
            dint = self._composite(dint, growth_rate, 3)
        with tf.name_scope(name + "_concat" + no):
            dint = tf.concat((x, dint), 4)

        return dint

    def _dense_block(self, x, growth_rate, internals, name="dense_block"):
        '''_DENSE_BLOCK
        '''

        dense = x
        for internal in range(internals):
            dense = self._dense_internal(dense, growth_rate, str(internal + 1), name)

        return dense

    def _transition(self, x, name="transition"):
        '''_TRANSITION
        '''

        out_channels = x.get_shape().as_list()[-1]
        with tf.variable_scope(name + "_comp"):
            tran = self._composite(x, out_channels, 1)
        with tf.name_scope(name + "_avgpool"):
            tran = self._average_pool(tran)

        return tran

    def _last_transition(self, x):
        '''_LOGITS_DENSE
        '''

        with tf.variable_scope("last_trans"):
            with tf.name_scope("batch_norm"):
                last_tran = self._batch_norm(x)
            with tf.name_scope("activate"):
                last_tran = self._activate(last_tran)

        out_channels = last_tran.get_shape().as_list()[-1]
        with tf.name_scope("global_avgpool"):
            last_tran = self._average_pool(last_tran, out_channels, out_channels)

        return last_tran

    #
    # A Simple Test Case
    #

    def _test(self):
        '''_TEST
        '''

        self.drop_rate = 0.5

        x = tf.placeholder(tf.float32, [5, 36, 36, 36, 4], "input")
        cba1 = self._conv3d_bn_act(x, 2, 3, 1, "layer1")
        max1 = self._max_pool(cba1, 2, 2, "max_pool1")
        cba2 = self._conv3d_bn_act(max1, 2, 3, 1, "layer2")
        avg2 = self._average_pool(cba2, 2, 2, "avg_pool2")
        cba3 = self._conv3d_bn_act(avg2, 2, 3, 1, "layer3")
        max3 = self._max_pool(cba3, 2, 2, "max_pool3")
        flat = self._flatten(max3, "flatten")
        fcn1 = self._fc_bn_act(flat, 64, "fcn1")
        drp1 = self._dropout(fcn1, "drop1")
        fcn2 = self._fc_bn_act(drp1, 64, "fcn2")
        drp2 = self._dropout(fcn2, "drop2")
        outp = self._logits_fc(drp2, 3, "logits")
        prob = tf.nn.softmax(logits=outp, name="softmax")

        print("Simple test of Class BTCModels")
        print("Input 5 volumes in 3 classes")
        print("Output probabilities' shape: ", prob.shape)

        return

    #
    # Contruct Models
    #

    def cnn(self, x, drop_rate):
        '''CNN

        '''

        self.drop_rate = drop_rate

        # Here is a very simple case to test btc_train first
        net = self._conv3d_bn_act(x, 1, 3, 1, "layer1")
        net = self._max_pool(net, 2, 2, "max_pool1")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer2")
        net = self._max_pool(net, 2, 2, "max_pool2")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer3")
        net = self._max_pool(net, 2, 2, "max_pool3")
        net = self._flatten(net, "flatten")
        net = self._fc_bn_act(net, 3, "fc1")
        net = self._dropout(net, "drop1")
        net = self._fc_bn_act(net, 3, "fc2")
        net = self._dropout(net, "drop2")
        net = self._logits_fc(net, "logits")

        return net

    def full_cnn(self, x, drop_rate):
        '''FULL_CNN
        '''

        self.drop_rate = drop_rate

        # Here is a very simple case to test btc_train first
        net = self._conv3d_bn_act(x, 1, 3, 1, "layer1")
        net = self._max_pool(net, 2, 2, "max_pool1")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer2")
        net = self._max_pool(net, 2, 2, "max_pool2")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer3")
        net = self._max_pool(net, 2, 2, "max_pool3")
        net = self._logits_conv(net, "logits_conv")
        net = self._flatten(net, "logits_flatten")

        return net

    def res_cnn(self, x, drop_rate):
        '''RES_CNN
        '''

        self.drop_rate = drop_rate

        # Here is a very simple case to test btc_train first
        net = self._conv3d_bn_act(x, 1, 5, 2, "preconv")
        net = self._res_block(net, [1, 1, 1], 2, "res1")
        net = self._res_block(net, [1, 1, 2], 2, "res2")
        net = self._max_pool(net, 7, 7, "max_pool")
        net = self._flatten(net, "flatten")
        net = self._logits_fc(net, "logits")

        return net

    def dense_cnn(self, x, drop_rate):
        '''DENSE_NET
        '''

        self.drop_rate = drop_rate

        # Here is a very simple case to test btc_train first
        net = self._conv3d(x, 1, 5, 2, "same", "preconv")
        net = self._dense_block(net, 1, 2, "dense1")
        net = self._transition(net, "trans1")
        net = self._dense_block(net, 1, 2, "dense2")
        net = self._last_transition(net)
        net = self._flatten(net, "flatten")
        net = self._logits_fc(net, "logits")

        return net


if __name__ == "__main__":

    models = BTCModels("dense_cnn", classes=3, act="relu", alpha=None,
                       momentum=0.99, train_bn=True)
    models._test()
