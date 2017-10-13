# Brain Tumor Classification
# Script for Creating Models
# Author: Qixun Qu
# Create on: 2017/10/12
# Modify on: 2017/10/13

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
import numpy as np
import tensorflow as tf


class BTCModels():

    def __init__(self):
        '''__INIT__
        '''
        return

    def _weight_variable(self, shape, name):
        '''_WEIGHT_VARIABLE
        '''

        with tf.variable_scope(name):

            return tf.get_variable(name="W", shape=shape, dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())

    def _bias_variable(self, shape, name):
        '''_BIAS_VARIABLE
        '''

        with tf.variable_scope(name):

            return tf.get_variable(name="b", shape=shape, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1))

    def _conv3d_bn_act(self, x, filter_shape, in_channel, out_channels,
                       padding="SAME", train_bn=True, act=tf.nn.relu, name="cba"):
        '''_CONV3D_BN_ACT
        '''

        def activate(act, x, name):
            return act(features=x, name=name)

        conv_filter_shape = filter_shape + [in_channel, out_channels]

        with tf.name_scope(name):
            W = self._weight_variable(conv_filter_shape, name)
            b = self._bias_variable([out_channels], name)

            cba = tf.nn.conv3d(input=x, filter=W, strides=[1] * 5,
                               padding=padding, name="conv")
            cba = tf.nn.bias_add(cba, b)
            cba = tf.contrib.layers.batch_norm(cba, decay=0.9, is_training=train_bn,
                                               zero_debias_moving_mean=True, scope="bn")
            cba = activate(act=act, inputs=cba, name="act")

            return cba

    def _max_pool(self, x, wsize=2, stride=1, name="max_pool"):
        '''_MAX_POOL
        '''

        return tf.nn.max_pool3d(input=x, ksize=self._get_ksize(wsize),
                                strides=self._get_strides(stride),
                                padding="VALID", name=name)

    def _average_pool(self, x, wsize=2, stride=1, name="avg_pool"):
        '''_AVERAGE_POOL
        '''

        return tf.nn.avg_pool3d(input=x, ksize=self._get_ksize(wsize),
                                strides=self._get_strides(stride),
                                padding="VALID", name=name)

    def _get_ksize(self, wsize):
        '''_GET_KSIZE
        '''

        return [1] + [wsize] * 3 + [1]

    def _get_strides(self, stride):
        '''_GET_STRIDES
        '''

        return [1] + [stride] * 3 + [1]

    def _flatten(self, x, name):
        '''_FLATTEN
        '''

        return tf.reshape(tensor=x, shape=[-1], name=name)

    def _fc_bn_act(self, x, units, train_bn=True, act=tf.nn.relu, name="fba"):
        '''_FULLY_CONNECTED
        '''

        def activate(act, x, name):
            return act(features=x, name=name)

        with tf.name_scope(name):
            fba = tf.contrib.layers.fully_connected(inputs=x, num_outputs=units,
                                                    activation_fn=None, scope="fcn")
            fba = tf.contrib.layers.batch_norm(fba, decay=0.9, is_training=train_bn,
                                               zero_debias_moving_mean=True, scope="bn")
            fba = activate(act=act, inputs=fba, name="act")

            return fba

    def _drop_out(self, x, keep_prob=0.5, name="dropout"):
        '''_DROP_OUT
        '''

        return tf.nn.dropout(x=x, keep_prob=keep_prob, name=name)

    def _output(self, x, classes=3, name="output"):
        '''_OUTPUT
        '''

        return tf.contrib.layers.fully_connected(inputs=x, num_outputs=classes,
                                                 activation_fn=None, scope="out")

    def cnn(self, x):
        '''CNN
        '''

        with tf.name_scope("cnn"):
            cba1 = self._conv3d_bn_act(x, [3, 3, 3], 4, 2, name="layer1")
            max1 = self._max_pool(cba1, name="max_pool1")
            cba2 = self._conv3d_bn_act(max1, [3, 3, 3], 2, 2, name="layer2")
            avg2 = self._avg_pool(cba2, name="avg_pool2")
            cba3 = self._conv3d_bn_act(avg2, [3, 3, 3], 2, 2, name="layer3")
            max3 = self._max_pool(cba3, wsize=3, name="max_pool3")
            flat = self._flatten(max3, name="flatten")
            fcn1 = self._fc_bn_act(flat, units=128, name="fcn1")
            drp1 = self._drop_out(fcn1, keep_prob=0.5, name="drp1")
            fcn2 = self._fc_bn_act(drp1, units=128, name="fcn2")
            drp2 = self._drop_out(fcn2, keep_prob=0.5, name="drp2")
            outp = self._output(drp2, classes=3, name="output")

            return outp


if __name__ == "__main__":

    model = BTCModels()
