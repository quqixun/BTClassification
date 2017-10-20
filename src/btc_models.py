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

'''

Class BTCModels

-1- Define several basic helper functions, which are the
    basic modules to build simple CNN models.
-2- Define several specific helper functions to construct
    CNN models with more complicate structures.
-3- Build models: CNN, Full-CNN, Res-CNN and Dense-CNN.

'''


import tensorflow as tf
from operator import mul
from functools import reduce
from tensorflow.contrib.layers import xavier_initializer


class BTCModels():

    def __init__(self, net, classes, act="relu", alpha=None,
                 momentum=0.99, drop_rate=0.5):
        '''__INIT__

            Initialization of BTCModels. In this functions,
            commen parameters of all models are set first.

            Inputs:
            -------
            - net: string, the name of the model applied to train
            - classes: int, the number of grading groups
            - act: string, indicate the activation method by either
                   "relu" or "lrelu" (leaky relu)
            - alpha: float, slope of the leaky relu at x < 0
            - momentum: float, momentum for removing average in
                        batch normalization, typically values are
                        0.999, 0.99, 0.9, etc
            - drop_rate: float, rate of dropout of input units,
                         which is between 0 and 1

        '''

        self.classes = classes

        self.act = act
        self.alpha = alpha
        self.momentum = momentum
        self.drop_rate = drop_rate

        # A symbol to indicate whether the model is used to train
        # The symbol will be assigned as a placeholder while
        # feeding the model in training and validating steps
        self.training = None

        return

    #
    # Basic Helper Functions
    #

    def _conv3d(self, x, filters, kernel_size, strides=1,
                padding="same", name="conv_var"):
        '''_CONV3D

            Return 3D convolution layer with variables
            initialized by xavier method.

            Usages:
            -------
            - full:  self._conv3d(x, 32, 3, 1, "same", "conv")
            - short: self._conv3d(x, 32, 3)

            Inputs:
            -------
            - x: tensor, input layer
            - filters: int, the number of kernels
            - kernel_size: int, the size of cube kernel
            - strides: int, strides along three dimentions
            - padding: string, "same" or "valid"
            - name: string, layer's name

            Output:
            -------
            - a 3D convolution layer

        '''

        with tf.name_scope("conv3d"):
            return tf.layers.conv3d(inputs=x,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    kernel_initializer=xavier_initializer(),
                                    name=name)

    def _fully_connected(self, x, units, name="fc_var"):
        '''_FULLY_CONNECTED

            Return fully connected layer with variables
            initialized by xavier method.

            Usages:
            -------
            - full:  self._fully_connected(x, 128, "fc")
            - short: self._fully_connected(x, 128)

            Inputs:
            -------
            - x: tensor, input layer
            - units: int, the number of neurons
            - name: string, layer's name

            Outputs:
            --------
            - a fully connected layer

        '''

        with tf.name_scope("full_connection"):
            return tf.layers.dense(inputs=x,
                                   units=units,
                                   kernel_initializer=xavier_initializer(),
                                   name=name)

    def _batch_norm(self, x, name="bn_var"):
        '''_BATCH_NORM

            Normalize the input layer.
            Momentum and symbol of training have been
            assigned while the class is initialized.

            Usages:
            -------
            - full:  self._batch_norm(x, "bn")
            - short: self._batch_norm(x)

            Inputs:
            -------
            - x: tensor, input layer
            - name: string, layer's name

            Output:
            -------
            - normalized layer

        '''

        with tf.name_scope("batch_norm"):
            return tf.contrib.layers.batch_norm(inputs=x,
                                                decay=self.momentum,
                                                is_training=self.training,
                                                scope=name)

    def _activate(self, x, name="act"):
        '''_ACTIVATE

           Activate input layer. Two approaches can be available,
           which are ReLU or leaky ReLU.
           Activation method and setting has been set while the
           class is initialized.

           Usages:
           -------
           - full:  self._activate(x, "act")
           - short: self._activate(x)

           Inputs:
           -------
           - x: tensor, input layer
           - name: string, layer's name

           Output:
           -------
           - an activated layer

        '''

        with tf.name_scope("activate"):
            if self.act == "relu":
                return tf.nn.relu(x, name)
            elif self.act == "lrelu":
                # Set slope of leaky ReLU
                alpha = 0.2 if self.alpha is None else self.alpha
                return tf.nn.leaky_relu(x, alpha, name)
            else:  # Raise error if activation method cannot be found
                raise ValueError("Could not find act in ['relu', 'lrelu']")

        return

    def _conv3d_bn_act(self, x, filters, kernel_size, strides=1,
                       name="cba", padding="same", act=True):
        '''_CONV3D_BN_ACT

            A convolution block, including three sections:
            - 3D convolution layer
            - batch normalization
            - activation

            Usages:
            -------
            - full:  self._conv3d_bn_act(x, 32, 3, 1, "cba", same", True)
            - short: self._conv3d_bn_act(x, 32, 3, 1, "cba")

            Inputs:
            -------
            - x: tensor, input layer
            - filters: int, the number of kernels
            - kernel_size: int, the size of cube kernel
            - strides: int, strides along three dimentions
            - name: string, layer's name
            - padding: string, "same" or "valid"
            - act: string or None, indicates the activation,
                   method, id None, return inactivated layer

            Output:
            - a convoluted, normalized and activated (if not None) layer

        '''

        with tf.variable_scope(name):
            cba = self._conv3d(x, filters, kernel_size, strides, padding)
            cba = self._batch_norm(cba)
            if act:  # If act is None, return inactivated result
                cba = self._activate(cba)

        return cba

    def _fc_bn_act(self, x, units, name="fba"):
        '''_FULLY_CONNECTED

            Full:  self._fc_bn_act(x, 128, "fba")
            Short: self._fc_bn_act(x, 128)

        '''

        with tf.variable_scope(name):
            fba = self._fully_connected(x, units)
            fba = self._batch_norm(fba)
            fba = self._activate(fba)

        return fba

    def _max_pool(self, x, psize=2, name="max_pool"):
        '''_MAX_POOL

            Full:  self._max_pool(x, 2, 2, "max_pool")
            Short: self._max_pool(x)

        '''

        return tf.layers.max_pooling3d(inputs=x,
                                       pool_size=psize,
                                       strides=psize,
                                       name=name)

    def _average_pool(self, x, psize=2, name="avg_pool"):
        '''_AVERAGE_POOL

            Full:  self._average_pool(x, 2, 2, "avg_pool")
            Short: self._average_pool(x)

        '''

        return tf.layers.average_pooling3d(inputs=x,
                                           pool_size=psize,
                                           strides=psize,
                                           name=name)

    def _flatten(self, x, name="flt"):
        '''_FLATTEN

            Full:  self._flatten(x, "flatten")
            Short: self._flatten(x)

        '''

        x_shape = x.get_shape().as_list()
        f_shape = reduce(mul, x_shape[1:], 1)

        with tf.name_scope("flatten"):
            return tf.reshape(tensor=x, shape=[-1, f_shape], name=name)

    def _dropout(self, x, name="dropout"):
        '''_DROP_OUT

            Full:  self._dropout(x, "dropout")

        '''

        return tf.layers.dropout(inputs=x, rate=self.drop_rate,
                                 training=self.training, name=name)

    def _logits_fc(self, x, name="logits"):
        '''_LOGITS_FC

            Full:  self._logits_fc(x, "logits")
            Short: self._logits_fc(x)

        '''

        with tf.variable_scope(name):
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

        return self._activate(res)

    #
    # Helpher functions for dense cnn
    #

    def _composite(self, x, filters, kernel_size=3, name="composite"):
        '''_COPOSITE
        '''

        comp = self._batch_norm(x)
        comp = self._activate(comp)
        comp = self._conv3d(comp, filters, kernel_size)
        comp = self._dropout(comp)

        return comp

    def _bottleneck(self, x, filters, name="bottleneck"):
        '''_BOTTLENECK
        '''

        bott = self._batch_norm(x)
        bott = self._activate(bott)
        bott = self._conv3d(bott, filters * 4, 1, padding="valid")
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
        tran = self._average_pool(tran, 2, name + "_avgpool")

        return tran

    def _last_transition(self, x):
        '''_LOGITS_DENSE
        '''

        with tf.variable_scope("last_trans"):
            last_tran = self._batch_norm(x)
            last_tran = self._activate(last_tran)

        out_channels = last_tran.get_shape().as_list()[-1]
        last_tran = self._average_pool(last_tran, out_channels, "global_avgpool")

        return last_tran

    #
    # A Simple Test Case
    #

    def _test(self):
        '''_TEST
        '''

        self.training = True

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

    def cnn(self, x, training):
        '''CNN

        '''

        self.training = training

        # Here is a very simple case to test btc_train first
        net = self._conv3d_bn_act(x, 1, 3, 1, "layer1")
        net = self._max_pool(net, 2, "max_pool1")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer2")
        net = self._max_pool(net, 2, "max_pool2")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer3")
        net = self._max_pool(net, 2, "max_pool3")
        net = self._flatten(net, "flatten")
        net = self._fc_bn_act(net, 3, "fc1")
        net = self._dropout(net, "dropout1")
        net = self._fc_bn_act(net, 3, "fc2")
        net = self._dropout(net, "dropout2")
        net = self._logits_fc(net, "logits")

        return net

    def full_cnn(self, x, training):
        '''FULL_CNN
        '''

        self.training = training

        # Here is a very simple case to test btc_train first
        net = self._conv3d_bn_act(x, 1, 3, 1, "layer1")
        net = self._max_pool(net, 2, "max_pool1")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer2")
        net = self._max_pool(net, 2, "max_pool2")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer3")
        net = self._max_pool(net, 2, "max_pool3")
        net = self._logits_conv(net, "logits_conv")
        net = self._flatten(net, "logits_flatten")

        return net

    def res_cnn(self, x, training):
        '''RES_CNN
        '''

        self.training = training

        # Here is a very simple case to test btc_train first
        net = self._conv3d_bn_act(x, 1, 5, 2, "preconv")
        net = self._res_block(net, [1, 1, 1], 2, "res1")
        net = self._res_block(net, [1, 1, 2], 2, "res2")
        net = self._max_pool(net, 7, "max_pool")
        net = self._flatten(net, "flatten")
        net = self._logits_fc(net, "logits")

        return net

    def dense_cnn(self, x, training):
        '''DENSE_NET
        '''

        self.training = training
        self.bc = True

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

    models = BTCModels("test", classes=3, act="relu", alpha=None,
                       momentum=0.99, drop_rate=0.5)
    models._test()
