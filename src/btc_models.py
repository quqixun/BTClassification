# Brain Tumor Classification
# Script for Creating Models
# Author: Qixun Qu
# Create on: 2017/10/12
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

Class BTCModels

-1- Define several basic helper functions, which are the
    basic modules to build simple CNN models.
-2- Define several specific helper functions to construct
    CNN models with more complicate structures.
-3- Build models: CNN, Full-CNN, Res-CNN and Dense-CNN.

'''


from __future__ import print_function

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
        self.is_training = None

        # A symbol for bottleneck in dense cnn
        self.bc = None

        return

    #
    # Basic Helper Functions
    #

    def _conv3d(self, x, filters, kernel_size, strides=1, name="conv_var"):
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
            - name: string, layer's name

            Output:
            -------
            - a 3D convolution layer

        '''

        padding = "valid" if kernel_size == 1 else "same"

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
            Momentum and symbol of is_training have been
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
                                                is_training=self.is_training,
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
                f1 = 0.5 * (1 + self.alpha)
                f2 = 0.5 * (1 - self.alpha)
                return f1 * x + f2 * tf.abs(x)
                # alpha = 0.2 if self.alpha is None else self.alpha
                # return tf.nn.leaky_relu(x, alpha, name)
            elif self.act == "sigmoid":
                return tf.nn.sigmoid(x, name)
            elif self.act == "tanh":
                return tf.nn.tanh(x, name)
            else:  # Raise error if activation method cannot be found
                raise ValueError("Could not find act in ['relu', 'lrelu', 'sigmoid', 'tanh']")

        return

    def _conv3d_bn_act(self, x, filters, kernel_size,
                       strides=1, name="cba", act=True):
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
            - act: string or None, indicates the activation,
                   method, id None, return inactivated layer

            Output:
            - a convoluted, normalized and activated (if not None) layer

        '''

        with tf.variable_scope(name):
            cba = self._conv3d(x, filters, kernel_size, strides)
            cba = self._batch_norm(cba)
            if act:  # If act is None, return inactivated result
                cba = self._activate(cba)

        return cba

    def _fc_bn_act(self, x, units, name="fba"):
        '''_FC_BN_ACT

            A fully connected block, including three sections:
            - full connected layer with given units
            - batch normalization
            - activation

            Usages:
            -------
            - full:  self._fc_bn_act(x, 128, "fba")
            - short: self._fc_bn_act(x, 128)

            Inputs:
            -------
            - x: tensor, input layer
            - units: int, the number of neurons
            - name: string, layer's name

            Output:
            -------
            - a fully connected, normalized and activated layer

        '''

        with tf.variable_scope(name):
            fba = self._fully_connected(x, units)
            fba = self._batch_norm(fba)
            fba = self._activate(fba)

        return fba

    def _max_pool(self, x, psize=2, name="max_pool"):
        '''_MAX_POOL

            3D max pooling layer.

            Usages:
            -------
            - full:  self._max_pool(x, 2, 2, "max_pool")
            - short: self._max_pool(x)

            Inputs:
            -------
            - x: tensor, input layer
            - psize: int or a list of 3 ints,
                     the size of pooling window, and the
                     strides of pooling operation as well,
                     if it equals to -1, the function performs
                     global max pooling
            name: string, layer's name

            Output:
            -------
            - the layer after max pooling

        '''

        if psize == -1:
            psize = x.get_shape().as_list()[1:-1]

        return tf.layers.max_pooling3d(inputs=x,
                                       pool_size=psize,
                                       strides=psize,
                                       padding="same",
                                       name=name)

    def _average_pool(self, x, psize=2, name="avg_pool"):
        '''_AVERAGE_POOL

            3D average pooling layer.

            Usages:
            -------
            - full:  self._average_pool(x, 2, 2, "avg_pool")
            - short: self._average_pool(x)

            Inputs:
            -------
            - x: tensor, input layer
            - psize: int or a list of 3 ints,
                     the size of pooling window, and the
                     strides of pooling operation as well,
                     if it equals to -1, the function performs
                     global max pooling
            name: string, layer's name

            Output:
            -------
            - the layer after average pooling

        '''

        if psize == -1:
            psize = x.get_shape().as_list()[1:-1]

        return tf.layers.average_pooling3d(inputs=x,
                                           pool_size=psize,
                                           strides=psize,
                                           padding="same",
                                           name=name)

    def _pooling(self, x, psize=2, mode="max", name="pool"):
        '''_POOLING
        '''

        if mode == "max":
            pool = self._max_pool
        elif mode == "avg":
            pool = self._average_pool
        else:  # Could not find pooling method
            raise ValueError("Pooling mode is 'max' or 'avg'.")

        return pool(x, psize, name)

    def _flatten(self, x, name="flt"):
        '''_FLATTEN

            Flatten 5D tensor into 1D tensor.

            Usages:
            -------
            - full:  self._flatten(x, "flatten")
            - short: self._flatten(x)

            Inputs:
            -------
            - x: 5D tensor, input layer
            - name: string, layer's name

            Output:
            -------
            - a flattened layer

        '''

        # Obtain the number of features contained in
        # the input layer
        x_shape = x.get_shape().as_list()
        f_shape = reduce(mul, x_shape[1:], 1)

        with tf.name_scope("flatten"):
            return tf.reshape(tensor=x, shape=[-1, f_shape], name=name)

    def _dropout(self, x, name="dropout"):
        '''_DROP_OUT

            Apply dropout to the input tensor.
            Drop rate has been set while creating the instance.
            If the is_training symbol is True, apply dropout to the input;
            if not, the untouched input will be returned.

            Usage:
            ------
            - full:  self._dropout(x, "dropout")

            Inputs:
            - x: tensor in 5D or 1D, input layer
            - name: string, layer's name

            Output:
            -------
            - the dropout layer or untouched layer

        '''

        return tf.layers.dropout(inputs=x, rate=self.drop_rate,
                                 training=self.is_training, name=name)

    def _logits_fc(self, x, name="logits"):
        '''_LOGITS_FC

            Generate logits by fully conneted layer.
            The output size is equal to the number of classes.

            Usages:
            -------
            - full:  self._logits_fc(x, "logits")
            - short: self._logits_fc(x)

            Inputs:
            -------
            - x: tensor, input layer
            - name: layer's name

            Output:
            - logit of each class

        '''

        with tf.variable_scope(name):
            return self._fully_connected(x, self.classes, name)

    #
    # Helper function for full cnn
    #

    def _logits_conv(self, x, name="logits"):
        '''_LOGITS_CONV

            Generate logits by convolutional layer.
            The output size is equal to the number of classes.

            Usages:
            -------
            - full:  self._logits_conv(x, "logits")
            - short: self._logits_conv(x)

            Inputs:
            -------
            - x: tensor, input layer
            - name: layer's name

            Output:
            - logit of each class

        '''

        x_shape = x.get_shape().as_list()
        with tf.variable_scope(name):
            return self._conv3d(x, self.classes, x_shape[1:-1], 1)

    #
    # Helper function for residual cnn
    #

    def _res_block(self, x, filters, strides=1, name="res"):
        '''_RES_BLOCK

            The basic bloack for residual network.
            - check whether shortcut is necessary
            - three convolutional layers
            - obtain shortcut if necessary
            - elementwisely sum convoluted result and original
              inputs (or shortcut if necessary)

            Usage:
            -------
            - full: self._res_block(x, [8, 16, 32], 1, "res")

            Inputs:
            -------
            - x: tensor, input layer
            - filters: list with three ints, indicates the number
                       of filters of each convolutional layer
            - strides: int, strides along three dimentions for the
                       first convolution layer or the shortcut
            - name: string, layer's name

            Output:
            -------
            - a tensor after one residual block

        '''

        # As default, shortcut is unnecessary
        shortcut = False

        # If the shape of output is not same as the input's,
        # now, shortcut has to be obtained
        if (x.get_shape().as_list()[-1] != filters[2]) or strides != 1:
            shortcut = True

        # Three convolutional layers
        # Note: the strides of first layer can be changed
        res = self._conv3d_bn_act(x, filters[0], 1, strides, name + "_conv1")
        res = self._conv3d_bn_act(res, filters[1], 3, 1, name + "_conv2")
        # Note: the third layer is inactivated
        res = self._conv3d_bn_act(res, filters[2], 1, 1, name + "_conv3", False)

        # Shortcut layer is inactivated
        if shortcut:
            x = self._conv3d_bn_act(x, filters[2], 1, strides,
                                    name + "_shortcut", False)

        # Elementwisely add
        with tf.name_scope(name + "_add"):
            res = tf.add(res, x)

        # Return the activated summation
        return self._activate(res)

    #
    # Helpher functions for dense cnn
    #

    def _dense_block(self, x, growth_rate,
                     internals, name="dense_block"):
        '''_DENSE_BLOCK

            The basic block of dense network.
            The struction of one block:
            --- dense block
             |--- internal1
               |--- bottleneck (if self.bs is true)
               |--- composite
               |--- concatenate
             |--- internal2
               |--- same as internal1
             ...
             |--- internaln
               |--- same as internal1

            Usage:
            ------
            - full: self._dense_block(x, 16, 4, "block1")

            Inputs:
            -------
            - x: tensor, input layer
            - growth_rate: int, the number of kernels in
                           each internal section
            - internals: int, the number of internals
            - name: string, block's name

            Output:
            -------
            - a dense block

        '''

        dense = x

        # Combine all internals
        for internal in range(internals):
            dense = self._dense_internal(dense, growth_rate,
                                         str(internal + 1), name)

        return dense

    def _dense_internal(self, x, growth_rate, no, name):
        '''_DENSE_INTERNAL

            Internal section of a dense block.
            - bottleneck (if self.bc is True)
            - composite
            - concate

            Usage:
            ------
            - full: self._dense_internal(x, 16, 1, "block1")

            Inputs:
            -------
            - x: tensor, input layer
            - growth_rate: int, the number of kernels
            - no: string, internal number
            - name: string, block's name

            Output:
            -------
            - one internal section of one dense block

        '''

        dint = x

        # Obtain bottleneck section
        if self.bc:
            with tf.variable_scope(name + "_bott" + no):
                dint = self._bottleneck(x, growth_rate)

        # Obtain composite section
        with tf.variable_scope(name + "_comp" + no):
            dint = self._composite(dint, growth_rate, 3)

        # Concatenate original input (or bottleneck section)
        # with composite section
        with tf.name_scope(name + "_concat" + no):
            dint = tf.concat((x, dint), 4)

        return dint

    def _bottleneck(self, x, filters, name="bottleneck"):
        '''_BOTTLENECK

            Bottleneck section to reduce the number of input
            features to improve computational efficiency.
            - batch normalization
            - activation
            - convolution
            - dropout if self.is_training is a True placeholder

            Usage:
            ------
            - full: self._bottleneck(x, 16, "bottleneck")

            Inputs:
            -------
            - x: tensor, input layer
            - filters: int, also known as growth rate, the number of
                       filters in composite section
            - name: string, section's name

            Output:
            -------
            - the bottleneck layer

        '''

        bott = self._batch_norm(x)
        bott = self._activate(bott)
        bott = self._conv3d(bott, filters * 4, 1)
        bott = self._dropout(bott)

        return bott

    def _composite(self, x, filters, kernel_size=3, name="composite"):
        '''_COPOSITE

            The convolutional section of dense block.
            - batch normalization
            - activation
            - convolution
            - dropout if self.is_training is a True placeholder

            Usage:
            - full: self._composite(x, 16, 3, "composite")

            Inputs:
            -------
            - x: tensor, input layer
            - filters: int, also known as growth rate,
                       the number of filters
            - kernel_size: int, the size of kernels
            - name: string, section's name

            output:
            -------
            - a composite section

        '''

        comp = self._batch_norm(x)
        comp = self._activate(comp)
        comp = self._conv3d(comp, filters, kernel_size)
        comp = self._dropout(comp)

        return comp

    def _transition(self, x, name="transition"):
        '''_TRANSITION

            The transition layer between two dense blocks.
            - batch normalization
            - activation
            - convolution
            - dropout if self.is_training is a True placeholder
            - average pooling

            Usage:
            ------
            - full: self._transition(x, "trans")

            Inputs:
            -------
            - x: tensor, input layer
            - name: string, section's name

            Output:
            - a tensor after average pooling

        '''

        out_channels = x.get_shape().as_list()[-1]
        with tf.variable_scope(name + "_comp"):
            tran = self._composite(x, out_channels, 1)
        tran = self._average_pool(tran, 2, name + "_avgpool")

        return tran

    def _last_transition(self, x, name="global_avgpool"):
        '''_LAST_TRANSITION

            The last transition section before logits layer.
            - batch normalization
            - activation
            - global average polling

            Usage:
            ------
            - full: self._last_transition(x, "trans")

            Inputs:
            -------
            - x: tensor, input layer
            - name: string, section's name

            Output:
            - a tensor after global average pooling

        '''

        with tf.variable_scope("last_trans"):
            last_tran = self._batch_norm(x)
            last_tran = self._activate(last_tran)

        last_tran = self._average_pool(last_tran, -1, name)

        return last_tran

    #
    # Helper functions for autoencoder
    #

    def _deconv3d(self, x, filters, kernel_size, strides=1, name="deconv_var"):
        '''_DECONV3D
        '''

        padding = "valid" if kernel_size == 1 else "same"

        with tf.name_scope("deconv3d"):
            return tf.layers.conv3d_transpose(inputs=x,
                                              filters=filters,
                                              kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              kernel_initializer=xavier_initializer(),
                                              name=name)

    def _deconv3d_bn_act(self, x, filters, kernel_size,
                         strides=1, name="dba", act=True):
        '''_DECONV3D_BN_ACT
        '''

        with tf.variable_scope(name):
            dba = self._deconv3d(x, filters, kernel_size, strides)
            dba = self._batch_norm(dba)
            if act:  # If False, return inactivated result
                dba = self._activate(dba)

        return dba

    #
    # A Simple Test Case
    #

    def _test(self):
        '''_TEST

            A function to test basic helpers.

        '''

        self.is_training = True

        x = tf.placeholder(tf.float32, [5, 49, 49, 49, 4], "input")
        net = self._conv3d_bn_act(x, 2, 3, 1, "layer1")
        net = self._max_pool(net, 2, "max_pool1")
        net = self._conv3d_bn_act(net, 2, 3, 1, "layer2")
        net = self._average_pool(net, 2, "avg_pool2")
        net = self._conv3d_bn_act(net, 2, 3, 1, "layer3")
        net = self._max_pool(net, 2, "max_pool3")
        net = self._flatten(net, "flatten")
        net = self._fc_bn_act(net, 64, "fcn1")
        net = self._dropout(net, "drop1")
        net = self._fc_bn_act(net, 64, "fcn2")
        net = self._dropout(net, "drop2")
        net = self._logits_fc(net, "logits")
        net = tf.nn.softmax(logits=net, name="softmax")

        print("Simple test of Class BTCModels")
        print("Input 5 volumes in 3 classes")
        print("Output probabilities' shape: ", net.shape)

        return

    #
    # Contruct Models
    #

    def cnn(self, x, is_training):
        '''CNN

            VGG-like CNN model.

            Inputs:
            -------
            - x: tensor placeholder, input volumes in batch
            - is_training: boolean placeholder, indicates the mode,
                           True: training mode,
                           False: validating and inferencing mode

            Output:
            -------
            - output logits after VGG-like CNN

        '''

        self.is_training = is_training

        # Here is a very simple case to test btc_train first
        net = self._conv3d_bn_act(x, 1, 3, 1, "layer1")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer2")
        net = self._pooling(net, 2, "max", "max_pool1")
        net = self._dropout(net, "dropout1")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer3")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer4")
        net = self._pooling(net, 2, "max", "max_pool2")
        net = self._dropout(net, "dropout2")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer5")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer6")
        net = self._pooling(net, 2, "max", "max_pool3")
        net = self._dropout(net, "dropout3")
        net = self._flatten(net, "flatten")
        net = self._fc_bn_act(net, 3, "fc1")
        net = self._dropout(net, "dropout4")
        net = self._fc_bn_act(net, 3, "fc2")
        net = self._dropout(net, "dropout5")
        net = self._logits_fc(net, "logits")

        return net

    def full_cnn(self, x, is_training):
        '''FULL_CNN

            CNN with convolutional logits layer, without
            fully connected layers.

            Inputs:
            -------
            - x: tensor placeholder, input volumes in batch
            - is_training: boolean placeholder, indicates the mode,
                           True: training mode,
                           False: validating and inferencing mode

            Output:
            -------
            - output logits after Fully CNN

        '''

        self.is_training = is_training

        # Here is a very simple case to test btc_train first
        net = self._conv3d_bn_act(x, 1, 3, 1, "layer1")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer2")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer3")
        net = self._pooling(net, 2, "max", "max_pool1")
        net = self._dropout(net, "dropout1")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer4")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer5")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer6")
        net = self._pooling(net, 2, "max", "max_pool2")
        net = self._dropout(net, "dropout2")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer7")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer8")
        net = self._conv3d_bn_act(net, 1, 3, 1, "layer9")
        net = self._pooling(net, -1, "max", "max_pool3")
        net = self._dropout(net, "dropout3")
        net = self._logits_conv(net, "logits_conv")
        net = self._flatten(net, "logits_flatten")

        return net

    def res_cnn(self, x, is_training):
        '''RES_CNN

            Residual CNN (ResNet).

            Inputs:
            -------
            - x: tensor placeholder, input volumes in batch
            - is_training: boolean placeholder, indicates the mode,
                           True: training mode,
                           False: validating and inferencing mode

            Output:
            -------
            - output logits after ResNet

        '''

        self.is_training = is_training

        # Here is a very simple case to test btc_train first
        net = self._conv3d_bn_act(x, 1, 5, 2, "preconv")
        net = self._res_block(net, [1, 1, 1], 2, "res1")
        net = self._res_block(net, [1, 1, 2], 2, "res2")
        net = self._pooling(net, -1, "max", "global_maxpool")
        net = self._flatten(net, "flatten")
        net = self._logits_fc(net, "logits")

        return net

    def dense_cnn(self, x, is_training):
        '''DENSE_NET

            Densely CNN (DenseNet).

            Inputs:
            -------
            - x: tensor placeholder, input volumes in batch
            - is_training: boolean placeholder, indicates the mode,
                           True: training mode,
                           False: validating and inferencing mode

            Output:
            -------
            - output logits after DenseNet

        '''

        self.is_training = is_training

        # Set the bottleneck symbol
        self.bc = True

        # Here is a very simple case to test btc_train first
        # Preconv layer before dense block
        with tf.variable_scope("preconv"):
            net = self._conv3d(x, 1, 5, 2)
        net = self._dense_block(net, 1, 2, "dense1")
        net = self._transition(net, "trans1")
        net = self._dense_block(net, 1, 2, "dense2")
        net = self._last_transition(net, "global_avgpool")
        net = self._flatten(net, "flatten")
        net = self._logits_fc(net, "logits")

        return net

    def autoencoder_stride(self, x, is_training):
        '''AUTOENCODER_STRIDE

            Autoencoder with stride pooling.

            Inputs:
            -------
            - x: tensor placeholder, input volumes in batch
            - is_training: boolean placeholder, indicates the mode,
                           True: training mode,
                           False: validating and inferencing mode

            Output:
            -------
            - a tensor whose shape is same as input

        '''

        self.is_training = is_training

        net = self._conv3d_bn_act(x, 6, 5, 2, "conv1")
        net = self._dropout(net, "dropout1")
        net = self._conv3d_bn_act(net, 9, 5, 2, "conv2")
        net = self._dropout(net, "dropout2")
        net = self._conv3d_bn_act(net, 12, 5, 2, "conv3")
        net = self._dropout(net, "dropout3")
        net = self._deconv3d_bn_act(net, 9, 5, 2, "deconv1")
        net = self._dropout(net, "dropout4")
        net = self._deconv3d_bn_act(net, 6, 5, 2, "deconv2")
        net = self._dropout(net, "dropout5")
        net = self._deconv3d_bn_act(net, 1, 5, 2, "deconv3")

        return net

    def autoencoder_pool(self, x, is_training):
        '''AUTOENCODER_POOL

            Autoencoder with max or average pooling.

            Inputs:
            -------
            - x: tensor placeholder, input volumes in batch
            - is_training: boolean placeholder, indicates the mode,
                           True: training mode,
                           False: validating and inferencing mode

            Output:
            -------
            - a tensor whose shape is same as input

        '''

        self.is_training = is_training

        net = self._conv3d_bn_act(x, 6, 3, 1, "conv1")
        net = self._pooling(net, 2, "max", "max_pool1")
        net = self._conv3d_bn_act(net, 6, 3, 1, "conv2")
        net = self._pooling(net, 2, "max", "max_pool2")
        net = self._conv3d_bn_act(net, 6, 3, 1, "conv3")
        net = self._pooling(net, 2, "max", "max_pool3")
        net = self._deconv3d_bn_act(net, 6, 3, 2, "deconv1")
        net = self._deconv3d_bn_act(net, 6, 3, 2, "deconv2")
        net = self._deconv3d_bn_act(net, 4, 3, 2, "deconv3")

        return net


if __name__ == "__main__":

    models = BTCModels("test", classes=3, act="relu", alpha=None,
                       momentum=0.99, drop_rate=0.5)

    # Test basic helper functions
    # models._test()

    # Test function for cnn, full_cnn, res_cnn, dense_cnn and autoencoder
    x = tf.placeholder(tf.float32, [32, 112, 112, 88, 4])
    is_training = tf.placeholder(tf.bool, [])

    # net = models.cnn(x, is_training)
    # net = models.full_cnn(x, is_training)
    # net = models.res_cnn(x, is_training)
    # net = models.dense_cnn(x, is_training)
    net = models.autoencoder_stride(x, is_training)
    # net = models.autoencoder_pool(x, is_training)
