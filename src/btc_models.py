# Brain Tumor Classification
# Script for Creating Models
# Author: Qixun Qu
# Create on: 2017/10/12
# Modify on: 2017/11/28

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
-3- Build models: CNN, Full-CNN, Res-CNN and Dense-CNN,
    and sparsity autoencoder with either KL constraint and
    Winner-Take-All constraint.

'''


from __future__ import print_function

import tensorflow as tf
from operator import mul
from functools import reduce
from tensorflow.contrib.layers import xavier_initializer


class BTCModels():

    def __init__(self, classes, act="relu", alpha=None,
                 momentum=0.99, drop_rate=0.5, dims="3d",
                 cae_pool=None, lifetime_rate=None):
        '''__INIT__

            Initialization of BTCModels. In this functions,
            commen parameters of all models are set first.

            Inputs:
            -------
            - classes: int, the number of grading groups
            - act: string, indicate the activation method by either
                   "relu" or "lrelu" (leaky relu)
            - alpha: float, slope of the leaky relu at x < 0
            - momentum: float, momentum for removing average in
                        batch normalization, typically values are
                        0.999, 0.99, 0.9, etc
            - drop_rate: float, rate of dropout of input units,
                         which is between 0 and 1
            - dims: string, "3d" ("3D") or "2d" ("2D")
            - cae_pool: sreing, "stride" or "pool"
            - lifetime_rate: float, the percentage of how many
                             sparsity code are kept in autoencoder

        '''

        # The number of classes
        self.classes = classes

        # Settings for activation
        self.act = act
        self.alpha = alpha

        # Settings for batch normalization
        self.momentum = momentum
        self.drop_rate = drop_rate

        # Set functions to construct models according to
        # the dimentions of input tensor
        self.dims = dims
        if dims == "3d" or dims == "3D":
            self.conv_func = tf.layers.conv3d
            self.deconv_func = tf.layers.conv3d_transpose
            self.max_pool_func = tf.layers.max_pooling3d
            self.avg_pool_func = tf.layers.average_pooling3d
            self.concat_axis = 4
            self.right_dims = 5
        elif dims == "2d" or dims == "2D":
            self.conv_func = tf.layers.conv2d
            self.deconv_func = tf.layers.conv2d_transpose
            self.max_pool_func = tf.layers.max_pooling2d
            self.avg_pool_func = tf.layers.average_pooling2d
            self.concat_axis = 3
            self.right_dims = 4
        else:
            raise ValueError("Cannot found dimentions in '2d' or '3d'.")

        # Set encoder function
        self.encoder = None
        if cae_pool is not None:
            if cae_pool == "stride":
                self.encoder = self._encoder_stride
            elif cae_pool == "pool":
                self.encoder = self._encoder_pool
            else:
                raise ValueError("Cannot found pool method in 'stride' or 'pool'.")

        # A symbol to indicate whether the model is used to train
        # The symbol will be assigned as a placeholder while
        # feeding the model in training and validating steps
        self.is_training = None

        # A symbol for bottleneck in dense cnn
        self.bc = None

        # Set lifetime rate for autoencoder with
        # Winner-Take-All constraint
        self.lifetime_rate = lifetime_rate

        return

    #
    # Basic Helper Functions
    #

    def _conv(self, x, filters, kernel_size, strides=1,
              name="conv_var"):
        '''_CONV

            Return 3D or 2D convolutional tensor with variables
            initialized by xavier method.

            Usages:
            -------
            - full:  self._conv(x, 32, 3, 1, "conv")
            - short: self._conv(x, 32, 3)

            Inputs:
            -------
            - x: tensor, input tensor
            - filters: int, the number of kernels
            - kernel_size: int, the size of kernel
            - strides: int, strides along dimentions
            - name: string, layer's name

            Output:
            -------
            - a 3D or 2D convolutional tensor

        '''

        padding = "valid" if kernel_size == 1 else "same"

        with tf.name_scope("conv"):
            return self.conv_func(inputs=x,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  kernel_initializer=xavier_initializer(),
                                  name=name)

    def _fully_connected(self, x, units, name="fc_var"):
        '''_FULLY_CONNECTED

            Return fully connected tensor with variables
            initialized by xavier method.

            Usages:
            -------
            - full:  self._fully_connected(x, 128, "fc")
            - short: self._fully_connected(x, 128)

            Inputs:
            -------
            - x: tensor, input tensor
            - units: int, the number of neurons
            - name: string, layer's name

            Outputs:
            --------
            - a fully connected tensor

        '''

        with tf.name_scope("full_connection"):
            return tf.layers.dense(inputs=x,
                                   units=units,
                                   kernel_initializer=xavier_initializer(),
                                   name=name)

    def _batch_norm(self, x, name="bn_var"):
        '''_BATCH_NORM

            Normalize the input tensor.
            Momentum and symbol of is_training have been
            assigned while the class is initialized.

            Usages:
            -------
            - full:  self._batch_norm(x, "bn")
            - short: self._batch_norm(x)

            Inputs:
            -------
            - x: tensor, input tensor
            - name: string, layer's name

            Output:
            -------
            - normalized tensor

        '''

        with tf.name_scope("batch_norm"):
            return tf.contrib.layers.batch_norm(inputs=x,
                                                decay=self.momentum,
                                                is_training=self.is_training,
                                                scope=name)

    def _activate(self, x, name="act"):
        '''_ACTIVATE

           Activate input tensor. Several approaches can be available,
           which are ReLU, leaky ReLU, Sigmoid or Tanh.
           Activation method and setting has been set while the
           class is initialized.

           Usages:
           -------
           - full:  self._activate(x, "act")
           - short: self._activate(x)

           Inputs:
           -------
           - x: tensor, input tensor
           - name: string, layer's name

           Output:
           -------
           - an activated tensor

        '''

        with tf.name_scope("activate"):
            if self.act == "relu":
                return tf.nn.relu(x, name)
            elif self.act == "lrelu":
                f1 = 0.5 * (1 + self.alpha)
                f2 = 0.5 * (1 - self.alpha)
                return f1 * x + f2 * tf.abs(x)
            elif self.act == "sigmoid":
                return tf.nn.sigmoid(x, name)
            elif self.act == "tanh":
                return tf.nn.tanh(x, name)
            else:  # Raise error if activation method cannot be found
                raise ValueError("Could not find act in ['relu', 'lrelu', 'sigmoid', 'tanh']")

        return

    def _conv_bn_act(self, x, filters, kernel_size, strides=1,
                     name="cba", act=True):
        '''_CONV_BN_ACT

            A convolution block, including three sections:
            - 3D or 2D convolution layer
            - batch normalization
            - activation

            Usages:
            -------
            - full:  self._conv_bn_act(x, 32, 3, 1, "cba", True)
            - short: self._conv_bn_act(x, 32, 3, 1, "cba")

            Inputs:
            -------
            - x: tensor, input tensor
            - filters: int, the number of kernels
            - kernel_size: int, the size of kernel
            - strides: int, strides along dimentions
            - name: string, layer's name
            - act: string or None, indicates the activation,
                   method, if None, return inactivated layer

            Output:
            - a convoluted, normalized and activated (if not None) tensor

        '''

        with tf.variable_scope(name):
            cba = self._conv(x, filters, kernel_size, strides)
            cba = self._batch_norm(cba)
            if act:  # If act is None, return inactivated tensor
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
            - x: tensor, input tensor
            - units: int, the number of neurons
            - name: string, layer's name

            Output:
            -------
            - a fully connected, normalized and activated tensor

        '''

        with tf.variable_scope(name):
            fba = self._fully_connected(x, units)
            fba = self._batch_norm(fba)
            fba = self._activate(fba)

        return fba

    def _max_pool(self, x, psize=2, name="max_pool"):
        '''_MAX_POOL

            3D or 2D max pooling layer.

            Usages:
            -------
            - full:  self._max_pool(x, 2, 2, "max_pool")
            - short: self._max_pool(x)

            Inputs:
            -------
            - x: tensor, input tensor
            - psize: int or a list of ints,
                     the size of pooling window, and the
                     strides of pooling operation as well,
                     if it equals to -1, the function performs
                     global max pooling
            name: string, layer's name

            Output:
            -------
            - the tensor after max pooling

        '''

        # Global max pooling if psize is -1
        if psize == -1:
            psize = x.get_shape().as_list()[1:-1]

        return self.max_pool_func(inputs=x,
                                  pool_size=psize,
                                  strides=psize,
                                  padding="same",
                                  name=name)

    def _average_pool(self, x, psize=2, name="avg_pool"):
        '''_AVERAGE_POOL

            3D or 2D average pooling layer.

            Usages:
            -------
            - full:  self._average_pool(x, 2, 2, "avg_pool")
            - short: self._average_pool(x)

            Inputs:
            -------
            - x: tensor, input tensor
            - psize: int or a list of ints,
                     the size of pooling window, and the
                     strides of pooling operation as well,
                     if it equals to -1, the function performs
                     global max pooling
            name: string, layer's name

            Output:
            -------
            - the tensor after average pooling

        '''

        # Global acerage pooling if psize is -1
        if psize == -1:
            psize = x.get_shape().as_list()[1:-1]

        return self.avg_pool_func(inputs=x, pool_size=psize,
                                  strides=psize, padding="same",
                                  name=name)

    def _pooling(self, x, psize=2, mode="max", name="pool"):
        '''_POOLING

            Apply pooling method on input tensor with either
            max pooling or average pooling.

            Inputs:
            -------
            - x: tensor, the input tensor
            - psize: int or a list of ints,
                     the size of pooling window, and the
                     strides of pooling operation as well,
                     if it equals to -1, the function performs
                     global max pooling
            - mode: string, "max" or "avg"
            - name: string, layer's name

            Output:
            -------
            - the tensor after pooling

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

            Flatten 5D or 4D tensor into 1D tensor.

            Usages:
            -------
            - full:  self._flatten(x, "flatten")
            - short: self._flatten(x)

            Inputs:
            -------
            - x: 5D or 4D tensor, input tensor
            - name: string, layer's name

            Output:
            -------
            - a flattened tensor

        '''

        # Obtain the number of features
        # contained in the input layer
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
            - x: tensor in 5D, 4D or 1D, input tensor
            - name: string, layer's name

            Output:
            -------
            - the dropout layer or untouched tensor

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
            - x: tensor, input tensor
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

            Generate logits by convolutional tensor.
            The output size is equal to the number of classes.

            Usages:
            -------
            - full:  self._logits_conv(x, "logits")
            - short: self._logits_conv(x)

            Inputs:
            -------
            - x: tensor, input tensor
            - name: layer's name

            Output:
            - logit of each class

        '''

        x_shape = x.get_shape().as_list()
        with tf.variable_scope(name):
            return self._conv(x, self.classes, x_shape[1:-1], 1)

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
            - x: tensor, input tensor
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
        res = self._conv_bn_act(x, filters[0], 1, strides, name + "_conv1")
        res = self._conv_bn_act(res, filters[1], 3, 1, name + "_conv2")
        # Note: the third layer is inactivated
        res = self._conv_bn_act(res, filters[2], 1, 1, name + "_conv3", False)

        # Shortcut layer is inactivated
        if shortcut:
            x = self._conv_bn_act(x, filters[2], 1, strides,
                                  name + "_shortcut", False)

        # Elementwisely add
        with tf.name_scope(name + "_add"):
            res = tf.add(res, x)

        # Return the activated summation
        return self._activate(res)

    #
    # Helpher functions for dense cnn
    #

    def _dense_block(self, x, growth_rate, internals, name="dense_block"):
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
            - x: tensor, input tensor
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
            - x: tensor, input tensor
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
            dint = tf.concat((x, dint), self.concat_axis)

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
            - x: tensor, input tensor
            - filters: int, also known as growth rate, the number of
                       filters in composite section
            - name: string, section's name

            Output:
            -------
            - the bottleneck layer

        '''

        bott = self._batch_norm(x)
        bott = self._activate(bott)
        bott = self._conv(bott, filters * 4, 1)
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
            - x: tensor, input tensor
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
        comp = self._conv(comp, filters, kernel_size)
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
            - x: tensor, input tensor
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
            - x: tensor, input tensor
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

    def _deconv(self, x, filters, kernel_size, strides=1, name="deconv_var"):
        '''_DECONV

            Return 3D or 2D deconvolution layer with variables
            initialized by xavier method.

            Usages:
            -------
            - full:  self._deconv(x, 32, 3, 1, "deconv")
            - short: self._deconv(x, 32, 3)

            Inputs:
            -------
            - x: tensor, input layer
            - filters: int, the number of kernels
            - kernel_size: int, the size of kernel
            - strides: int, strides along dimentions
            - name: string, layer's name

            Output:
            -------
            - a 3D or 2D deconvolution layer

        '''

        padding = "valid" if kernel_size == 1 else "same"

        with tf.name_scope("deconv"):
            return self.deconv_func(inputs=x,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    kernel_initializer=xavier_initializer(),
                                    name=name)

    def _deconv_bn_act(self, x, filters, kernel_size,
                       strides=1, name="dba", act=True):
        '''_DECONV_BN_ACT

            A deconvolution block, including three sections:
            - 3D or 2D deconvolution layer
            - batch normalization
            - activation

            Usages:
            -------
            - full:  self._deconv_bn_act(x, 32, 3, 1, "dba", True)
            - short: self._deconv_bn_act(x, 32, 3, 1, "dba")

            Inputs:
            -------
            - x: tensor, input layer
            - filters: int, the number of kernels
            - kernel_size: int, the size of kernel
            - strides: int, strides along dimentions
            - name: string, layer's name
            - act: string or None, indicates the activation,
                   method, if None, return inactivated layer

            Output:
            - a deconvoluted, normalized and activated (if not None) layer

        '''

        with tf.variable_scope(name):
            dba = self._deconv(x, filters, kernel_size, strides)
            dba = self._batch_norm(dba)
            if act:  # If False, return inactivated result
                dba = self._activate(dba)

        return dba

    def _encoder_stride(self, x):
        '''_ENCODER_STRIDE

            Encoder sectio of autoencoder.
            Each convolutional layer has strides 2.
            No other pooling methods are applied.

            Inputs:
            -------
            - x: tensor, original input

            Output:
            -------
            - compressed representation of input sample

        '''

        code = self._conv_bn_act(x, 32, 3, 2, "conv1")
        code = self._dropout(code, "en_dropout1")
        code = self._conv_bn_act(code, 128, 3, 2, "conv2")
        code = self._dropout(code, "en_dropout2")
        code = self._conv_bn_act(code, 512, 3, 2, "conv3")

        return code

    def _encoder_pool(self, x):
        '''_ENCODER_POOL

            Encoder sectio of autoencoder.
            Convolutional layer has strides 1.
            Max pooling method is applied after
            each convolutional layer.

            Inputs:
            -------
            - x: tensor, original input

            Output:
            -------
            - compressed representation of input sample

        '''

        code = self._conv_bn_act(x, 1, 3, 1, "conv1")
        code = self._pooling(code, 2, "max", "max_pool1")
        code = self._dropout(code, "en_dropout1")
        code = self._conv_bn_act(code, 1, 3, 1, "conv2")
        code = self._pooling(code, 2, "max", "max_pool2")
        code = self._dropout(code, "en_dropout2")
        code = self._conv_bn_act(code, 1, 3, 1, "conv3")
        code = self._pooling(code, 2, "max", "max_pool3")

        return code

    def _wta_constraint(self, code, k=1):
        '''_WTA_CONSTRAINT

            Winner-Take-All constraint to generate sparse
            representation by keeping largest values of
            the compression code, which consists of two steps:
            - get spatial sparsity
            - get lifetime sparsity

            Inputs:
            -------
            - code: tensor, compressed code after encoder
            - k: int, the number of largest values to be kept
            - another parameter, lifetime_rate, has been assigned
              when the instance is initialized

            Output:
            -------
            - sparse representation

        '''

        # The function to kept k largest values in
        # code, and set others to zeros.
        def spatial_sparsity(code, k):
            # Obtain the shape of code
            # n: batch size
            # c: the number of feaure maps
            #    (or the number of filters)
            shape = code.get_shape().as_list()
            n, c = shape[0], shape[-1]

            # As the input tensor could be 5D or 4D,
            # set up parameters for different inputs
            if len(shape) == 5:
                transpose_perm = [0, 4, 1, 2, 3]
                threshold_shape = [n, 1, 1, 1, c]
            elif len(shape) == 4:
                transpose_perm = [0, 3, 1, 2]
                threshold_shape = [n, 1, 1, c]
            else:
                raise ValueError("Cannot handle with the input.")

            code_transpose = tf.transpose(code, transpose_perm)
            code_reshape = tf.reshape(code_transpose, [n, c, -1])

            # Get top k values of code
            code_top_k, _ = tf.nn.top_k(code_reshape, k)
            # Get the minimum of top k values to do thresholding
            code_top_k_min = code_top_k[..., k - 1]

            # Threshold the code, the indices of top k values is 1,
            # and set others to 0
            threshold = tf.reshape(code_top_k_min, threshold_shape)
            drop_map = tf.where(code < threshold,
                                tf.zeros(shape, tf.float32),
                                tf.ones(shape, tf.float32))

            # Keep top k value in code
            code = code * drop_map
            # Save top k values as winner
            winner = tf.reshape(code_top_k, [n, c, k])

            return code, winner

        # The function to carry out lifetime sparsity.
        # For example, the batch size is 64, which means each
        # filter will lead to 64 winners, lifetime sparsity
        # is going to keep largest winners in the percentage of
        # self.lifetime_rate, and set others to zeros. Those left
        # feature maps are winners in winners.
        def lifetime_sparsity(code, winner):
            # Obtain the shape of code and winner
            # n: batch size
            # c: the number of feaure maps
            #    (or the number of filters)
            # k: the number of winners to be kept
            code_shape = code.get_shape().as_list()
            winner_shape = winner.get_shape().as_list()
            n, c = winner_shape[0], winner_shape[1]
            k = int(self.lifetime_rate * n) + 1

            # Compute mean value of each winner
            winner_mean = tf.reduce_mean(winner, axis=2)
            winner_mean = tf.transpose(winner_mean)
            # Get top k mean values of top k winners
            winner_mean_top_k, _ = tf.nn.top_k(winner_mean, k)
            # Get the minimum of top k values to do thresholding
            winner_mean_top_k_min = winner_mean_top_k[..., k - 1]
            winner_mean_top_k_min = tf.reshape(winner_mean_top_k_min, [c, 1])

            # Threshold the winner, the indices of top k winners is 1,
            # and set others to 0
            drop_map = tf.where(winner_mean < winner_mean_top_k_min,
                                tf.zeros([c, n], tf.float32),
                                tf.ones([c, n], tf.float32))
            drop_map = tf.transpose(drop_map)

            # As the input tensor could be 5D or 4D,
            # set up parameters for different inputs
            if len(code_shape) == 5:
                reform_shape = [n, 1, 1, 1, c]
            elif len(code_shape) == 4:
                reform_shape = [n, 1, 1, c]
            else:
                raise ValueError("Cannot handle with the input.")

            # Keep top k winners in code
            code = code * tf.reshape(drop_map, reform_shape)

            return code

        # Winner-Take-All constraint
        code, winner = spatial_sparsity(code, k)
        code = lifetime_sparsity(code, winner)

        return code

    def _decoder(self, code):
        '''_DECODER

            Decoder section of autoencoder to
            reconstruct code to input.

            Input:
            ------
            - code: tensor, compressed representation

            Output:
            -------
            - the reconstruction from code

        '''

        decode = self._dropout(code, "de_dropout1")
        decode = self._deconv_bn_act(decode, 128, 3, 2, "deconv1")
        decode = self._dropout(decode, "de_dropout2")
        decode = self._deconv_bn_act(decode, 32, 3, 2, "deconv2")
        decode = self._dropout(decode, "de_dropout3")
        decode = self._deconv_bn_act(decode, 4, 3, 2, "deconv3", False)
        decode = tf.nn.sigmoid(decode, "sigmoid")

        return decode

    #
    # Error Check
    #

    def _check_input(self, x):
        '''_CHECK_INPUT

            Chech the dimentions of input tensor whether
            satisfy the requirement for the model. If not,
            raise an error and quit program.

            Input:
            ------
            - x: tensor, the tensor input to the model

        '''

        # Obtain the shape of input
        x_dims = len(x.get_shape().as_list())

        if (x_dims == 5 and (self.dims == "3d" or self.dims == "3D")) or \
           (x_dims == 4 and (self.dims == "2d" or self.dims == "2D")):
            pass
        else:  # The input is unwanted
            msg = ("Your model deals with {0} data, the input tensor should be {1}D. " +
                   "But your input is {2}D.").format(self.dims, self.right_dims, x_dims)
            raise ValueError(msg)

        return

    def _check_output(self, x, output):
        '''_CHECK_OUTPUT

            Obtain the dimentions of input and ouput respectively,
            and check whether they are same. If not, raise an error
            and quit program. This check is only for AUTOENCODER.

            Inputs:
            -------
            - x: tensor, the input tensor
            - output: tensor, the output generated from model

        '''

        # Obtain dimentions
        x_dims = x.get_shape().as_list()
        output_dims = output.get_shape().as_list()

        if x_dims == output_dims:
            return
        else:  # They ate not same
            msg = ("Input tensor shape: {0}, output tensor shape: {1}. " +
                   "They should be same.").format(x_dims, output_dims)
            raise ValueError(msg)

        return

    #
    # A Simple Test Case
    #

    def test(self, x, is_training):
        '''_TEST

            A function to test basic helpers.

            Inputs:
            -------
            - x: tensor placeholder, input volumes in batch
            - is_training: boolean placeholder, indicates the mode,
                           True: training mode,
                           False: validating and inferencing mode

        '''

        self._check_input(x)
        self.is_training = is_training

        net = self._conv_bn_act(x, 2, 3, 1, "layer1")
        net = self._max_pool(net, 2, "max_pool1")
        net = self._conv_bn_act(net, 2, 3, 1, "layer2")
        net = self._average_pool(net, 2, "avg_pool2")
        net = self._conv_bn_act(net, 2, 3, 1, "layer3")
        net = self._max_pool(net, 2, "max_pool3")
        net = self._flatten(net, "flatten")
        net = self._fc_bn_act(net, 64, "fcn1")
        net = self._dropout(net, "drop1")
        net = self._fc_bn_act(net, 64, "fcn2")
        net = self._dropout(net, "drop2")
        net = self._logits_fc(net, "logits")
        net = tf.nn.softmax(logits=net, name="softmax")

        print("Simple test of Class BTCModels")
        print("Input n volumes in 3 classes")
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

        self._check_input(x)
        self.is_training = is_training

        # Here is a very simple case to test btc_train first
        net = self._conv_bn_act(x, 32, 3, 1, "layer1")
        net = self._conv_bn_act(net, 32, 3, 1, "layer2")
        net = self._pooling(net, 2, "max", "max_pool1")
        net = self._dropout(net, "dropout1")
        net = self._conv_bn_act(net, 64, 3, 1, "layer3")
        net = self._conv_bn_act(net, 64, 3, 1, "layer4")
        net = self._pooling(net, 2, "max", "max_pool2")
        net = self._dropout(net, "dropout2")
        net = self._conv_bn_act(net, 128, 3, 1, "layer5")
        net = self._conv_bn_act(net, 128, 3, 1, "layer6")
        net = self._pooling(net, 2, "max", "max_pool3")
        net = self._dropout(net, "dropout3")
        net = self._conv_bn_act(net, 256, 3, 1, "layer7")
        net = self._conv_bn_act(net, 256, 3, 1, "layer8")
        net = self._pooling(net, -1, "max", "global_maxpool")
        net = self._flatten(net, "flatten")
        net = self._dropout(net, "dropout4")
        net = self._fc_bn_act(net, 256, "fc1")
        net = self._dropout(net, "dropout5")
        net = self._fc_bn_act(net, 512, "fc2")
        net = self._dropout(net, "dropout6")
        net = self._logits_fc(net, "logits")

        return net

    def _cnn_branch(self, x, branch):
        net = self._conv_bn_act(x, 32, 5, 1, branch + "_layer1")
        net = self._pooling(net, 2, "max", branch + "maxpool1")
        net = self._conv_bn_act(net, 64, 5, 1, branch + "_layer2")
        net = self._pooling(net, 2, "max", branch + "maxpool2")
        net = self._conv_bn_act(net, 128, 5, 1, branch + "_layer3")
        net = self._pooling(net, 2, "max", branch + "maxpool3")
        net = self._conv_bn_act(net, 256, 5, 1, branch + "_layer4")

        # net = self._conv_bn_act(x, 16, 5, 1, branch + "_preconv")
        # net = self._res_block(net, [16, 32, 32], 2, branch + "_res1")
        # net = self._res_block(net, [32, 64, 64], 2, branch + "_res2")
        # net = self._res_block(net, [64, 128, 128], 2, branch + "_res3")

        net = self._pooling(net, -1, "max", branch + "_global_maxpool")
        net = self._flatten(net, "flatten")

        return net

    def multi_cnn(self, x, is_training):
        self._check_input(x)
        self.is_training = is_training

        dims = x.get_shape().as_list()[:-1] + [1]
        input0 = tf.reshape(x[..., 0], dims)
        input1 = tf.reshape(x[..., 1], dims)
        input2 = tf.reshape(x[..., 2], dims)
        input3 = tf.reshape(x[..., 3], dims)

        net0 = self._cnn_branch(input0, "branch0")
        net1 = self._cnn_branch(input1, "branch1")
        net2 = self._cnn_branch(input2, "branch2")
        net3 = self._cnn_branch(input3, "branch3")

        # nets = [net0]

        net = tf.concat([net0, net1, net2, net3], 1, "concate")
        # net = tf.concat([net0, net1, net2, net3], 4, "concate")
        # net = self._res_block(net0, [256, 512, 512], 1, "res")
        # net = self._pooling(net, -1, "max", "global_maxpool")
        # net = self._flatten(net, "flatten")

        # net = self._dropout(net, "dropout1")
        net = self._fc_bn_act(net, 1024, "fc")
        net = self._dropout(net, "dropout2")
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

        self._check_input(x)
        self.is_training = is_training

        # Here is a very simple case to test btc_train first
        net = self._conv_bn_act(x, 32, 3, 1, "layer1")
        net = self._pooling(net, 2, "max", "max_pool1")
        net = self._dropout(net, "dropout1")
        net = self._conv_bn_act(net, 64, 3, 1, "layer2")
        net = self._pooling(net, 2, "max", "max_pool2")
        net = self._dropout(net, "dropout2")
        net = self._conv_bn_act(net, 128, 3, 1, "layer3")
        net = self._pooling(net, 2, "max", "max_pool3")
        net = self._dropout(net, "dropout3")
        net = self._conv_bn_act(net, 256, 3, 1, "layer4")
        net = self._dropout(net, "dropout4")
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

        self._check_input(x)
        self.is_training = is_training

        # Here is a very simple case to test btc_train first
        net = self._conv_bn_act(x, 1, 5, 1, "preconv")
        net = self._res_block(net, [1, 1, 1], 2, "res1")
        net = self._res_block(net, [1, 1, 1], 2, "res2")
        net = self._res_block(net, [1, 1, 1], 2, "res3")
        net = self._res_block(net, [1, 1, 1], 2, "res4")
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

        self._check_input(x)
        self.is_training = is_training

        # Set the bottleneck symbol
        self.bc = True

        # Here is a very simple case to test btc_train first
        # Preconv layer before dense block
        with tf.variable_scope("preconv"):
            net = self._conv(x, 1, 5, 2)
        net = self._dense_block(net, 1, 2, "dense1")
        net = self._transition(net, "trans1")
        net = self._dense_block(net, 1, 2, "dense2")
        net = self._last_transition(net, "global_avgpool")
        net = self._flatten(net, "flatten")
        net = self._logits_fc(net, "logits")

        return net

    def autoencoder(self, x, is_training, sparse_type=None, k=None):
        '''AUTOENCODER

            Autoencoder with stride pooling.

            Inputs:
            -------
            - x: tensor placeholder, input volumes in batch
            - is_training: boolean placeholder, indicates the mode,
                           True: training mode,
                           False: validating and inferencing mode
            - sparse_type: string, "kl" or "wta"
            - k: int, parameters for Winner-Take-All constraint

            Output:
            -------
            - a reconstructed tensor of input

        '''

        if self.encoder is None:
            raise ValueError("Pool method is None.")

        self._check_input(x)
        self.is_training = is_training

        # Encoder section
        code = self.encoder(x)

        # Winner-Take-All constraint
        if sparse_type == "wta":
            code = self._wta_constraint(code, k)

        # Decoder section
        decode = self._decoder(code)

        self._check_output(x, decode)

        return code, decode

    def autoencoder_classier(self, x, is_training):
        '''CAE_CLASSIER_STRIDE

            Apply pre-trained model to generate code
            for each case. Train logistic regression
            to classify input case.

            Inputs:
            -------
            - x: tensor placeholder, input volumes in batch
            - is_training: boolean placeholder, indicates the mode,
                           True: training mode,
                           False: validating and inferencing mode

            Output:
            -------
            - output logits after classifier

        '''

        self._check_input(x)
        self.is_training = is_training

        # Encoder section
        code = self.encoder(x)
        # Global max pooling
        code = self._pooling(code, -1, "avg", "global_maxpool")
        code = self._flatten(code, "flatten")
        code = self._dropout(code, "dropout")
        output = self._logits_fc(code, "logits")

        return output


if __name__ == "__main__":

    models = BTCModels(classes=3, act="relu", alpha=None,
                       momentum=0.99, drop_rate=0.5, dims="3d",
                       cae_pool="stride", lifetime_rate=0.2)

    # Test function for cnn, full_cnn, res_cnn, dense_cnn and autoencoder
    x_3d = tf.placeholder(tf.float32, [32, 49, 49, 49, 4])
    # x_3d = tf.placeholder(tf.float32, [32, 112, 112, 88, 4])
    x_2d = tf.placeholder(tf.float32, [32, 112, 112, 4])
    is_training = tf.placeholder(tf.bool, [])

    # models.test(x_3d, is_training)
    # net = models.cnn(x_3d, is_training)
    # net = models.full_cnn(x_3d, is_training)
    # net = models.res_cnn(x_3d, is_training)
    # net = models.dense_cnn(x_3d, is_training)
    # net = models.autoencoder(x_3d, is_training)
    # net = models.autoencoder(x_3d, is_training, "wta", 10)
    # net = models.autoencoder_classier(x_3d, is_training)
    net = models.multi_cnn(x_3d, is_training)
