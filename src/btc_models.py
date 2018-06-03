# Brain Tumor Classification
# Construct 3D Multi-Scale CNN.
# Author: Qixun QU
# Copyleft: MIT Licience

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


from __future__ import print_function


from keras.layers import *
from keras.models import Model
from keras.regularizers import l2


class BTCModels(object):

    def __init__(self,
                 model_name="pyramid",
                 input_shape=[112, 96, 96, 1],
                 pooling="max",
                 l2_coeff=5e-5,
                 drop_rate=0.5,
                 bn_momentum=0.9,
                 initializer="glorot_uniform"):
        '''__INIT__

            Intialization to generate model.

            Inputs:
            -------

            - model_name: string, selecte model, in this project,
                          only one choice is "pyramid".
            - input_shape: list, dimentions of input data,
                           [112, 96, 96, 1] is required.
            - pooling: string, pooling mathods, "max" for max pooling,
                       "avg" for average pooling. Default is "max".
            - l2_coeff: float, coefficient of L2 penalty. Default is 5e-5.
            - drop_rate: float, dropout rate, default is 0.5.
            - bn_momentum: float, momentum of batch normalization,
                           default is 0.9.
            - initializer: string, method to initialize parameters,
                           default is "glorot_uniform".

        '''

        # Set parameters
        self.input_shape = input_shape
        self.pooling = pooling
        self.l2_coeff = l2_coeff
        self.drop_rate = drop_rate
        self.bn_momentum = bn_momentum
        self.initializer = initializer

        # Build pyramid model, which is referred as
        # 3D Multi-Scale CNN in this project
        if model_name == "pyramid":
            self.model = self._pyramid()

        return

    def _conv3d(self, inputs, filter_num, filter_size,
                strides=(1, 1, 1), name=None):
        '''_CONV3D

            Construct a convolutional layer.

            Inputs:
            -------

            - inputs: input tensor, it should be original input,
                      or the output from previous layer.
            - filter_num: int, the number of filters.
            - filter_size: int or int list, the dimensions of filters.
            - strides: int tuple with length 3, the stride step in
                       each dimension.
            - name: string, layer's name.

            Output:
            -------

            - output tensor from convolutional layer.

        '''

        return Convolution3D(filter_num, filter_size,
                             strides=strides,
                             kernel_initializer=self.initializer,
                             kernel_regularizer=l2(self.l2_coeff),
                             activation="relu",
                             padding="same",
                             name=name)(inputs)

    def _dense(self, inputs, units, activation="relu", name=None):
        '''_DENSE

            Construct a densely layer.

            Inputs:
            -------

            - inputs: input tensor, the output from previous layer.
            - units: int, number of neurons in this layer.
            - activation: string, activation function, default is "relu".
            - name: string, layer's name.

            Output:
            -------

            - output tensor from densely layer.

        '''

        return Dense(units,
                     kernel_initializer=self.initializer,
                     kernel_regularizer=l2(self.l2_coeff),
                     activation=activation,
                     name=name)(inputs)

    def _extract_features(self, inputs, name=None):
        '''_EXTRACT_FEATURES

            Extract features from input tensor by:
            - Pooling (max or avg) in size 7*6*6.
            - Flatten + Batch normalization + Dropout.
            - Dense + Batch normalization.

            Inputs:
            -------

            - inputs: input tensor, the output from each scale.
            - name: string, prefix of layer's name.

            Output:
            -------

            - fc1: tensor in size of 256, features extracted from input.

        '''

        # Pooling (max or avg) in size of 7*6*6
        if self.pooling == "max":
            pool = MaxPooling3D
        elif self.pooling == "avg":
            pool = AveragePooling3D
        fts_pool = pool((7, 6, 6), name=name + "_pre_pool")(inputs)

        # Flatten + Batch normalization + Dropout
        fts_flt = Flatten(name=name + "_pre_flt")(fts_pool)
        fts_bn = BatchNormalization(momentum=self.bn_momentum, name=name + "_pre_bn")(fts_flt)
        fts_dp = Dropout(self.drop_rate, name=name + "_pre_dp")(fts_bn)

        # Dense + Batch normalization
        fc1 = self._dense(fts_dp, 256, "relu", name)
        fc1 = BatchNormalization(momentum=self.bn_momentum, name=name + "_bn")(fc1)

        return fc1

    def _pyramid(self):
        '''_PYRAMID

            Build and return 3D Multi-Scale CNN.

            Output:
            -------

            - model: Keras Models instance, 3D Multi-Scale CNN.

        '''

        # Input layer
        inputs = Input(shape=self.input_shape)
        # 112 * 96 * 96 * 1

        # Conv1 + BN
        conv1 = self._conv3d(inputs, 32, 5, strides=(2, 2, 2), name="conv1")
        conv1_bn = BatchNormalization(momentum=self.bn_momentum, name="conv1_bn")(conv1)
        # 56 * 48 * 48 * 32

        # Conv2 + Max Pooling + BN
        conv2 = self._conv3d(conv1_bn, 64, 3, name="conv2")
        conv2_mp = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name="conv2_mp")(conv2)
        conv2_bn = BatchNormalization(momentum=self.bn_momentum, name="conv2_bn")(conv2_mp)
        # 28 * 24 * 24 * 64

        # Conv3 + Max Pooling + BN
        conv3 = self._conv3d(conv2_bn, 128, 3, name="conv3")
        conv3_mp = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name="conv3_mp")(conv3)
        conv3_bn = BatchNormalization(momentum=self.bn_momentum, name="conv3_bn")(conv3_mp)
        # 14 * 12 * 12 * 128

        # Conv4 + Max Pooling + BN
        conv4 = self._conv3d(conv3_bn, 256, 3, name="conv4")
        conv4_mp = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name="conv4_mp")(conv4)
        conv4_bn = BatchNormalization(momentum=self.bn_momentum, name="conv4_bn")(conv4_mp)
        # 7 * 6 * 6 * 256

        # Conv5 (Scale1)
        conv5 = self._conv3d(conv4_bn, 256, 3, name="conv5")
        # 7 * 6 * 6 * 256
        # Upsampling1
        conv5_up = UpSampling3D((2, 2, 2), name="conv5_up")(conv5)
        # 14 * 12 * 12 * 256

        # Conv4 ADD Upsampling1 + BN
        sum1 = Add(name="sum1")([conv4, conv5_up])
        sum1_bn = BatchNormalization(momentum=self.bn_momentum, name="sum1_bn")(sum1)

        # Conv6 (Scale2)
        conv6 = self._conv3d(sum1_bn, 128, 3, name="conv6")
        # 14 * 12 * 12 * 128
        # Upsampling2
        conv6_up = UpSampling3D((2, 2, 2), name="conv6_up")(conv6)
        # 28 * 24 * 24 * 128

        # Conv3 ADD Upsampling2 + BN
        sum2 = Add(name="sum2")([conv3, conv6_up])
        sum2_bn = BatchNormalization(momentum=self.bn_momentum, name="sum2_bn")(sum2)

        # Conv7 (Scale3)
        conv7 = self._conv3d(sum2_bn, 64, 3, name="conv7")
        # 28 * 24 * 24 * 64
        # Upsampling3
        conv7_up = UpSampling3D((2, 2, 2), name="conv7_up")(conv7)
        # 56 * 48 * 48 * 64

        # Conv2 ADD Upsampling3 + BN
        sum3 = Add(name="sum3")([conv2, conv7_up])
        sum3_bn = BatchNormalization(momentum=self.bn_momentum, name="sum3_bn")(sum3)

        # Conv8 (Scale4)
        conv8 = self._conv3d(sum3_bn, 32, 3, name="conv8")
        # 56 * 48 * 48 * 32

        # Extracte features from Scale1
        fts1 = self._extract_features(conv5, name="fc1_1")  # 256    -->   256
        # Extracte features from Scale2
        fts2 = self._extract_features(conv6, name="fc1_2")  # 1024   -->   256
        # Extracte features from Scale3
        fts3 = self._extract_features(conv7, name="fc1_3")  # 4096   -->   256
        # Extracte features from Scale4
        fts4 = self._extract_features(conv8, name="fc1_4")  # 16384  -->   256

        # Fuse features of 4 scales + Dropout + Dense (256) + BN
        fts = Concatenate(name="fts_all")([fts1, fts2, fts3, fts4])  # 1024
        fts_dp = Dropout(rate=self.drop_rate, name="fts_all_dp")(fts)
        fc2 = self._dense(fts_dp, 256, "relu", name="fc2")
        fc2_bn = BatchNormalization(momentum=self.bn_momentum, name="fc2_bn")(fc2)

        # Output layer
        fc3 = self._dense(fc2_bn, 2, "softmax", name="fc3")  # 2
        model = Model(inputs=inputs, outputs=fc3)
        return model


if __name__ == "__main__":

    # A test to print model's architecture.

    from keras.optimizers import Adam

    model = BTCModels(model_name="pyramid",
                      input_shape=[112, 96, 96, 1],
                      pooling="max",
                      l2_coeff=5e-5,
                      drop_rate=0.5,
                      bn_momentum=0.9,
                      initializer="glorot_uniform").model
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(lr=1e-3),
                  metrics=["accuracy"])
    model.summary()
