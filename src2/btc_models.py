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
        '''

        self.input_shape = input_shape
        self.pooling = pooling
        self.l2_coeff = l2_coeff
        self.drop_rate = drop_rate
        self.bn_momentum = bn_momentum
        self.initializer = initializer

        if model_name == "pyramid":
            self.model = self._pyramid()

        return

    def _conv3d(self, inputs, filter_num, filter_size,
                strides=(1, 1, 1), name=None):
        '''_CONV3D
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
        '''

        return Dense(units,
                     kernel_initializer=self.initializer,
                     kernel_regularizer=l2(self.l2_coeff),
                     activation=activation,
                     name=name)(inputs)

    def _extract_features(self, inputs, name=None):
        '''_EXTRACT_FEATURES
        '''

        if self.pooling == "max":
            pool = MaxPooling3D
        elif self.pooling == "avg":
            pool = AveragePooling3D
        fts_pool = pool((7, 6, 6), name=name + "_pre_pool")(inputs)

        fts_flt = Flatten(name=name + "_pre_flt")(fts_pool)
        fts_bn = BatchNormalization(momentum=self.bn_momentum, name=name + "_pre_bn")(fts_flt)
        fts_dp = Dropout(self.drop_rate, name=name + "_pre_dp")(fts_bn)

        fc1 = self._dense(fts_dp, 256, "relu", name)
        fc1 = BatchNormalization(momentum=self.bn_momentum, name=name + "_bn")(fc1)
        return fc1

    def _pyramid(self):
        '''_PYRAMID
        '''

        inputs = Input(shape=self.input_shape)
        # 112 * 96 * 96 * 1

        conv1 = self._conv3d(inputs, 32, 5, strides=(2, 2, 2), name="conv1")
        conv1_bn = BatchNormalization(momentum=self.bn_momentum, name="conv1_bn")(conv1)
        # 56 * 48 * 48 * 32

        conv2 = self._conv3d(conv1_bn, 64, 3, name="conv2")
        conv2_mp = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name="conv2_mp")(conv2)
        conv2_bn = BatchNormalization(momentum=self.bn_momentum, name="conv2_bn")(conv2_mp)
        # 28 * 24 * 24 * 64

        conv3 = self._conv3d(conv2_bn, 128, 3, name="conv3")
        conv3_mp = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name="conv3_mp")(conv3)
        conv3_bn = BatchNormalization(momentum=self.bn_momentum, name="conv3_bn")(conv3_mp)
        # 14 * 12 * 12 * 128

        conv4 = self._conv3d(conv3_bn, 256, 3, name="conv4")
        conv4_mp = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name="conv4_mp")(conv4)
        conv4_bn = BatchNormalization(momentum=self.bn_momentum, name="conv4_bn")(conv4_mp)
        # 7 * 6 * 6 * 256

        conv5 = self._conv3d(conv4_bn, 256, 3, name="conv5")
        conv5_up = UpSampling3D((2, 2, 2), name="conv5_up")(conv5)
        # 14 * 12 * 12 * 256

        sum1 = Add(name="sum1")([conv4, conv5_up])
        sum1_bn = BatchNormalization(momentum=self.bn_momentum, name="sum1_bn")(sum1)
        conv6 = self._conv3d(sum1_bn, 128, 3, name="conv6")
        conv6_up = UpSampling3D((2, 2, 2), name="conv6_up")(conv6)
        # 28 * 24 * 24 * 128

        sum2 = Add(name="sum2")([conv3, conv6_up])
        sum2_bn = BatchNormalization(momentum=self.bn_momentum, name="sum2_bn")(sum2)
        conv7 = self._conv3d(sum2_bn, 64, 3, name="conv7")
        conv7_up = UpSampling3D((2, 2, 2), name="conv7_up")(conv7)
        # 56 * 48 * 48 * 64

        sum3 = Add(name="sum3")([conv2, conv7_up])
        sum3_bn = BatchNormalization(momentum=self.bn_momentum, name="sum3_bn")(sum3)
        conv8 = self._conv3d(sum3_bn, 32, 3, name="conv8")
        # 56 * 48 * 48 * 32

        fts1 = self._extract_features(conv5, name="fc1_1")  # 256    -->   256
        fts2 = self._extract_features(conv6, name="fc1_2")  # 1024   -->   256
        fts3 = self._extract_features(conv7, name="fc1_3")  # 4096   -->   256
        fts4 = self._extract_features(conv8, name="fc1_4")  # 16384  -->   256

        fts = Concatenate(name="fts_all")([fts1, fts2, fts3, fts4])  # 1024
        fts_dp = Dropout(rate=self.drop_rate, name="fts_all_dp")(fts)
        fc2 = self._dense(fts_dp, 256, "relu", name="fc2")
        fc2_bn = BatchNormalization(momentum=self.bn_momentum, name="fc2_bn")(fc2)

        fc3 = self._dense(fc2_bn, 2, "softmax", name="fc3")  # 2
        model = Model(inputs=inputs, outputs=fc3)
        return model


if __name__ == "__main__":

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
