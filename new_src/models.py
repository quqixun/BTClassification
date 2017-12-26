import keras
from keras.layers import *
from keras.models import Model, Sequential
from keras.regularizers import l1, l2, l1_l2


def vggish(weights_path=None):

    l2_coeff = 1e-5
    bn_momentum = 0.9

    model = Sequential()
    model.add(ZeroPadding3D(1, input_shape=(112, 112, 112, 1)))
    model.add(Convolution3D(32, 3,
                            kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(l2_coeff),
                            activation='relu'))
    model.add(ZeroPadding3D(1))
    model.add(Convolution3D(32, 3,
                            kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(l2_coeff),
                            activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(BatchNormalization(momentum=bn_momentum))

    model.add(ZeroPadding3D(1))
    model.add(Convolution3D(64, 3,
                            kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(l2_coeff),
                            activation='relu'))
    model.add(ZeroPadding3D(1))
    model.add(Convolution3D(64, 3,
                            kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(l2_coeff),
                            activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(BatchNormalization(momentum=bn_momentum))

    model.add(ZeroPadding3D(1))
    model.add(Convolution3D(128, 3,
                            kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(l2_coeff),
                            activation='relu'))
    model.add(ZeroPadding3D(1))
    model.add(Convolution3D(128, 3,
                            kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(l2_coeff),
                            activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(BatchNormalization(momentum=bn_momentum))

    model.add(ZeroPadding3D(1))
    model.add(Convolution3D(256, 3,
                            kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(l2_coeff),
                            activation='relu'))
    model.add(ZeroPadding3D(1))
    model.add(Convolution3D(256, 3,
                            kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(l2_coeff),
                            activation='relu'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2)))
    model.add(BatchNormalization(momentum=bn_momentum))

    model.add(ZeroPadding3D(1))
    model.add(Convolution3D(256, 3,
                            kernel_initializer="glorot_normal",
                            kernel_regularizer=l2(l2_coeff),
                            activation='relu'))
    # model.add(AveragePooling3D((7, 7, 7), strides=(7, 7, 7)))
    model.add(MaxPooling3D((7, 7, 7), strides=(7, 7, 7)))
    model.add(BatchNormalization(momentum=bn_momentum))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256,
                    kernel_initializer="glorot_normal",
                    kernel_regularizer=l2(l2_coeff),
                    activation='relu'))
    model.add(BatchNormalization(momentum=bn_momentum))
    model.add(Dropout(0.5))
    model.add(Dense(256,
                    kernel_initializer="glorot_normal",
                    kernel_regularizer=l2(l2_coeff),
                    activation='relu'))
    model.add(BatchNormalization(momentum=bn_momentum))
    model.add(Dropout(0.5))
    model.add(Dense(2,
                    kernel_initializer="glorot_normal",
                    kernel_regularizer=l2(l2_coeff),
                    activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def pyramid():

    l2_coeff = 1e-4
    bn_momentum = 0.9
    initializer = "glorot_uniform"

    # 112 * 112 * 112
    inputs = Input(shape=[112, 112, 112, 1])
    zp = ZeroPadding3D(1)(inputs)
    preconv = Convolution3D(64, 3, strides=(2, 2, 2),
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(l2_coeff),
                            activation="relu")(zp)
    # preconv = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(preconv)
    preconv = BatchNormalization(momentum=bn_momentum)(preconv)

    # 56 * 56 * 56
    zp = ZeroPadding3D(1)(preconv)
    conv1 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    mp1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)
    mp1 = BatchNormalization(momentum=bn_momentum)(mp1)

    # 28 * 28 * 28
    zp = ZeroPadding3D(1)(mp1)
    conv2 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    mp2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
    mp2 = BatchNormalization(momentum=bn_momentum)(mp2)

    # 14 * 14 * 14
    zp = ZeroPadding3D(1)(mp2)
    conv3 = Convolution3D(256, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    mp3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
    mp3 = BatchNormalization(momentum=bn_momentum)(mp3)

    # 7 * 7 * 7
    zp = ZeroPadding3D(1)(mp3)
    conv4 = Convolution3D(256, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    up1 = UpSampling3D((2, 2, 2))(conv4)

    # 14 * 14 * 14
    sum1 = Add()([conv3, up1])
    sum1 = BatchNormalization(momentum=bn_momentum)(sum1)
    zp = ZeroPadding3D(1)(sum1)
    conv5 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    up2 = UpSampling3D((2, 2, 2))(conv5)

    # 28 * 28 * 28
    sum2 = Add()([conv2, up2])
    sum2 = BatchNormalization(momentum=bn_momentum)(sum2)
    zp = ZeroPadding3D(1)(sum2)
    conv6 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    up3 = UpSampling3D((2, 2, 2))(conv6)

    # 56 * 56 * 56
    sum3 = Add()([conv1, up3])
    sum3 = BatchNormalization(momentum=bn_momentum)(sum3)
    zp = ZeroPadding3D(1)(sum3)
    conv7 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)

    max_conv1 = Flatten()(MaxPooling3D((56, 56, 56))(conv1))
    max_conv2 = Flatten()(MaxPooling3D((28, 28, 28))(conv2))
    max_conv3 = Flatten()(MaxPooling3D((14, 14, 14))(conv3))
    max_conv4 = Flatten()(MaxPooling3D((7, 7, 7))(conv4))
    max_conv5 = Flatten()(MaxPooling3D((14, 14, 14))(conv5))
    max_conv6 = Flatten()(MaxPooling3D((28, 28, 28))(conv6))
    max_conv7 = Flatten()(MaxPooling3D((56, 56, 56))(conv7))
    max_feats = Concatenate()([max_conv1, max_conv2, max_conv3,
                               max_conv4, max_conv5, max_conv6, max_conv7])
    max_feats = BatchNormalization(momentum=bn_momentum)(max_feats)
    max_feats = Dropout(0.75)(max_feats)

    avg_conv1 = Flatten()(AveragePooling3D((56, 56, 56))(conv1))
    avg_conv2 = Flatten()(AveragePooling3D((28, 28, 28))(conv2))
    avg_conv3 = Flatten()(AveragePooling3D((14, 14, 14))(conv3))
    avg_conv4 = Flatten()(AveragePooling3D((7, 7, 7))(conv4))
    avg_conv5 = Flatten()(AveragePooling3D((14, 14, 14))(conv5))
    avg_conv6 = Flatten()(AveragePooling3D((28, 28, 28))(conv6))
    avg_conv7 = Flatten()(AveragePooling3D((56, 56, 56))(conv7))
    avg_feats = Concatenate()([avg_conv1, avg_conv2, avg_conv3,
                               avg_conv4, avg_conv5, avg_conv6, avg_conv7])
    avg_feats = BatchNormalization(momentum=bn_momentum)(avg_feats)
    avg_feats = Dropout(0.75)(avg_feats)

    max_dense = Dense(256,
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(l2_coeff),
                      activation="relu")(max_feats)

    avg_dense = Dense(256,
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(l2_coeff),
                      activation="relu")(avg_feats)

    merge_feats = Concatenate()([max_dense, avg_dense])
    merge_feats = BatchNormalization(momentum=bn_momentum)(merge_feats)
    merge_feats = Dropout(0.75)(merge_feats)

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(merge_feats)

    model = Model(inputs=inputs, outputs=outputs)
    return model
