from keras.layers import *
from keras.models import Model
from keras.initializers import *
from keras.regularizers import l2


INPUT_SHAPE = [112, 96, 96, 1]
SHAPE1 = [i / 2 for i in INPUT_SHAPE[:3]]
SHAPE2 = [i / 4 for i in INPUT_SHAPE[:3]]
SHAPE3 = [i / 8 for i in INPUT_SHAPE[:3]]
SHAPE4 = [i / 16 for i in INPUT_SHAPE[:3]]


def vggish(l2_coeff=5e-5,
           bn_momentum=0.9,
           initializer="glorot_uniform",
           drop_rate=0.5,
           pool="all",
           scale=None):

    inputs = Input(shape=INPUT_SHAPE)
    # 112 * 96 * 96 * 1

    if initializer == "glorot_uniform":
        init = glorot_uniform(seed=727000)

    zp = ZeroPadding3D(2)(inputs)
    conv1 = Convolution3D(16, 5, strides=(2, 2, 2),
                          kernel_initializer=init,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # conv1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)
    conv1 = BatchNormalization(momentum=bn_momentum)(conv1)
    # 56 * 48 * 48 * 32

    zp = ZeroPadding3D(1)(conv1)
    conv2 = Convolution3D(16, 3,
                          kernel_initializer=init,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    conv2 = BatchNormalization(momentum=bn_momentum)(conv2)
    # 56 * 48 * 48 * 32

    zp = ZeroPadding3D(1)(conv2)
    conv3 = Convolution3D(16, 3,
                          kernel_initializer=init,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    conv3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
    conv3 = BatchNormalization(momentum=bn_momentum)(conv3)
    # 28 * 24 * 24 * 64

    zp = ZeroPadding3D(1)(conv3)
    conv4 = Convolution3D(16, 3,
                          kernel_initializer=init,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    conv4 = BatchNormalization(momentum=bn_momentum)(conv4)
    # 28 * 24 * 24 * 64

    zp = ZeroPadding3D(1)(conv4)
    conv5 = Convolution3D(16, 3,
                          kernel_initializer=init,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    conv5 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv5)
    conv5 = BatchNormalization(momentum=bn_momentum)(conv5)
    # 14 * 12 * 12 * 128

    zp = ZeroPadding3D(1)(conv5)
    conv6 = Convolution3D(16, 3,
                          kernel_initializer=init,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    conv6 = BatchNormalization(momentum=bn_momentum)(conv6)
    # 14 * 12 * 12 * 128

    zp = ZeroPadding3D(1)(conv6)
    conv7 = Convolution3D(16, 3,
                          kernel_initializer=init,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    conv7 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv7)
    conv7 = BatchNormalization(momentum=bn_momentum)(conv7)
    # 7 * 6 * 6 * 256

    zp = ZeroPadding3D(1)(conv7)
    conv8 = Convolution3D(16, 3,
                          kernel_initializer=init,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    conv8 = BatchNormalization(momentum=bn_momentum)(conv8)
    # 7 * 6 * 6 * 256

    fc1 = Flatten()(conv8)
    fc1 = BatchNormalization(momentum=bn_momentum)(fc1)
    dp1 = Dropout(rate=drop_rate, seed=727000)(fc1)

    fc2 = Dense(16,
                kernel_initializer=initializer,
                kernel_regularizer=l2(l2_coeff),
                activation="relu")(dp1)
    fc2 = BatchNormalization(momentum=bn_momentum)(fc2)
    dp2 = Dropout(rate=0.5, seed=727000)(fc2)

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(dp2)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def vggish2(l2_coeff=5e-5,
            bn_momentum=0.9,
            initializer="glorot_uniform",
            drop_rate=0.5,
            pool="all",
            scale=None):

    alpha = 0.0

    inputs = Input(shape=INPUT_SHAPE)
    # 112 * 96 * 96 * 1

    zp = ZeroPadding3D(3)(inputs)
    conv1 = Convolution3D(32, 7, strides=(2, 2, 2),
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation=None)(zp)
    conv1 = LeakyReLU(alpha=alpha)(conv1)
    # conv1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)
    conv1 = BatchNormalization(momentum=bn_momentum)(conv1)
    # 56 * 48 * 48 * 32

    zp = ZeroPadding3D(1)(conv1)
    conv2 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation=None)(zp)
    conv1 = LeakyReLU(alpha=alpha)(conv2)
    conv2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
    conv2 = BatchNormalization(momentum=bn_momentum)(conv2)
    # 28 * 24 * 24 * 64

    zp = ZeroPadding3D(1)(conv2)
    conv3 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation=None)(zp)
    conv1 = LeakyReLU(alpha=alpha)(conv3)
    conv3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
    conv3 = BatchNormalization(momentum=bn_momentum)(conv3)
    # 14 * 12 * 12 * 128

    zp = ZeroPadding3D(1)(conv3)
    conv4 = Convolution3D(256, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation=None)(zp)
    conv1 = LeakyReLU(alpha=alpha)(conv4)
    conv4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv4)
    # 7 * 6 * 6 * 256

    fc1 = Flatten()(conv4)
    fc1 = BatchNormalization(momentum=bn_momentum)(fc1)
    dp1 = Dropout(rate=drop_rate)(fc1)

    fc2 = Dense(512,
                kernel_initializer=initializer,
                kernel_regularizer=l2(l2_coeff),
                activation=None)(dp1)
    fc2 = LeakyReLU(alpha=alpha)(fc2)
    fc2 = BatchNormalization(momentum=bn_momentum)(fc2)

    fc3 = Dense(256,
                kernel_initializer=initializer,
                kernel_regularizer=l2(l2_coeff),
                activation=None)(fc2)
    fc3 = LeakyReLU(alpha=alpha)(fc3)
    fc3 = BatchNormalization(momentum=bn_momentum)(fc3)

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(fc3)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def pyramid(l2_coeff=5e-5,
            bn_momentum=0.9,
            initializer="glorot_uniform",
            drop_rate=0.5,
            pool="all"):

    fnum = 16
    # fnum = 32

    inputs = Input(shape=INPUT_SHAPE)
    # 112 * 96 * 96 * 1

    zp = ZeroPadding3D(2)(inputs)
    preconv = Convolution3D(fnum, 5, strides=(2, 2, 2),
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(l2_coeff),
                            activation="relu")(zp)
    # preconv = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(preconv)
    preconv = BatchNormalization(momentum=bn_momentum)(preconv)
    # 56 * 56 * 48 * 32

    zp = ZeroPadding3D(1)(preconv)
    conv1 = Convolution3D(fnum * 2, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 56 * 56 * 48 * 64
    mp1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)
    mp1 = BatchNormalization(momentum=bn_momentum)(mp1)
    # 28 * 28 * 24 * 64

    zp = ZeroPadding3D(1)(mp1)
    conv2 = Convolution3D(fnum * 4, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 28 * 28 * 24 * 128
    mp2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
    mp2 = BatchNormalization(momentum=bn_momentum)(mp2)
    # 14 * 14 * 12 * 128

    zp = ZeroPadding3D(1)(mp2)
    conv3 = Convolution3D(fnum * 8, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 14 * 14 * 12 * 256
    mp3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
    mp3 = BatchNormalization(momentum=bn_momentum)(mp3)
    # 7 * 7 * 6 * 256

    zp = ZeroPadding3D(1)(mp3)
    conv4 = Convolution3D(fnum * 8, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 7 * 7 * 6 * 256
    up1 = UpSampling3D((2, 2, 2))(conv4)
    # 14 * 14 * 12 * 256

    sum1 = Add()([conv3, up1])
    sum1 = BatchNormalization(momentum=bn_momentum)(sum1)
    zp = ZeroPadding3D(1)(sum1)
    conv5 = Convolution3D(fnum * 4, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 14 * 14 * 12 * 128
    up2 = UpSampling3D((2, 2, 2))(conv5)
    # 28 * 28 * 24 * 128

    sum2 = Add()([conv2, up2])
    sum2 = BatchNormalization(momentum=bn_momentum)(sum2)
    zp = ZeroPadding3D(1)(sum2)
    conv6 = Convolution3D(fnum * 2, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 28 * 28 * 24 * 64
    up3 = UpSampling3D((2, 2, 2))(conv6)
    # 56 * 56 * 48 * 64

    sum3 = Add()([conv1, up3])
    sum3 = BatchNormalization(momentum=bn_momentum)(sum3)
    zp = ZeroPadding3D(1)(sum3)
    conv7 = Convolution3D(fnum, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 56 * 56 * 48 * 32

    dnum = 256

    max_conv1 = Flatten()(MaxPooling3D([16, 12, 12])(ZeroPadding3D((4, 0, 0))(conv1)))
    max_conv2 = Flatten()(MaxPooling3D([8, 6, 6])(ZeroPadding3D((2, 0, 0))(conv2)))
    max_conv3 = Flatten()(MaxPooling3D([4, 3, 3])(ZeroPadding3D((1, 0, 0))(conv3)))
    max_conv4 = Flatten()(MaxPooling3D([3, 2, 1])(ZeroPadding3D((1, 0, 0))(conv4)))
    max_conv5 = Flatten()(MaxPooling3D([4, 3, 3])(ZeroPadding3D((1, 0, 0))(conv5)))
    max_conv6 = Flatten()(MaxPooling3D([8, 6, 6])(ZeroPadding3D((2, 0, 0))(conv6)))
    max_conv7 = Flatten()(MaxPooling3D([16, 12, 12])(ZeroPadding3D((4, 0, 0))(conv7)))
    max_feats = Concatenate()([max_conv1, max_conv2, max_conv3,
                               max_conv4, max_conv5, max_conv6, max_conv7])
    max_feats = BatchNormalization(momentum=bn_momentum)(max_feats)

    max_feats = Dropout(drop_rate)(max_feats)
    max_dense = Dense(dnum,
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(l2_coeff),
                      activation="relu")(max_feats)

    avg_conv1 = Flatten()(AveragePooling3D([16, 12, 12])(ZeroPadding3D((4, 0, 0))(conv1)))
    avg_conv2 = Flatten()(AveragePooling3D([8, 6, 6])(ZeroPadding3D((2, 0, 0))(conv2)))
    avg_conv3 = Flatten()(AveragePooling3D([4, 3, 3])(ZeroPadding3D((1, 0, 0))(conv3)))
    avg_conv4 = Flatten()(AveragePooling3D([3, 2, 1])(ZeroPadding3D((1, 0, 0))(conv4)))
    avg_conv5 = Flatten()(AveragePooling3D([4, 3, 3])(ZeroPadding3D((1, 0, 0))(conv5)))
    avg_conv6 = Flatten()(AveragePooling3D([8, 6, 6])(ZeroPadding3D((2, 0, 0))(conv6)))
    avg_conv7 = Flatten()(AveragePooling3D([16, 12, 12])(ZeroPadding3D((4, 0, 0))(conv7)))
    avg_feats = Concatenate()([avg_conv1, avg_conv2, avg_conv3,
                               avg_conv4, avg_conv5, avg_conv6, avg_conv7])
    avg_feats = Dropout(drop_rate)(avg_feats)
    avg_feats = BatchNormalization(momentum=bn_momentum)(avg_feats)
    avg_dense = Dense(dnum,
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(l2_coeff),
                      activation="relu")(avg_feats)

    merge_feats = Concatenate()([max_dense, avg_dense])
    # merge_feats = avg_dense
    # merge_feats = max_dense
    merge_feats = BatchNormalization(momentum=bn_momentum)(merge_feats)
    merge_feats = Dropout(drop_rate)(merge_feats)

    # merge_dense = Dense(512,
    #                     kernel_initializer=initializer,
    #                     kernel_regularizer=l2(l2_coeff),
    #                     activation="relu")(merge_feats)
    # merge_dense = BatchNormalization(momentum=bn_momentum)(merge_dense)

    final_dense = Dense(256,
                        kernel_initializer=initializer,
                        kernel_regularizer=l2(l2_coeff),
                        activation="relu")(merge_feats)
    final_dense = BatchNormalization(momentum=bn_momentum)(final_dense)
    final_dense = Dropout(drop_rate)(final_dense)

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(final_dense)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def pyramid_bba(l2_coeff=5e-5,
                bn_momentum=0.9,
                initializer="glorot_uniform",
                drop_rate=0.5,
                pool="all",
                scale=None):

    # fnum = 16
    fnum = 32

    inputs = Input(shape=INPUT_SHAPE)
    # 112 * 96 * 96 * 1

    zp = ZeroPadding3D(2)(inputs)
    preconv = Convolution3D(fnum, 5, strides=(2, 2, 2),
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(l2_coeff))(zp)
    preconv = BatchNormalization(momentum=bn_momentum)(preconv)
    preconv = Activation("relu")(preconv)
    # 56 * 56 * 48 * 32

    zp = ZeroPadding3D(1)(preconv)
    conv1 = Convolution3D(fnum * 2, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff))(zp)
    conv1 = BatchNormalization(momentum=bn_momentum)(conv1)
    conv1 = Activation("relu")(conv1)
    # 56 * 56 * 48 * 64
    mp1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1_bn)
    # 28 * 28 * 24 * 64

    zp = ZeroPadding3D(1)(mp1)
    conv2 = Convolution3D(fnum * 4, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff))(zp)
    conv2 = BatchNormalization(momentum=bn_momentum)(conv2)
    conv2 = Activation("relu")(conv2)
    # 28 * 28 * 24 * 128
    mp2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2_bn)
    # 14 * 14 * 12 * 128

    zp = ZeroPadding3D(1)(mp2)
    conv3 = Convolution3D(fnum * 8, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff))(zp)
    conv3 = BatchNormalization(momentum=bn_momentum)(conv3)
    conv3 = Activation("relu")(conv3)
    # 14 * 14 * 12 * 256
    mp3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3_bn)
    # 7 * 7 * 6 * 256

    zp = ZeroPadding3D(1)(mp3)
    conv4 = Convolution3D(fnum * 8, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff))(zp)
    conv4 = BatchNormalization(momentum=bn_momentum)(conv4)
    conv4 = Activation("relu")(conv4)
    # 7 * 7 * 6 * 256
    up1 = UpSampling3D((2, 2, 2))(conv4_bn)
    # 14 * 14 * 12 * 256

    sum1 = Add()([conv3, up1])
    zp = ZeroPadding3D(1)(sum1)
    conv5 = Convolution3D(fnum * 4, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff))(zp)
    conv5 = BatchNormalization(momentum=bn_momentum)(conv5)
    conv5 = Activation("relu")(conv5)
    # 14 * 14 * 12 * 128
    up2 = UpSampling3D((2, 2, 2))(conv5_bn)
    # 28 * 28 * 24 * 128

    sum2 = Add()([conv2, up2])
    zp = ZeroPadding3D(1)(sum2)
    conv6 = Convolution3D(fnum * 2, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff))(zp)
    conv6 = BatchNormalization(momentum=bn_momentum)(conv6)
    conv6 = Activation("relu")(conv6)
    # 28 * 28 * 24 * 64
    up3 = UpSampling3D((2, 2, 2))(conv6_bn)
    # 56 * 56 * 48 * 64

    sum3 = Add()([conv1, up3])
    zp = ZeroPadding3D(1)(sum3)
    conv7 = Convolution3D(fnum, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff))(zp)
    conv7 = BatchNormalization(momentum=bn_momentum)(conv7)
    conv7 = Activation("relu")(conv7)
    # 56 * 56 * 48 * 32

    dnum = 256

    if pool != "avg":
        max_conv1 = Flatten()(MaxPooling3D([16, 12, 12])(ZeroPadding3D((4, 0, 0))(conv1)))
        max_conv2 = Flatten()(MaxPooling3D([8, 6, 6])(ZeroPadding3D((2, 0, 0))(conv2)))
        max_conv3 = Flatten()(MaxPooling3D([4, 3, 3])(ZeroPadding3D((1, 0, 0))(conv3)))
        max_conv4 = Flatten()(MaxPooling3D([3, 2, 1])(ZeroPadding3D((1, 0, 0))(conv4)))
        max_conv5 = Flatten()(MaxPooling3D([4, 3, 3])(ZeroPadding3D((1, 0, 0))(conv5)))
        max_conv6 = Flatten()(MaxPooling3D([8, 6, 6])(ZeroPadding3D((2, 0, 0))(conv6)))
        max_conv7 = Flatten()(MaxPooling3D([16, 12, 12])(ZeroPadding3D((4, 0, 0))(conv7)))
        max_feats = Concatenate()([max_conv1, max_conv2, max_conv3,
                                   max_conv4, max_conv5, max_conv6, max_conv7])

        max_feats = Dropout(drop_rate)(max_feats)
        max_dense = Dense(dnum,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff))(max_feats)
        max_dense = BatchNormalization(momentum=bn_momentum)(max_dense)
        max_dense = Activation("relu")(max_dense)

    if pool != "max":
        avg_conv1 = Flatten()(AveragePooling3D([16, 12, 12])(ZeroPadding3D((4, 0, 0))(conv1)))
        avg_conv2 = Flatten()(AveragePooling3D([8, 6, 6])(ZeroPadding3D((2, 0, 0))(conv2)))
        avg_conv3 = Flatten()(AveragePooling3D([4, 3, 3])(ZeroPadding3D((1, 0, 0))(conv3)))
        avg_conv4 = Flatten()(AveragePooling3D([3, 2, 1])(ZeroPadding3D((1, 0, 0))(conv4)))
        avg_conv5 = Flatten()(AveragePooling3D([4, 3, 3])(ZeroPadding3D((1, 0, 0))(conv5)))
        avg_conv6 = Flatten()(AveragePooling3D([8, 6, 6])(ZeroPadding3D((2, 0, 0))(conv6)))
        avg_conv7 = Flatten()(AveragePooling3D([16, 12, 12])(ZeroPadding3D((4, 0, 0))(conv7)))
        avg_feats = Concatenate()([avg_conv1, avg_conv2, avg_conv3,
                                   avg_conv4, avg_conv5, avg_conv6, avg_conv7])

        avg_feats = Dropout(drop_rate)(avg_feats)
        avg_dense = Dense(dnum,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff))(avg_feats)
        avg_dense = BatchNormalization(momentum=bn_momentum)(avg_dense)
        avg_dense = Activation("relu")(avg_dense)

    if pool == "all":
        merge_feats = Concatenate()([max_dense, avg_dense])
    elif pool == "avg":
        merge_feats = avg_dense
    elif pool == "max":
        merge_feats = max_dense
    merge_feats = Dropout(drop_rate)(merge_feats)

    if pool == "all":
        merge_feats = Dense(256,
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(l2_coeff))(merge_feats)
        merge_feats = BatchNormalization(momentum=bn_momentum)(merge_feats)
        merge_feats = Activation("relu")(merge_feats)
        merge_feats = Dropout(drop_rate)(merge_feats)

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(merge_feats)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def pyramid2(l2_coeff=5e-5,
             bn_momentum=0.9,
             initializer="glorot_uniform",
             drop_rate=0.5,
             pool="all",
             scale=None):

    inputs = Input(shape=INPUT_SHAPE)
    # 112 * 96 * 96 * 1

    zp = ZeroPadding3D(2)(inputs)
    preconv = Convolution3D(64, 5, strides=(2, 2, 2),
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(l2_coeff),
                            activation="relu")(zp)
    # preconv = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(preconv)
    preconv = BatchNormalization(momentum=bn_momentum)(preconv)
    # 56 * 56 * 48 * 32

    zp = ZeroPadding3D(1)(preconv)
    conv1 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 56 * 56 * 48 * 64
    mp1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)
    mp1 = BatchNormalization(momentum=bn_momentum)(mp1)
    # 28 * 28 * 24 * 64

    zp = ZeroPadding3D(1)(mp1)
    conv2 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 28 * 28 * 24 * 128
    mp2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
    mp2 = BatchNormalization(momentum=bn_momentum)(mp2)
    # 14 * 14 * 12 * 128

    zp = ZeroPadding3D(1)(mp2)
    conv3 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 14 * 14 * 12 * 256
    mp3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
    mp3 = BatchNormalization(momentum=bn_momentum)(mp3)
    # 7 * 7 * 6 * 256

    zp = ZeroPadding3D(1)(mp3)
    conv4 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 7 * 7 * 6 * 256
    up1 = UpSampling3D((2, 2, 2))(conv4)
    # 14 * 14 * 12 * 256

    sum1 = Add()([conv3, up1])
    sum1 = BatchNormalization(momentum=bn_momentum)(sum1)
    zp = ZeroPadding3D(1)(sum1)
    conv5 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 14 * 14 * 12 * 128
    up2 = UpSampling3D((2, 2, 2))(conv5)
    # 28 * 28 * 24 * 128

    sum2 = Add()([conv2, up2])
    sum2 = BatchNormalization(momentum=bn_momentum)(sum2)
    zp = ZeroPadding3D(1)(sum2)
    conv6 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 28 * 28 * 24 * 64
    up3 = UpSampling3D((2, 2, 2))(conv6)
    # 56 * 56 * 48 * 64

    sum3 = Add()([conv1, up3])
    sum3 = BatchNormalization(momentum=bn_momentum)(sum3)
    zp = ZeroPadding3D(1)(sum3)
    conv7 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 56 * 56 * 48 * 32

    dnum = 128

    max_conv1 = Flatten()(MaxPooling3D(SHAPE1)(conv1))
    max_conv2 = Flatten()(MaxPooling3D(SHAPE2)(conv2))
    max_conv3 = Flatten()(MaxPooling3D(SHAPE3)(conv3))
    max_conv4 = Flatten()(MaxPooling3D(SHAPE4)(conv4))
    max_conv5 = Flatten()(MaxPooling3D(SHAPE3)(conv5))
    max_conv6 = Flatten()(MaxPooling3D(SHAPE2)(conv6))
    max_conv7 = Flatten()(MaxPooling3D(SHAPE1)(conv7))
    max_feats = Concatenate()([max_conv1, max_conv2, max_conv3,
                               max_conv4, max_conv5, max_conv6, max_conv7])
    max_feats = BatchNormalization(momentum=bn_momentum)(max_feats)

    max_feats = Dropout(drop_rate)(max_feats)
    max_dense = Dense(dnum,
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(l2_coeff),
                      activation="relu")(max_feats)

    avg_conv1 = Flatten()(AveragePooling3D(SHAPE1)(conv1))
    avg_conv2 = Flatten()(AveragePooling3D(SHAPE2)(conv2))
    avg_conv3 = Flatten()(AveragePooling3D(SHAPE3)(conv3))
    avg_conv4 = Flatten()(AveragePooling3D(SHAPE4)(conv4))
    avg_conv5 = Flatten()(AveragePooling3D(SHAPE3)(conv5))
    avg_conv6 = Flatten()(AveragePooling3D(SHAPE2)(conv6))
    avg_conv7 = Flatten()(AveragePooling3D(SHAPE1)(conv7))
    avg_feats = Concatenate()([avg_conv1, avg_conv2, avg_conv3,
                               avg_conv4, avg_conv5, avg_conv6, avg_conv7])
    avg_feats = Dropout(drop_rate)(avg_feats)
    avg_feats = BatchNormalization(momentum=bn_momentum)(avg_feats)
    avg_dense = Dense(dnum,
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(l2_coeff),
                      activation="relu")(avg_feats)

    merge_feats = Concatenate()([max_dense, avg_dense])
    # merge_feats = avg_dense
    # merge_feats = max_dense
    merge_feats = BatchNormalization(momentum=bn_momentum)(merge_feats)
    merge_feats = Dropout(drop_rate)(merge_feats)

    # merge_dense = Dense(512,
    #                     kernel_initializer=initializer,
    #                     kernel_regularizer=l2(l2_coeff),
    #                     activation="relu")(merge_feats)
    # merge_dense = BatchNormalization(momentum=bn_momentum)(merge_dense)

    # final_dense = Dense(256,
    #                     kernel_initializer=initializer,
    #                     kernel_regularizer=l2(l2_coeff),
    #                     activation="relu")(merge_feats)
    # final_dense = BatchNormalization(momentum=bn_momentum)(final_dense)
    # final_dense = Dropout(drop_rate)(final_dense)

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(merge_feats)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def dense_clf(inputs, dnum,
              l2_coeff=5e-5,
              bn_momentum=0.9,
              initializer="glorot_uniform",
              drop_rate=0.5):

    bn1 = BatchNormalization(momentum=bn_momentum)(inputs)
    dp1 = Dropout(drop_rate)(bn1)
    dense1 = Dense(dnum,
                   kernel_initializer=initializer,
                   kernel_regularizer=l2(l2_coeff),
                   activation="relu")(dp1)
    dense1 = BatchNormalization(momentum=bn_momentum)(dense1)
    return dense1


def pyramid3(l2_coeff=5e-5,
             bn_momentum=0.9,
             initializer="glorot_uniform",
             drop_rate=0.5,
             pool="all",
             scale=None):

    inputs = Input(shape=INPUT_SHAPE)
    # 112 * 96 * 96 * 1

    zp = ZeroPadding3D(2)(inputs)
    preconv = Convolution3D(64, 5, strides=(2, 2, 2),
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(l2_coeff),
                            activation="relu")(zp)
    # preconv = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(preconv)
    preconv = BatchNormalization(momentum=bn_momentum)(preconv)
    # 56 * 48 * 48 * 64

    zp = ZeroPadding3D(1)(preconv)
    conv1 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 56 * 48 * 48 * 64
    mp1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)
    mp1 = BatchNormalization(momentum=bn_momentum)(mp1)
    # 28 * 24 * 24 * 64

    btn21 = Convolution3D(64, 1,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(mp1)
    btn21 = BatchNormalization(momentum=bn_momentum)(btn21)

    zp = ZeroPadding3D(1)(btn21)
    btn22 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    btn22 = BatchNormalization(momentum=bn_momentum)(btn22)

    conv2 = Convolution3D(128, 1,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(btn22)
    # 28 * 24 * 24 * 128

    mp2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
    mp2 = BatchNormalization(momentum=bn_momentum)(mp2)
    # 14 * 12 * 12 * 128

    btn31 = Convolution3D(64, 1,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(mp2)
    btn31 = BatchNormalization(momentum=bn_momentum)(btn31)

    zp = ZeroPadding3D(1)(btn31)
    btn32 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    btn32 = BatchNormalization(momentum=bn_momentum)(btn32)

    conv3 = Convolution3D(128, 1,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(btn32)
    # 14 * 12 * 12 * 128
    mp3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
    mp3 = BatchNormalization(momentum=bn_momentum)(mp3)
    # 7 * 6 * 6 * 128

    btn41 = Convolution3D(64, 1,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(mp3)
    btn41 = BatchNormalization(momentum=bn_momentum)(btn41)

    zp = ZeroPadding3D(1)(btn41)
    btn42 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    btn42 = BatchNormalization(momentum=bn_momentum)(btn42)

    conv4 = Convolution3D(128, 1,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(btn42)
    # 7 * 6 * 6 * 128
    up1 = UpSampling3D((2, 2, 2))(conv4)
    # 14 * 12 * 12 * 128

    sum1 = Add()([conv3, up1])
    sum1 = BatchNormalization(momentum=bn_momentum)(sum1)

    btn51 = Convolution3D(64, 1,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(sum1)
    btn51 = BatchNormalization(momentum=bn_momentum)(btn51)

    zp = ZeroPadding3D(1)(btn51)
    btn52 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    btn52 = BatchNormalization(momentum=bn_momentum)(btn52)

    conv5 = Convolution3D(128, 1,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(btn52)
    # 14 * 12 * 12 * 128
    up2 = UpSampling3D((2, 2, 2))(conv5)
    # 28 * 24 * 24 * 128

    sum2 = Add()([conv2, up2])
    sum2 = BatchNormalization(momentum=bn_momentum)(sum2)

    zp = ZeroPadding3D(1)(sum2)
    conv6 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 28 * 24 * 24 * 64
    up3 = UpSampling3D((2, 2, 2))(conv6)
    # 56 * 48 * 48 * 64

    sum3 = Add()([conv1, up3])
    sum3 = BatchNormalization(momentum=bn_momentum)(sum3)
    zp = ZeroPadding3D(1)(sum3)
    conv7 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(zp)
    # 56 * 48 * 48 * 32

    dnum = 256

    if pool != "avg":
        max_conv4 = Flatten()(conv4)
        max_conv5 = Flatten()(MaxPooling3D([2, 2, 2])(conv5))
        max_conv6 = Flatten()(MaxPooling3D([4, 4, 4])(conv6))
        max_conv7 = Flatten()(MaxPooling3D([8, 8, 8])(conv7))

        clf_max_conv4 = dense_clf(max_conv4, dnum,
                                  l2_coeff, bn_momentum,
                                  initializer, drop_rate)
        clf_max_conv5 = dense_clf(max_conv5, dnum,
                                  l2_coeff, bn_momentum,
                                  initializer, drop_rate)
        clf_max_conv6 = dense_clf(max_conv6, dnum,
                                  l2_coeff, bn_momentum,
                                  initializer, drop_rate)
        clf_max_conv7 = dense_clf(max_conv7, dnum,
                                  l2_coeff, bn_momentum,
                                  initializer, drop_rate)

    if pool != "max":
        avg_conv4 = Flatten()(conv4)
        avg_conv5 = Flatten()(AveragePooling3D([2, 2, 2])(conv5))
        avg_conv6 = Flatten()(AveragePooling3D([4, 4, 4])(conv6))
        avg_conv7 = Flatten()(AveragePooling3D([8, 8, 8])(conv7))

        clf_avg_conv4 = dense_clf(avg_conv4, dnum,
                                  l2_coeff, bn_momentum,
                                  initializer, drop_rate)
        clf_avg_conv5 = dense_clf(avg_conv5, dnum,
                                  l2_coeff, bn_momentum,
                                  initializer, drop_rate)
        clf_avg_conv6 = dense_clf(avg_conv6, dnum,
                                  l2_coeff, bn_momentum,
                                  initializer, drop_rate)
        clf_avg_conv7 = dense_clf(avg_conv7, dnum,
                                  l2_coeff, bn_momentum,
                                  initializer, drop_rate)

    if pool == "all":
        clfs = Concatenate()([clf_max_conv4, clf_max_conv5,
                              clf_max_conv6, clf_max_conv7,
                              clf_avg_conv4, clf_avg_conv5,
                              clf_avg_conv6, clf_avg_conv7])
    elif pool == "max":
        clfs = Concatenate()([clf_max_conv4, clf_max_conv5,
                              clf_max_conv6, clf_max_conv7])
    elif pool == "avg":
        clfs = Concatenate()([clf_avg_conv4, clf_avg_conv5,
                              clf_avg_conv6, clf_avg_conv7])

    clfs = Dropout(drop_rate)(clfs)
    final_dense = Dense(256,
                        kernel_initializer=initializer,
                        kernel_regularizer=l2(l2_coeff),
                        activation="softmax")(clfs)
    final_dense = BatchNormalization(momentum=bn_momentum)(final_dense)
    final_dense = Dropout(drop_rate)(final_dense)

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(final_dense)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def extract_features(inputs, mode="max",
                     l2_coeff=5e-5,
                     bn_momentum=0.9,
                     initializer="glorot_uniform",
                     drop_rate=0.5):
    if mode == "max":
        fts_conv = Flatten()(MaxPooling3D([7, 6, 6])(inputs))
    elif mode == "avg":
        fts_conv = Flatten()(AveragePooling3D([7, 6, 6])(inputs))
    fts_conv = BatchNormalization(momentum=bn_momentum)(fts_conv)
    fts_conv = Dropout(drop_rate)(fts_conv)
    fts_dense = Dense(256,
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(l2_coeff),
                      activation="relu")(fts_conv)
    fts_dense = BatchNormalization(momentum=bn_momentum)(fts_dense)
    return fts_dense


def pyramid4(l2_coeff=5e-5,
             bn_momentum=0.99,
             initializer="glorot_uniform",
             drop_rate=0.5,
             pool="max",
             scale=None):

    inputs = Input(shape=INPUT_SHAPE)
    # 112 * 96 * 96 * 1

    preconv = Convolution3D(32, 5, strides=(2, 2, 2),
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(l2_coeff),
                            activation="relu",
                            padding="same")(inputs)
    preconv = BatchNormalization(momentum=bn_momentum)(preconv)
    # 56 * 48 * 48 * 32

    conv1 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same")(preconv)
    # 56 * 48 * 48 * 64
    mp1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)
    mp1 = BatchNormalization(momentum=bn_momentum)(mp1)
    # 28 * 24 * 24 * 64

    conv2 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same")(mp1)
    # 28 * 24 * 24 * 128
    mp2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
    mp2 = BatchNormalization(momentum=bn_momentum)(mp2)
    # 14 * 12 * 12 * 128

    conv3 = Convolution3D(256, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same")(mp2)
    # 14 * 12 * 12 * 256
    mp3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
    mp3 = BatchNormalization(momentum=bn_momentum)(mp3)
    # 7 * 6 * 6 * 256

    conv4 = Convolution3D(256, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same",
                          name="conv4")(mp3)
    # 7 * 6 * 6 * 256

    up1 = UpSampling3D((2, 2, 2))(conv4)
    # 14 * 12 * 12 * 256
    sum1 = Add()([conv3, up1])
    sum1 = BatchNormalization(momentum=bn_momentum)(sum1)

    conv5 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same",
                          name="conv5")(sum1)
    # 14 * 12 * 12 * 128

    up2 = UpSampling3D((2, 2, 2))(conv5)
    # 28 * 24 * 24 * 128
    sum2 = Add()([conv2, up2])
    sum2 = BatchNormalization(momentum=bn_momentum)(sum2)

    conv6 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same",
                          name="conv6")(sum2)
    # 28 * 24 * 24 * 64
    up3 = UpSampling3D((2, 2, 2))(conv6)
    # 56 * 48 * 48 * 64

    sum3 = Add()([conv1, up3])
    sum3 = BatchNormalization(momentum=bn_momentum)(sum3)

    conv7 = Convolution3D(32, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same",
                          name="conv7")(sum3)
    # 56 * 48 * 48 * 32

    fts_scale0 = extract_features(conv4, pool, l2_coeff, bn_momentum, initializer, drop_rate)
    fts_scale1 = extract_features(conv5, pool, l2_coeff, bn_momentum, initializer, drop_rate)
    fts_scale2 = extract_features(conv6, pool, l2_coeff, bn_momentum, initializer, drop_rate)
    fts_scale3 = extract_features(conv7, pool, l2_coeff, bn_momentum, initializer, drop_rate)

    fts = Concatenate()([fts_scale0, fts_scale1, fts_scale2, fts_scale3])
    fts = Dropout(rate=drop_rate)(fts)
    fts_dense = Dense(256,
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(l2_coeff),
                      activation="relu")(fts)
    fts_dense = BatchNormalization(momentum=bn_momentum)(fts_dense)
    # fts_dense = Dropout(rate=drop_rate)(fts_dense)

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(fts_dense)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def pyramid5(l2_coeff=5e-5,
             bn_momentum=0.9,
             initializer="glorot_uniform",
             drop_rate=0.5,
             pool="max",
             scale=None):

    inputs = Input(shape=INPUT_SHAPE)
    # 112 * 96 * 96 * 1

    preconv = Convolution3D(64, 7, strides=(2, 2, 2),
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(l2_coeff),
                            activation="relu",
                            padding="same")(inputs)
    preconv = BatchNormalization(momentum=bn_momentum)(preconv)
    # 56 * 48 * 48 * 32

    conv1 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same")(preconv)
    # 56 * 48 * 48 * 64
    mp1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)
    mp1 = BatchNormalization(momentum=bn_momentum)(mp1)
    # 28 * 24 * 24 * 64

    conv2 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same")(mp1)
    # 28 * 24 * 24 * 128
    mp2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
    mp2 = BatchNormalization(momentum=bn_momentum)(mp2)
    # 14 * 12 * 12 * 128

    conv3 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same")(mp2)
    # 14 * 12 * 12 * 256
    mp3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
    mp3 = BatchNormalization(momentum=bn_momentum)(mp3)
    # 7 * 6 * 6 * 256

    conv4 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same",
                          name="conv4")(mp3)
    # 7 * 6 * 6 * 256

    up1 = UpSampling3D((2, 2, 2))(conv4)
    # 14 * 12 * 12 * 256
    sum1 = Add()([conv3, up1])
    sum1 = BatchNormalization(momentum=bn_momentum)(sum1)

    conv5 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same",
                          name="conv5")(sum1)
    # 14 * 12 * 12 * 128

    up2 = UpSampling3D((2, 2, 2))(conv5)
    # 28 * 24 * 24 * 128
    sum2 = Add()([conv2, up2])
    sum2 = BatchNormalization(momentum=bn_momentum)(sum2)

    conv6 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same",
                          name="conv6")(sum2)
    # 28 * 24 * 24 * 64
    up3 = UpSampling3D((2, 2, 2))(conv6)
    # 56 * 48 * 48 * 64

    sum3 = Add()([conv1, up3])
    sum3 = BatchNormalization(momentum=bn_momentum)(sum3)

    conv7 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same",
                          name="conv7")(sum3)
    # 56 * 48 * 48 * 32

    fts_scale0 = extract_features(conv4, pool, l2_coeff, bn_momentum, initializer, drop_rate)
    fts_scale1 = extract_features(conv5, pool, l2_coeff, bn_momentum, initializer, drop_rate)
    fts_scale2 = extract_features(conv6, pool, l2_coeff, bn_momentum, initializer, drop_rate)
    fts_scale3 = extract_features(conv7, pool, l2_coeff, bn_momentum, initializer, drop_rate)

    fts = Concatenate()([fts_scale0, fts_scale1, fts_scale2, fts_scale3])
    fts = Dropout(rate=drop_rate)(fts)
    fts_dense = Dense(256,
                      kernel_initializer=initializer,
                      kernel_regularizer=l2(l2_coeff),
                      activation="relu")(fts)
    fts_dense = BatchNormalization(momentum=bn_momentum)(fts_dense)
    # fts_dense = Dropout(rate=drop_rate)(fts_dense)

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(fts_dense)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def pyramid6(l2_coeff=5e-5,
             bn_momentum=0.9,
             initializer="glorot_uniform",
             drop_rate=0.5,
             pool="max",
             scale=5):

    inputs = Input(shape=INPUT_SHAPE)
    # 112 * 96 * 96 * 1

    preconv = Convolution3D(32, 5, strides=(2, 2, 2),
                            kernel_initializer=initializer,
                            kernel_regularizer=l2(l2_coeff),
                            activation="relu",
                            padding="same")(inputs)
    preconv = BatchNormalization(momentum=bn_momentum)(preconv)
    # 56 * 48 * 48 * 32

    conv1 = Convolution3D(64, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same")(preconv)
    # 56 * 48 * 48 * 64
    mp1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)
    mp1 = BatchNormalization(momentum=bn_momentum)(mp1)
    # 28 * 24 * 24 * 64

    conv2 = Convolution3D(128, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same")(mp1)
    # 28 * 24 * 24 * 128
    mp2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)
    mp2 = BatchNormalization(momentum=bn_momentum)(mp2)
    # 14 * 12 * 12 * 128

    conv3 = Convolution3D(256, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same")(mp2)
    # 14 * 12 * 12 * 256
    mp3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv3)
    mp3 = BatchNormalization(momentum=bn_momentum)(mp3)
    # 7 * 6 * 6 * 256

    conv4 = Convolution3D(256, 3,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu",
                          padding="same",
                          name="conv4")(mp3)
    # 7 * 6 * 6 * 256
    fts_scale1 = extract_features(conv4, pool, l2_coeff, bn_momentum,
                                  initializer, drop_rate)
    fts_used = fts_scale1

    if scale > 1:
        up1 = UpSampling3D((2, 2, 2))(conv4)
        # 14 * 12 * 12 * 256
        sum1 = Add()([conv3, up1])
        sum1 = BatchNormalization(momentum=bn_momentum)(sum1)

        conv5 = Convolution3D(128, 3,
                              kernel_initializer=initializer,
                              kernel_regularizer=l2(l2_coeff),
                              activation="relu",
                              padding="same",
                              name="conv5")(sum1)
        # 14 * 12 * 12 * 128
        fts_scale2 = extract_features(conv5, pool, l2_coeff, bn_momentum,
                                      initializer, drop_rate)
        fts_used = fts_scale2

    if scale > 2:
        up2 = UpSampling3D((2, 2, 2))(conv5)
        # 28 * 24 * 24 * 128
        sum2 = Add()([conv2, up2])
        sum2 = BatchNormalization(momentum=bn_momentum)(sum2)

        conv6 = Convolution3D(64, 3,
                              kernel_initializer=initializer,
                              kernel_regularizer=l2(l2_coeff),
                              activation="relu",
                              padding="same",
                              name="conv6")(sum2)
        fts_scale3 = extract_features(conv6, pool, l2_coeff, bn_momentum,
                                      initializer, drop_rate)
        fts_used = fts_scale3

    if scale > 3:
        # 28 * 24 * 24 * 64
        up3 = UpSampling3D((2, 2, 2))(conv6)
        # 56 * 48 * 48 * 64

        sum3 = Add()([conv1, up3])
        sum3 = BatchNormalization(momentum=bn_momentum)(sum3)

        conv7 = Convolution3D(32, 3,
                              kernel_initializer=initializer,
                              kernel_regularizer=l2(l2_coeff),
                              activation="relu",
                              padding="same",
                              name="conv7")(sum3)
        # 56 * 48 * 48 * 32
        fts_scale4 = extract_features(conv7, pool, l2_coeff, bn_momentum,
                                      initializer, drop_rate)
        fts_used = fts_scale4

    if scale > 4:
        fts = Concatenate()([fts_scale1, fts_scale2, fts_scale3, fts_scale4])
        fts = Dropout(rate=drop_rate)(fts)
        fts_dense = Dense(256,
                          kernel_initializer=initializer,
                          kernel_regularizer=l2(l2_coeff),
                          activation="relu")(fts)
        fts_dense = BatchNormalization(momentum=bn_momentum)(fts_dense)
    else:
        fts_dense = fts_used

    outputs = Dense(2,
                    kernel_initializer=initializer,
                    kernel_regularizer=l2(l2_coeff),
                    activation="softmax")(fts_dense)

    model = Model(inputs=inputs, outputs=outputs)
    return model
