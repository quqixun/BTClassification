import numpy as np
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt


NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 30
VALID_BATCH_SIZE = 10
LEARNING_RATE = 0.001

CLASS_NUM = 2
CHANNEL_NUM = 4

COR_VOLUME_SHAPE = [155, 240, 4]
SAG_VOLUME_SHAPE = [155, 240, 4]
AX_VOLUME_SHAPE = [240, 240, 4]

TRAIN_LABEL_SHAPE = [TRAIN_BATCH_SIZE, CLASS_NUM]
VALID_LABEL_SHAPE = [VALID_BATCH_SIZE, CLASS_NUM]


def decode_record(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int64),
            "cor": tf.FixedLenFeature([], tf.string),
            "sag": tf.FixedLenFeature([], tf.string),
            "ax": tf.FixedLenFeature([], tf.string),
        })

    cor = tf.decode_raw(features["cor"], tf.float32)
    sag = tf.decode_raw(features["sag"], tf.float32)
    ax = tf.decode_raw(features["ax"], tf.float32)

    cor = tf.reshape(cor, COR_VOLUME_SHAPE)
    sag = tf.reshape(sag, SAG_VOLUME_SHAPE)
    ax = tf.reshape(ax, AX_VOLUME_SHAPE)

    label = tf.cast(features['label'], tf.int64)

    return cor, sag, ax, label


def inputs(path, batch_size, num_epochs,
           capacity=500, mad=300):
    if not num_epochs:
        num_epochs = None

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([path], num_epochs=num_epochs)
        cor, sag, ax, label = decode_record(filename_queue)

        cors, sags, axs, labels = tf.train.shuffle_batch([cor, sag, ax, label],
                                                         batch_size=batch_size,
                                                         num_threads=4,
                                                         capacity=capacity,
                                                         min_after_dequeue=mad)

    return cors, sags, axs, labels


def conv_weight(shape):
    sd = 1 / np.sqrt(np.prod(shape[0:3]) * CLASS_NUM)
    return tf.truncated_normal_initializer(stddev=sd)


def dense_weight(n_units):
    sd = 1 / np.sqrt(n_units * CLASS_NUM)
    return tf.truncated_normal_initializer(stddev=sd)


def conv2d(x, shape, act=tf.nn.relu, name=None):
    return tl.layers.Conv2dLayer(x,
                                 act=act,
                                 shape=shape,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 W_init=conv_weight(shape),
                                 b_init=None,
                                 name=name)


def dense(x, n_units, act=tf.nn.relu, name=None):
    return tl.layers.DenseLayer(x,
                                n_units=n_units,
                                act=act,
                                W_init=dense_weight(n_units),
                                b_init=None,
                                name=name)


def batch_norm(x, is_train=True, name=None):
    return tl.layers.BatchNormLayer(x,
                                    decay=0.9,
                                    epsilon=1e-5,
                                    act=tf.identity,
                                    is_train=is_train,
                                    name=name)


def max_pool(x, name=None):
    return tl.layers.PoolLayer(x,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               pool=tf.nn.max_pool,
                               name=name)


def drop_out(x, keep=0.5, is_train=False, name=None):
    return tf.layers.DropoutLayer(x,
                                  keep=keep,
                                  is_train=is_train,
                                  name=name)


def branch(x, is_train, name):
    x = tl.layers.InputLayer(x, name + "_input")
    b = conv2d(x, [7, 7, 4, 32], name=name + "_conv1")
    b = max_pool(b, name + "_mp1")
    # b = batch_norm(b, is_train, name + "_bn1")
    b = conv2d(b, [7, 7, 32, 32], name=name + "_conv2")
    b = max_pool(b, name + "_mp2")
    # b = batch_norm(b, is_train, name + "_bn2")
    b = conv2d(b, [5, 5, 32, 64], name=name + "_conv3")
    b = max_pool(b, name + "_mp3")
    # b = batch_norm(b, is_train, name + "_bn3")
    b = conv2d(b, [5, 5, 64, 64], name=name + "_conv4")
    b = max_pool(b, name + "_mp4")
    # b = batch_norm(b, is_train, name + "_bn4")
    b = conv2d(b, [3, 3, 64, 128], name=name + "_conv5")
    # b = max_pool(b, name + "_mp5")
    # b = batch_norm(b, is_train, name + "_bn5")
    # b = conv2d(b, [3, 3, 128, 128], name=name + "_conv6")
    psize = b.outputs.get_shape().as_list()[1:-1]
    b = tl.layers.MeanPool2d(b, psize, psize, name=name + "_gap")
    # b = tl.layers.MaxPool2d(b, psize, psize, name=name + "_gmp")
    b = tl.layers.FlattenLayer(b, name + "_flt")
    return b


def build_network(x_cor, x_sag, x_ax, is_train, reuse):
    with tf.variable_scope("model", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        cor_b = branch(x_cor, is_train, "cor")
        sag_b = branch(x_sag, is_train, "sag")
        ax_b = branch(x_ax, is_train, "ax")

        c = tl.layers.ConcatLayer([cor_b, sag_b, ax_b], name="concat")
        # c = batch_norm(c, is_train, "bn1")
        c = dense(c, 256, name="fc1")
        # c = dense(c, 256, name="fc2")
        # c = batch_norm(c, is_train, "bn2")
        c = dense(c, 2, act=tf.identity, name="output")

        return c


def get_loss(y_in, y_out):
    y_in = tf.one_hot(y_in, 2, axis=-1)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_in,
                                                logits=y_out))
    return loss


def get_accuracy(y_in, y_out):
    y_arg = tf.argmax(y_out, 1)
    correct_prediction = tf.equal(y_arg, y_in)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def train_model(train_set_path, valid_set_path, save_model_path):
    x_cor = tf.placeholder(tf.float32, [None] + COR_VOLUME_SHAPE)
    x_sag = tf.placeholder(tf.float32, [None] + SAG_VOLUME_SHAPE)
    x_ax = tf.placeholder(tf.float32, [None] + AX_VOLUME_SHAPE)
    y = tf.placeholder(tf.int64, [None, 1])

    tnet = build_network(x_cor, x_sag, x_ax, True, False)
    vnet = build_network(x_cor, x_sag, x_ax, False, True)

    tout = tnet.outputs
    vout = vnet.outputs

    train_loss = get_loss(y, tout)
    valid_loss = get_loss(y, vout)

    train_accuracy = get_accuracy(y, tout)
    valid_accuracy = get_accuracy(y, vout)

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)

    tcors, tsags, taxs, tlabels = inputs(path=train_set_path,
                                         batch_size=TRAIN_BATCH_SIZE,
                                         num_epochs=NUM_EPOCHS)

    vcors, vsags, vaxs, vlabels = inputs(path=valid_set_path,
                                         batch_size=VALID_BATCH_SIZE,
                                         num_epochs=NUM_EPOCHS)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    sess = tf.InteractiveSession()
    sess.run(init)

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        train_steps, valid_steps, epoch_nos = 0, 0, 1
        while not coord.should_stop():
            train_steps += 1
            tcor, tsag, tax, tlabel = sess.run([tcors, tsags, taxs, tlabels])

            # tcor0 = tcor[0, :, :, 2]
            # tsag0 = tsag[0, :, :, 2]
            # tax0 = tax[0, :, :, 2]
            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(tcor0, cmap="gray")
            # plt.subplot(1, 3, 2)
            # plt.imshow(tsag0, cmap="gray")
            # plt.subplot(1, 3, 3)
            # plt.imshow(tax0, cmap="gray")
            # plt.show()

            tlabel = np.reshape(tlabel, [-1, 1])
            fd_train = {x_cor: tcor, x_sag: tsag, x_ax: tax, y: tlabel}

            to, tacc, tloss, _ = sess.run([tout, train_accuracy, train_loss, train_step], feed_dict=fd_train)
            # print(to.shape)
            # print(to)
            print("Epoch {0} Train Step {1}:".format(epoch_nos, train_steps),
                  "accuracy {0:.6f}".format(tacc),
                  "loss {0:.6f}".format(tloss))

            if train_steps % 9 == 0:
                for _ in range(9):
                    valid_steps += 1
                    vcor, vsag, vax, vlabel = sess.run([vcors, vsags, vaxs, vlabels])
                    vlabel = np.reshape(vlabel, [-1, 1])
                    fd_valid = {x_cor: vcor, x_sag: vsag, x_ax: vax, y: vlabel}

                    vo, vacc, vloss = sess.run([vout, valid_accuracy, valid_loss], feed_dict=fd_valid)
                    # print(vo.shape)
                    # print(vo)
                    print("Epoch {0} Valid Step {1}:".format(epoch_nos, valid_steps),
                          "accuracy {0:.6f}".format(vacc),
                          "loss {0:.6f}".format(vloss))
                train_steps, valid_steps = 0, 0
                epoch_nos += 1
                print()
            # if epoch_no > NUM_EPOCHS:
            #     break
    except tf.errors.OutOfRangeError:
        print("Training has stopped.")
    finally:
        coord.request_stop()

    tl.files.save_npz(tnet.all_params, save_model_path)
    coord.join(thread)
    sess.close()

    return


if __name__ == '__main__':
    train_set_path = "/home/user4/Desktop/btc/data/TFRecords/MultiViews/train.tfrecord"
    valid_set_path = "/home/user4/Desktop/btc/data/TFRecords/MultiViews/valid.tfrecord"
    save_model_path = "model.npz"
    train_model(train_set_path, valid_set_path, save_model_path)
