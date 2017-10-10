import os
import numpy as np
import tensorflow as tf
from btc_settings import *


def decode_example(queue, patch_shape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int64),
            "volume": tf.FixedLenFeature([], tf.string)
        })

    volume = tf.decode_raw(features["volume"], tf.float32)
    volume = tf.reshape(volume, patch_shape)
    label = tf.cast(features["label"], tf.uint8)

    return volume, label


def decode_tfrecord(path, batch_size, num_epoches, patch_shape,
                    min_after_dequeue, capacity):
    if not num_epoches:
        num_epoches = None

    with tf.name_scope("input"):
        queue = tf.train.string_input_producer([path], num_epochs=num_epoches)
        volume, label = decode_example(queue, patch_shape)

        volumes, labels = tf.train.shuffle_batch([volume, label],
                                                 batch_size=batch_size,
                                                 num_threads=4,
                                                 capacity=capacity,
                                                 min_after_dequeue=min_after_dequeue)

    return volumes, labels


if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    tra_path = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER, "partial_train.tfrecord")
    val_path = os.path.join(parent_dir, DATA_FOLDER, TFRECORDS_FOLDER, "partial_validate.tfrecord")

    epoches_num = 3
    batch_size = 10

    tra_volumes, tra_labels = decode_tfrecord(path=tra_path, batch_size=batch_size,
                                              num_epoches=epoches_num, patch_shape=PATCH_SHAPE,
                                              min_after_dequeue=300, capacity=350)
    val_volumes, val_labels = decode_tfrecord(path=val_path, batch_size=batch_size,
                                              num_epoches=epoches_num, patch_shape=PATCH_SHAPE,
                                              min_after_dequeue=300, capacity=350)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    tra_num = 236
    val_num = 224

    index = np.arange(1, epoches_num + 1)
    tra_epoch_iters = np.floor(index * (tra_num / batch_size)).astype(np.int64)
    val_epoch_iters = np.floor(index * (val_num / batch_size)).astype(np.int64)

    tra_iters = 1
    val_iters = 1
    epoch_no = 0

    try:
        while not coord.should_stop():
            [tx, ty] = sess.run([tra_volumes, tra_labels])
            print("Train: ", tra_iters, ty)

            if tra_iters % tra_epoch_iters[epoch_no] == 0:

                while val_iters <= val_epoch_iters[epoch_no]:
                    [vx, vy] = sess.run([val_volumes, val_labels])
                    print("Validatei: ", val_iters, vy)

                    val_iters += 1

                epoch_no += 1

            tra_iters += 1

    except tf.errors.OutOfRangeError:
        print("Training has stopped.")

    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
