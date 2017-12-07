import os
import numpy as np
import tensorflow as tf
from btc_settings import *
from btc_train import BTCTrain
import matplotlib.pyplot as plt
from btc_cae_parameters import get_parameters


class BTCInferenceCAE(BTCTrain):

    def __init__(self, paras, input_path, model_path):
        super().__init__(paras)
        self.model_path = model_path
        self.input = np.load(input_path)
        self.network = self.models.autoencoder

        self._inference()

        return

    def _compare(self, xr):
        plt.figure(num="compare")
        for i in range(4):
            plt.subplot(2, 4, 2 * i + 1)
            plt.title("original " + str(i))
            plt.axis("off")
            plt.imshow(self.input[..., i], cmap="gray")
            plt.subplot(2, 4, 2 * i + 2)
            plt.title("recontruction " + str(i))
            plt.axis("off")
            plt.imshow(xr[..., i], cmap="gray")
        plt.show()

        return

    def _inference(self):
        x = tf.placeholder(tf.float32, [1] + self.patch_shape)
        is_training = tf.placeholder_with_default(False, [])
        _, r = self.network(x, is_training)

        loader = tf.train.Saver()
        sess = tf.InteractiveSession()
        loader.restore(sess, self.model_path)

        xr = sess.run([r], feed_dict={x: np.reshape(self.input, [1] + self.patch_shape)})
        xr = np.reshape(np.array(xr), self.patch_shape)

        self._compare(xr)

        return


if __name__ == "__main__":
    parameters = get_parameters("cae", "slice", "kl")
    input_path = "/home/user4/btc/data/Slices/TCGA-CS-4944/0_1.npy"
    model_path = "/home/user4/btc/models/cae_2D_pool_kl/last/model"

    BTCInferenceCAE(parameters, input_path, model_path)
