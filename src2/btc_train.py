from __future__ import print_function


import os
import json
import shutil
from btc_models import BTCModels

from keras.optimizers import Adam
from keras.callbacks import (CSVLogger,
                             TensorBoard,
                             ModelCheckpoint,
                             LearningRateScheduler)


class BTCTrain(object):

    def __init__(self,
                 paras_name, paras_json_path,
                 weights_save_dir, logs_save_dir,
                 save_best_weights=True):
        self.data = None
        self.save_best_weights = save_best_weights
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._resolve_paras()

        self.weights_dir = os.path.join(weights_save_dir, paras_name)
        self.logs_dir = os.path.join(logs_save_dir, paras_name)

        self.create_dir(self.weights_dir)
        self.create_dir(self.logs_dir)

        self.last_weights_path = os.path.join(self.weights_dir, "last.h5")
        self.best_weights_path = os.path.join(self.weights_dir, "best.h5")
        self.curves_path = os.path.join(self.logs_dir, "curves.csv")

        return

    def _resolve_paras(self):
        # Parameters to construct model
        self.model_name = self.paras["model_name"]
        self.input_shape = self.paras["input_shape"]
        self.pooling = self.paras["pooling"]
        self.l2_coeff = self.paras["l2_coeff"]
        self.drop_rate = self.paras["drop_rate"]
        self.bn_momentum = self.paras["bn_momentum"]
        self.initializer = self.paras["initializer"]

        # Parameters to train model
        self.optimizer = self.paras["optimizer"]
        self.lr_start = self.paras["lr_start"]
        self.epochs_num = self.paras["epochs_num"]
        self.batch_size = self.paras["batch_size"]
        return

    def _load_model(self):
        self.model = BTCModels(model_name=self.model_name,
                               input_shape=self.input_shape,
                               pooling=self.pooling,
                               l2_coeff=self.l2_coeff,
                               drop_rate=self.drop_rate,
                               bn_momentum=self.bn_momentum,
                               initializer=self.initializer).model
        return

    def _set_optimizer(self):
        if self.optimizer == "adam":
            self.opt_fcn = Adam(lr=self.lr_start)
        return

    def _set_lr_scheduler(self, epoch):
        lrs = [self.lr_start] * 40 + \
              [self.lr_start * 0.1] * 30 + \
              [self.lr_start * 0.01] * 30
        return lrs[epoch]

    def _set_callbacks(self):
        csv_logger = CSVLogger(self.curves_path,
                               append=True,
                               separator=",")
        lr_scheduler = LearningRateScheduler(self._set_lr_scheduler)
        tb = TensorBoard(log_dir=self.logs_dir,
                         batch_size=self.batch_size)
        self.callbacks = [csv_logger, lr_scheduler, tb]

        if self.save_best_weights:
            checkpoint = ModelCheckpoint(filepath=self.best_weights_path,
                                         monitor="val_loss",
                                         verbose=0,
                                         save_best_only=True)
            self.callbacks += [checkpoint]

        return

    def _print_score(self):

        def evaluate(x, y, data_str):
            score = self.model.evaluate(x, y, self.batch_size, 0)
            print(data_str + " Set: Loss: {0:.4f}, Accuracy: {1:.4f}".format(
                  score[0], score[1]))
            return

        evaluate(self.data.train_x, self.data.train_y, "Training")
        evaluate(self.data.valid_x, self.data.valid_y, "Validation")
        evaluate(self.data.test_x, self.data.test_y, "Testing")

        return

    def run(self, data):

        self.data = data

        self._load_model()
        self._set_optimizer()

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=self.opt_fcn,
                           metrics=["accuracy"])
        self.model.summary()

        self._set_callbacks()
        self.model.fit(self.data.train_x, self.data.train_y,
                       batch_size=self.batch_size,
                       epochs=self.epochs_num,
                       validation_data=(self.data.valid_x,
                                        self.data.valid_y),
                       shuffle=True,
                       callbacks=self.callbacks)

        self.model.save(self.last_weights_path)
        self._print_score()

        return

    @staticmethod
    def load_paras(paras_json_path, paras_name):
        paras = json.load(open(paras_json_path))
        return paras[paras_name]

    @staticmethod
    def create_dir(dir_path, rm=True):
        if os.path.isdir(dir_path):
            if rm:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
        else:
            os.makedirs(dir_path)
        return


if __name__ == "__main__":

    from btc_dataset import BTCDataset

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data", "BraTS")
    hgg_dir = os.path.join(data_dir, "HGGSegTrimmed")
    lgg_dir = os.path.join(data_dir, "LGGSegTrimmed")

    data = BTCDataset(hgg_dir, lgg_dir,
                      volume_type="t1ce",
                      pre_split=True,
                      pre_trainset_path="DataSplit/trainset.csv",
                      pre_validset_path="DataSplit/validset.csv",
                      pre_testset_path="DataSplit/testset.csv")

    paras_name = "paras-1"
    paras_json_path = "paras.json"
    weights_save_dir = os.path.join(parent_dir, "weights")
    logs_save_dir = os.path.join(parent_dir, "logs")

    train = BTCTrain(paras_name=paras_name,
                     paras_json_path=paras_json_path,
                     weights_save_dir=weights_save_dir,
                     logs_save_dir=logs_save_dir,
                     save_best_weights=True)
    train.run(data)
