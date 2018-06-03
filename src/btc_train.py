# Brain Tumor Classification
# Train 3D Multi-Scale CNN.
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


import os
import json
import shutil
import argparse
from btc_models import BTCModels

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import (CSVLogger,
                             TensorBoard,
                             ModelCheckpoint,
                             LearningRateScheduler)


class BTCTrain(object):

    def __init__(self,
                 paras_name,
                 paras_json_path,
                 weights_save_dir,
                 logs_save_dir,
                 save_best_weights=True):
        '''__INIT__

            Initalization before training model.

            Inputs:
            -------

            - paras_name: string, name of hyperparameters set,
                          can be found in hyper_paras.json.
            - paras_json_path: string, path of file which provides
                               hyperparamters, "hyper_paras.json"
                               in this project.
            - weights_save_dir: string, directory path where saves
                                trained model.
            - logs_save_dir: string, directory path where saves
                             logs of training process.
            - save_best_weights: boolean, if save the model with best
                                 validation accuracy. Default is True.

        '''

        # Dataset: training, validation and test
        self.data = None

        # If save the model which provides best validation accuracy
        self.save_best_weights = save_best_weights

        # Load hyperparameters
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._load_paras()

        # Create folder for saving weights
        self.weights_dir = os.path.join(weights_save_dir, paras_name)
        self.create_dir(self.weights_dir)

        # Create folder for saving training logs
        self.logs_dir = os.path.join(logs_save_dir, paras_name)
        self.create_dir(self.logs_dir)

        # Initialize files' names for weights at last or best epoch
        self.last_weights_path = os.path.join(self.weights_dir, "last.h5")
        self.best_weights_path = os.path.join(self.weights_dir, "best.h5")

        # CSV file path for writing learning curves
        self.curves_path = os.path.join(self.logs_dir, "curves.csv")

        return

    def _load_paras(self):
        '''_LOAD_PARAS

            Load hyperparameters from hyper_paras.json.

        '''

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
        '''_LOAD_MODEL

            Create 3D Multi-Scale CNN.

        '''

        self.model = BTCModels(model_name=self.model_name,
                               input_shape=self.input_shape,
                               pooling=self.pooling,
                               l2_coeff=self.l2_coeff,
                               drop_rate=self.drop_rate,
                               bn_momentum=self.bn_momentum,
                               initializer=self.initializer).model
        return

    def _set_optimizer(self):
        '''_SET_OPTIMIZER

            Set optimizer according to the given parameter.
            Use "Adam" in this project.

        '''

        if self.optimizer == "adam":
            self.opt_fcn = Adam(lr=self.lr_start)
        return

    def _set_lr_scheduler(self, epoch):
        '''_SET_LR_SCHEDULER

            Learning rate scheduler for training process.
            LR: [init] * 40 + [init * 0.1] * 30 + [init * 0.01] * 30

            Input:
            ------

            - epoch: int, nth training epoch.

            Output:
            -------

            - Learning rate, a float, for nth training epoch.

        '''

        lrs = [self.lr_start] * 40 + \
              [self.lr_start * 0.1] * 30 + \
              [self.lr_start * 0.01] * 30
        print("Learning rate:", lrs[epoch])

        return lrs[epoch]

    def _set_callbacks(self):
        '''_SET_CALLBACKS

            Set callback functions while training model.
            -1- Save learning curves while training.
            -2- Set learning rate scheduler.
            -3- Add support for TensorBoard.
            -4- Save best model while training. (optional)

        '''

        # Save learning curves in csv file while training
        csv_logger = CSVLogger(self.curves_path,
                               append=True, separator=",")

        # Set learning rate scheduler
        lr_scheduler = LearningRateScheduler(self._set_lr_scheduler)

        # Add support for TensorBoard
        tb = TensorBoard(log_dir=self.logs_dir,
                         batch_size=self.batch_size)
        self.callbacks = [csv_logger, lr_scheduler, tb]

        if self.save_best_weights:
            # Save best model while training
            checkpoint = ModelCheckpoint(filepath=self.best_weights_path,
                                         monitor="val_loss",
                                         verbose=0,
                                         save_best_only=True)
            self.callbacks += [checkpoint]

        return

    def _print_score(self):
        '''_PRINT_SCORE

            Print out metrics (loss and accuracy) of
            training, validation and testing set.

        '''

        # Helper function to compute and print metrics
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
        '''RUN

            Train model using given data.

            Input:
            ------

            - data: an BTCDataset instance, including features and
                    labels of training, validation and testing set.

        '''

        print("\nTraining the model.\n")

        self.data = data

        # Configurations of model and optimizer
        self._load_model()
        self._set_optimizer()

        # Compile model and print its structure
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=self.opt_fcn,
                           metrics=["accuracy"])
        self.model.summary()

        self._set_callbacks()
        # Train model
        self.model.fit(self.data.train_x, self.data.train_y,
                       batch_size=self.batch_size,
                       epochs=self.epochs_num,
                       validation_data=(self.data.valid_x,
                                        self.data.valid_y),
                       shuffle=True,
                       callbacks=self.callbacks)

        # Save model in last epoch
        self.model.save(self.last_weights_path)
        # Print metrics
        self._print_score()

        # Destroy the current TF graph
        K.clear_session()

        return

    @staticmethod
    def load_paras(paras_json_path, paras_name):
        '''LOAD_PARAS

            Load heperparameters from json file.
            See hyper_paras.json.

            Inputs:
            -------

            - paras_name: string, name of hyperparameters set,
                          can be found in hyper_paras.json.
            - paras_json_path: string, path of file which provides
                               hyperparamters, "hyper_paras.json"
                               in this project.

            Output:
            -------

            - A dictionay pf hyperparameters.

        '''

        paras = json.load(open(paras_json_path))
        return paras[paras_name]

    @staticmethod
    def create_dir(dir_path, rm=True):
        '''CREATE_DIR

            Create directory.

            Inputs:
            -------

            - dir_path: string, path of new directory.
            - rm: boolean, remove existing directory or not.

        '''

        if os.path.isdir(dir_path):
            if rm:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
        else:
            os.makedirs(dir_path)
        return


def main(hyper_paras_name):
    '''MAIN

        Main process to train model.

        Inputs:
        -------

        - hyper_paras_name: string, the name of hyperparameters set,
                            which can be found in hyper_paras.json.

    '''

    from btc_dataset import BTCDataset

    # Basic settings in pre_paras.json, including
    # 1. directory paths for input and output
    # 2. necessary information for splitting dataset
    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    # Get root path of input data
    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, pre_paras["data_dir"])

    # Set directories of preprocessed images
    hgg_dir = os.path.join(data_dir, pre_paras["hgg_out"])
    lgg_dir = os.path.join(data_dir, pre_paras["lgg_out"])

    # Set directory to save weights
    weights_save_dir = os.path.join(parent_dir, pre_paras["weights_save_dir"])
    # Set directory to save training and validation logs
    logs_save_dir = os.path.join(parent_dir, pre_paras["logs_save_dir"])

    # Partition dataset
    data = BTCDataset(hgg_dir, lgg_dir,
                      volume_type=pre_paras["volume_type"],
                      pre_trainset_path=pre_paras["pre_trainset_path"],
                      pre_validset_path=pre_paras["pre_validset_path"],
                      pre_testset_path=pre_paras["pre_testset_path"],
                      data_format=pre_paras["data_format"])
    data.run(pre_split=pre_paras["pre_split"],
             save_split=pre_paras["save_split"],
             save_split_dir=pre_paras["save_split_dir"])

    # Train the model
    train = BTCTrain(paras_name=hyper_paras_name,
                     paras_json_path=pre_paras["paras_json_path"],
                     weights_save_dir=weights_save_dir,
                     logs_save_dir=logs_save_dir,
                     save_best_weights=pre_paras["save_best_weights"])
    train.run(data)


if __name__ == "__main__":

    # Command line
    # python btc_train.py --paras=paras-1

    parser = argparse.ArgumentParser()

    # Set json file path to extract hyperparameters
    help_str = "Select a set of hyper-parameters in hyper_paras.json."
    parser.add_argument("--paras", action="store", default="paras-1",
                        dest="hyper_paras_name", help=help_str)

    args = parser.parse_args()
    main(args.hyper_paras_name)
