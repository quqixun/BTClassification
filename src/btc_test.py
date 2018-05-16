from __future__ import print_function


import os
import json
import shutil
import numpy as np
import pandas as pd
from keras import backend as K
from btc_models import BTCModels
from sklearn.metrics import (log_loss,
                             roc_curve,
                             recall_score,
                             roc_auc_score,
                             precision_score,
                             confusion_matrix)


class BTCTest(object):

    def __init__(self,
                 paras_name,
                 paras_json_path,
                 weights_save_dir,
                 results_save_dir,
                 test_weights="last",
                 pred_trainset=False):
        '''_INIT__
        '''

        if not os.path.isdir(weights_save_dir):
            raise IOError("Model directory is not exist.")

        self.paras_name = paras_name
        self.results_save_dir = results_save_dir
        self.weights = test_weights
        self.pred_trainset = pred_trainset
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._resolve_paras()

        self.weights_path = os.path.join(weights_save_dir,
                                         paras_name, test_weights + ".h5")
        self.results_dir = os.path.join(results_save_dir, paras_name)
        self.create_dir(self.results_dir, rm=False)

        return

    def _resolve_paras(self):
        self.model_name = self.paras["model_name"]
        self.batch_size = self.paras["batch_size"]
        return

    def _load_model(self):
        self.model = BTCModels(model_name=self.model_name).model
        return

    def _pred_evaluate(self, x, y, dataset):

        def acc(true_y, pred_y):
            return (true_y == pred_y).all(axis=1).mean()

        def loss(true_y, pred_y):
            return log_loss(true_y, pred_y, normalize=True)

        def precision(true_y, pred_y, label):
            return precision_score(true_y, pred_y, pos_label=label)

        def recall(true_y, pred_y, label):
            return recall_score(true_y, pred_y, pos_label=label)

        print("Dataset to be predicted: " + dataset)
        pred = self.model.predict(x, self.batch_size, 0)

        arg_y = np.argmax(y, axis=1)
        arg_y = np.reshape(arg_y, (-1, 1))
        hgg = np.where(arg_y == 1)[0]
        lgg = np.where(arg_y == 0)[0]

        arg_pred = np.argmax(pred, axis=1)
        arg_pred = np.reshape(arg_pred, (-1, 1))

        roc_line = roc_curve(arg_y, pred[:, 1], pos_label=1)
        tn, fp, fn, tp = confusion_matrix(arg_y, arg_pred).ravel()

        results = {"name": self.paras_name,
                   "acc": acc(arg_y, arg_pred),
                   "hgg_acc": acc(arg_y[hgg], arg_pred[hgg]),
                   "lgg_acc": acc(arg_y[lgg], arg_pred[lgg]),
                   "loss": loss(y, pred),
                   "hgg_loss": loss(y[hgg], pred[hgg]),
                   "lgg_loss": loss(y[lgg], pred[lgg]),
                   "hgg_precision": precision(arg_y, arg_pred, 1),
                   "lgg_precision": precision(arg_y, arg_pred, 0),
                   "hgg_recall": recall(arg_y, arg_pred, 1),
                   "lgg_recall": recall(arg_y, arg_pred, 0),
                   "roc_auc": roc_auc_score(arg_y, pred[:, 1]),
                   "tn": tn, "fp": fp, "fn": fn, "tp": tp}

        res_df = pd.DataFrame(data=results, index=[0])
        res_df = res_df[["name", "acc", "hgg_acc", "lgg_acc",
                         "loss", "hgg_loss", "lgg_loss",
                         "hgg_precision", "hgg_recall",
                         "lgg_precision", "lgg_recall",
                         "roc_auc", "tn", "fp", "fn", "tp"]]

        root_name = [dataset, self.weights]
        res_csv_name = "_".join(root_name + ["res.csv"])
        roc_line_name = "_".join(root_name + ["roc_curve.npy"])

        res_csv_path = os.path.join(self.results_dir, res_csv_name)
        res_df.to_csv(res_csv_path, index=False)

        roc_line_path = os.path.join(self.results_dir, roc_line_name)
        np.save(roc_line_path, roc_line)

        return

    def run(self, data):
        '''RUN
        '''

        print("\nTesting the model.\n")

        self._load_model()
        self.model.load_weights(self.weights_path)

        if self.pred_trainset:
            self._pred_evaluate(data.train_x, data.train_y, "train")

        self._pred_evaluate(data.valid_x, data.valid_y, "valid")
        self._pred_evaluate(data.test_x, data.test_y, "test")
        K.clear_session()

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
    data_dir = os.path.join(parent_dir, "data")
    hgg_dir = os.path.join(data_dir, "HGGSegTrimmed")
    lgg_dir = os.path.join(data_dir, "LGGSegTrimmed")

    data = BTCDataset(hgg_dir, lgg_dir,
                      volume_type="t1ce",
                      pre_trainset_path="DataSplit/trainset.csv",
                      pre_validset_path="DataSplit/validset.csv",
                      pre_testset_path="DataSplit/testset.csv")
    data.run(pre_split=True)

    paras_name = "paras-1"
    paras_json_path = "hyper_paras.json"
    weights_save_dir = os.path.join(parent_dir, "weights")
    results_save_dir = os.path.join(parent_dir, "results")

    test = BTCTest(paras_name=paras_name,
                   paras_json_path=paras_json_path,
                   weights_save_dir=weights_save_dir,
                   results_save_dir=results_save_dir,
                   test_weights="last",
                   pred_trainset=True)
    test.run(data)
