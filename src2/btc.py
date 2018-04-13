from __future__ import print_function


import os
import json
import argparse
from btc_test import BTCTest
from btc_train import BTCTrain
from btc_dataset import BTCDataset
from btc_preprocess import BTCPreprocess


def main(hyper_paras_name):

    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, pre_paras["data_dir"])

    hgg_in_dir = os.path.join(data_dir, pre_paras["hgg_in"])
    lgg_in_dir = os.path.join(data_dir, pre_paras["lgg_in"])

    hgg_out_dir = os.path.join(data_dir, pre_paras["hgg_out"])
    lgg_out_dir = os.path.join(data_dir, pre_paras["lgg_out"])

    weights_save_dir = os.path.join(parent_dir, pre_paras["weights_save_dir"])
    logs_save_dir = os.path.join(parent_dir, pre_paras["logs_save_dir"])
    results_save_dir = os.path.join(parent_dir, pre_paras["results_save_dir"])

    # Preprocessing
    # prep = BTCPreprocess([hgg_in_dir, lgg_in_dir],
    #                      [hgg_out_dir, lgg_out_dir],
    #                      pre_paras["volume_type"])
    # prep.run(is_mask=pre_paras["is_mask"],
    #          non_mask_coeff=pre_paras["non_mask_coeff"],
    #          processes=pre_paras["processes_num"])

    # Getting splitted dataset
    data = BTCDataset(hgg_out_dir, lgg_out_dir,
                      volume_type=pre_paras["volume_type"],
                      train_prop=pre_paras["train_prop"],
                      valid_prop=pre_paras["valid_prop"],
                      random_state=pre_paras["random_state"],
                      pre_trainset_path=pre_paras["pre_trainset_path"],
                      pre_validset_path=pre_paras["pre_validset_path"],
                      pre_testset_path=pre_paras["pre_testset_path"],
                      data_format=pre_paras["data_format"])
    data.run(pre_split=pre_paras["pre_split"],
             save_split=pre_paras["save_split"],
             save_split_dir=pre_paras["save_split_dir"])

    # Training the model
    train = BTCTrain(paras_name=hyper_paras_name,
                     paras_json_path=pre_paras["paras_json_path"],
                     weights_save_dir=weights_save_dir,
                     logs_save_dir=logs_save_dir,
                     save_best_weights=pre_paras["save_best_weights"])
    train.run(data)

    # Testing the model
    test = BTCTest(paras_name=hyper_paras_name,
                   paras_json_path=pre_paras["paras_json_path"],
                   weights_save_dir=weights_save_dir,
                   results_save_dir=results_save_dir,
                   test_weights=pre_paras["test_weights"],
                   pred_trainset=pre_paras["pred_trainset"])
    test.run(data)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    help_str = "Select a set of hyper-parameters in hyper_paras.json"
    parser.add_argument("--paras", action="store", default="paras-1",
                        dest="hyper_paras_name", help=help_str)

    args = parser.parse_args()
    main(args.hyper_paras_name)
