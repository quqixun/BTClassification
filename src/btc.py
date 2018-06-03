# Brain Tumor Classification
# Main script contains whole process.
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
import argparse
from btc_test import BTCTest
from btc_train import BTCTrain
from btc_dataset import BTCDataset
from btc_preprocess import BTCPreprocess


def main(hyper_paras_name):
    '''MAIN

        Main process of Brain Tumor Classification.
        -1- Split dataset for training, validating and testing.
        -2- Train model.
        -3- Test model.

        Inputs:
        -------

        - hyper_paras_name: string, the name of hyperparanters set,
                            which can be found in hyper_paras.json.

    '''

    # Basic settings in pre_paras.json, including
    # 1. directory paths for input and output
    # 2. necessary information for splitting dataset
    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    # Get root path of input data
    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, pre_paras["data_dir"])

    # Set directories of input images
    hgg_in_dir = os.path.join(data_dir, pre_paras["hgg_in"])
    lgg_in_dir = os.path.join(data_dir, pre_paras["lgg_in"])

    # Set output directory to save preprocesses images
    hgg_out_dir = os.path.join(data_dir, pre_paras["hgg_out"])
    lgg_out_dir = os.path.join(data_dir, pre_paras["lgg_out"])

    # Set directory to save weights
    weights_save_dir = os.path.join(parent_dir, pre_paras["weights_save_dir"])
    # Set directory to save training and validation logs
    logs_save_dir = os.path.join(parent_dir, pre_paras["logs_save_dir"])
    # Set directory to save metrics
    results_save_dir = os.path.join(parent_dir, pre_paras["results_save_dir"])

    # Preprocessing to enhance tumor regions
    prep = BTCPreprocess([hgg_in_dir, lgg_in_dir],
                         [hgg_out_dir, lgg_out_dir],
                         pre_paras["volume_type"])
    prep.run(is_mask=pre_paras["is_mask"],
             non_mask_coeff=pre_paras["non_mask_coeff"],
             processes=pre_paras["processes_num"])

    # Split dataset
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

    # Training the model using enhanced tumor regions
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

    # Command line
    # python add.py --paras=paras-1

    parser = argparse.ArgumentParser()

    # Set json file path to extract hyperparameters
    help_str = "Select a set of hyper-parameters in hyper_paras.json"
    parser.add_argument("--paras", action="store", default="paras-1",
                        dest="hyper_paras_name", help=help_str)

    args = parser.parse_args()
    main(args.hyper_paras_name)
