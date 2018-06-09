#!/bin/sh


# Brain Tumor Classification
# Commands for training and testing models.
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


#
# Section 1
#
# Train and test model
# Command:
# python add.py --paras=paras_name
# Parameters:
# - paras: hyperparameters set in hyper_paras.json

# Using enhanced tumor regions to train model:
# In pre_paras.json, set
#    "hgg_out": "HGGSegTrimmed"
#    "lgg_out": "LGGSegTrimmed"

# Using non-enhanced tumor regions to train model:
# In pre_paras.json, set
#    "hgg_out": "HGGTrimmed"
#    "lgg_out": "LGGTrimmed"

python btc.py --paras=paras-1
# python btc.py --paras=paras-2


#
# Section 2
#
# Train and test model respectively
# Commands:
# python btc_train.py --paras=paras_name
# python btc_test.py --paras=paras_name
# Same parameter as in Section 1

# python btc_train.py --paras=paras-1
# python btc_test.py --paras=paras-1
