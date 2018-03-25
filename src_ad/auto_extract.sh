#!/bin/sh

python features.py --data train
python features.py --data valid
python features.py --data test
