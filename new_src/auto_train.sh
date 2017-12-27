#!/bin/sh
python train.py --volume t1ce --model pyramid --opt adam
python train.py --volume flair --model pyramid --opt adam
python train.py --volume t1ce --model vggish --opt adam
python train.py --volume flair --model vggish --opt adam
