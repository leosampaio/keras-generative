#! /bin/bash

python train.py --model ali_SVHN --epoch 100 --batchsize 100 --output=output090218 --dataset svhn --zdims 256 --label_smoothing 0.1