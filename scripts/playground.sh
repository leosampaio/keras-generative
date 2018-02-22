#! /bin/bash

python train.py --model ali_SVHN --epoch 100 --batchsize 100 --output=output090218 --dataset svhn --zdims 256 --label_smoothing 0.1
 python train.py --model ali_SVHN --dataset svhn --zdims 256 --output=output140218 --epoch 300 --batchsize 256 --input_noise 0.01 -r 6 --resume output140218/ali_for_svhn_r6/weights/epoch_00140/