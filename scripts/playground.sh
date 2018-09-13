EXP_ID=111
python train.py --model ganomaly-small --dataset mnist-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 100. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID  &
sleep 30

EXP_ID=112
python train.py --model ganomaly-small --dataset svhn-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 100. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID  &
sleep 30

EXP_ID=113
python train.py --model ganomaly-small --dataset cifar10-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 100. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID  &
sleep 30

EXP_ID=114
python train.py --model ganomaly-dcgan --dataset mnist-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 100. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID  &
sleep 30

EXP_ID=115
python train.py --model ganomaly-dcgan --dataset svhn-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 100. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID  &
sleep 30

wait

EXP_ID=116
python train.py --model ganomaly-dcgan --dataset cifar10-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 100. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID  &
sleep 30

EXP_ID=117
python train.py --model ganomaly-began --dataset mnist-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 100. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID  &
sleep 30

EXP_ID=118
python train.py --model ganomaly-began --dataset svhn-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 100. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID  &
sleep 30

EXP_ID=119
python train.py --model ganomaly-began --dataset cifar10-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 100. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID  &
sleep 30

wait