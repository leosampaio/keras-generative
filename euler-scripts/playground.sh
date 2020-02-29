#PBS -N TOPGAN
#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

# export PATH="$PATH:/home/lribeiro/lib/cuda-9.0/bin"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/lribeiro/lib/cuda-9.0/lib64"
# /home/lribeiro/lib/cuda-9.0/lib64/libcudnn*


module load cudnn/7.0
module load cuda-toolkit/9.0.176
module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

# EXP_ID=svhn_r6_gtriplet01_dtripletae2_mae_b256_lr5
# python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn --batchsize 128 --zdims 100 --checkpoint-every 2 --notify-every 2 --epoch 400 --triplet-weight 0.1 --lr 1e-5 --resume  /lustre/lribeiro/output/topgan_ae_ebgan_r$EXP_ID/weights/epoch_00100/ -r $EXP_ID \
# >> /lustre/lribeiro/logs/$EXP_ID 2>&1 &

# EXP_ID=svhn_r6_gtriplet01_dtripletae2_mae_b256
# python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn --batchsize 128 --zdims 100 --checkpoint-every 2 --notify-every 2 --epoch 400 --triplet-weight 0.1 --resume  /lustre/lribeiro/output/topgan_ae_ebgan_r$EXP_ID/weights/epoch_00100/ -r $EXP_ID \
# >> /lustre/lribeiro/logs/$EXP_ID 2>&1 &

# --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4a_r2_cifar10/weights/pi_24750080 -r cvpr_4a_ivom_r2 >> /lustre/lribeiro/logs/cvpr_4a_ivom_r2 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4a_r3_cifar10/weights/pi_24750080 -r cvpr_4a_ivom_r3 >> /lustre/lribeiro/logs/cvpr_4a_ivom_r3 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4a_r4_cifar10/weights/pi_24750080 -r cvpr_4a_ivom_r4 >> /lustre/lribeiro/logs/cvpr_4a_ivom_r4 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4a_r5_cifar10/weights/pi_24750080 -r cvpr_4a_ivom_r5 >> /lustre/lribeiro/logs/cvpr_4a_ivom_r5 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/improved_wgan_dcgan_rcvpr_4c_r2_cifar10/weights/pi_24750080 -r cvpr_4c_ivom_r2 >> /lustre/lribeiro/logs/cvpr_4c_ivom_r2 2>&1 &

wait

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/improved_wgan_dcgan_rcvpr_4c_r3_cifar10/weights/pi_24750080 -r cvpr_4c_ivom_r3 >> /lustre/lribeiro/logs/cvpr_4c_ivom_r3 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/improved_wgan_dcgan_rcvpr_4c_r4_cifar10/weights/pi_24750080 -r cvpr_4c_ivom_r4 >> /lustre/lribeiro/logs/cvpr_4c_ivom_r4 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/improved_wgan_dcgan_rcvpr_4c_r5_cifar10/weights/pi_24750080 -r cvpr_4c_ivom_r5 >> /lustre/lribeiro/logs/cvpr_4c_ivom_r5 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/began_dcgan_rcvpr_4e_r2_cifar10/weights/pi_24750080 -r cvpr_4e_ivom_r2 >> /lustre/lribeiro/logs/cvpr_4e_ivom_r2 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/began_dcgan_rcvpr_4e_r3_cifar10/weights/pi_24750080 -r cvpr_4e_ivom_r3 >> /lustre/lribeiro/logs/cvpr_4e_ivom_r3 2>&1 &

wait



python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/began_dcgan_rcvpr_4e_r4_cifar10/weights/pi_24750080 -r cvpr_4e_ivom_r4 >> /lustre/lribeiro/logs/cvpr_4e_ivom_r4 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume /lustre/lribeiro/output/began_dcgan_rcvpr_4e_r5_cifar10/weights/pi_24750080 -r cvpr_4e_ivom_r5 >> /lustre/lribeiro/logs/cvpr_4e_ivom_r5 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume /lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4g_r2_cifar10/weights/pi_24750080 -r cvpr_4g_ivom_r2 >> /lustre/lribeiro/logs/cvpr_4g_ivom_r2 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume /lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4g_r3_cifar10/weights/pi_24750080 -r cvpr_4g_ivom_r3 >> /lustre/lribeiro/logs/cvpr_4g_ivom_r3 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume /lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4g_r4_cifar10/weights/pi_24750080 -r cvpr_4g_ivom_r4 >> /lustre/lribeiro/logs/cvpr_4g_ivom_r4 2>&1 &

python eval-cifar-ivom.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume /lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4g_r5_cifar10/weights/pi_24750080 -r cvpr_4g_ivom_r5 >> /lustre/lribeiro/logs/cvpr_4g_ivom_r5 2>&1 &

wait