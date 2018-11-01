#PBS -N topgan-2
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

# python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn-extra --batchsize 512 --zdims 128 --checkpoint-every 10 --notify-every 1 --epoch 200 --embedding-dim 256 --loss-weights 1.0 1.0 1.0 0.1 1.0 --loss-control hold hold hold-inc hold-dec none --loss-control-epoch 50 50 50 55 1 -r svhn_clas1hold_dtriplet1holdinc_gtriplet01holddec_ae1none >> /lustre/lribeiro/logs/output1.log 2>&1 &

# python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn-extra --batchsize 512 --zdims 128 --checkpoint-every 10 --notify-every 1 --epoch 100 --embedding-dim 256 --loss-weights 1.0 1.0 1.0 0.1 2.0 --loss-control hold hold hold-inc hold-dec none --loss-control-epoch 50 50 50 55 1 -r svhn_clas1hold_dtriplet1holdinc_gtriplet01holddec_ae2none >> /lustre/lribeiro/logs/output2.log 2>&1 &

# python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn --batchsize 64 --zdims 100 --checkpoint-every 5 --notify-every 2 --epoch 300 --g-triplet-weight 1.0 --d-triplet-weight 1.0 --ae-weight 0. --aux-clas-weight 0. -r $EXP_ID >> /lustre/lribeiro/logs/output2.log 2>&1 &

# python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-began --dataset celeba-128 --batchsize 32 --embedding-dim 1024 --checkpoint-every 500. --epoch 10000 --controlled-losses ae_loss:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics dagostino-normality shapiro-normality histogram-normality reconstruction samples --notify-every 1. --z-dims 100 --input-noise 0.05 --triplet-margin 1 --lr 1e-4 --use-gradnorm -r testing210 >> /lustre/lribeiro/logs/testing210.log 2>&1 &

python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-small --dataset celeba --batchsize 32 --embedding-dim 2048 --checkpoint-every 500. --epoch 10000 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics histogram-normality reconstruction samples --notify-every 5. --z-dims 100 --input-noise 0.01 --triplet-margin 1 --lr 1e-5 --use-gradnorm --gradnorm-alpha 0. --distance-metric l2 --slack-channel cvpr-experiments -r cvpr_1b >> /lustre/lribeiro/logs/testing216.log 2>&1 &

wait