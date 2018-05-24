#PBS -N topgan
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

python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset mnist --batchsize 512 --zdims 128 --checkpoint-every 20 --notify-every 1 --epoch 100 --embedding-dim 128 --loss-weights 1.0 1.0 1.0 0.2 2.0 --loss-control hold hold hold-inc hold-dec none --loss-control-epoch 70 70 40 55 90 -r mnist_clas1hold_dtriplet1holdinc_gtriplet01holddec_ae2halt_lc70_70_40_55_90 >> /lustre/lribeiro/logs/output3.log 2>&1 &

python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset mnist --batchsize 512 --zdims 128 --checkpoint-every 20 --notify-every 1 --epoch 100 --embedding-dim 128 --loss-weights 1.0 1.0 1.0 0.1 2.0 --loss-control hold hold hold-inc hold-dec none --loss-control-epoch 70 70 40 55 1 -r mnist_clas1hold_dtriplet1holdinc_gtriplet01holddec_ae2none_lc70_70_40_55_1 >> /lustre/lribeiro/logs/output4.log 2>&1 &

python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset mnist --batchsize 512 --zdims 128 --checkpoint-every 20 --notify-every 1 --epoch 100 --embedding-dim 128 --loss-weights 1.0 1.0 1.0 1.0 2.0 --loss-control hold hold hold-inc hold-dec none --loss-control-epoch 80 80 40 20 1 -r mnist_clas1hold_dtriplet1holdinc_gtriplet1holddec_ae2none_lc70_70_40_55_1 >> /lustre/lribeiro/logs/output5.log 2>&1 &

python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset mnist --batchsize 512 --zdims 128 --checkpoint-every 20 --notify-every 1 --epoch 100 --embedding-dim 128 --loss-weights 0.0 0.0 1.0 1.0 2.0 --loss-control none none hold-inc hold-inc none --loss-control-epoch 1 1 40 20 1 -r mnist_clas0none_dtriplet1holdinc_gtriplet1holddec_ae2none_lc70_70_40_55_1 >> /lustre/lribeiro/logs/output5.log 2>&1 &

# python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn --batchsize 64 --zdims 100 --checkpoint-every 5 --notify-every 2 --epoch 300 --g-triplet-weight 1.0 --d-triplet-weight 1.0 --ae-weight 0. --aux-clas-weight 0. -r $EXP_ID >> /lustre/lribeiro/logs/output2.log 2>&1 &

wait