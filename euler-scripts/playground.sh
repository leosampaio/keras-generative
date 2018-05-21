#PBS -N triplet_ealice_elcc_ds_stylel_r3
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00

export PATH="$PATH:/home/lribeiro/lib/cuda-9.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/lribeiro/lib/cuda-9.0/lib64"
# /home/lribeiro/lib/cuda-9.0/lib64/libcudnn*

module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

# EXP_ID=svhn_r6_gtriplet01_dtripletae2_mae_b256_lr5
# python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn --batchsize 128 --zdims 100 --checkpoint-every 2 --notify-every 2 --epoch 400 --triplet-weight 0.1 --lr 1e-5 --resume  /lustre/lribeiro/output/topgan_ae_ebgan_r$EXP_ID/weights/epoch_00100/ -r $EXP_ID \
# >> /lustre/lribeiro/logs/$EXP_ID 2>&1 &

# EXP_ID=svhn_r6_gtriplet01_dtripletae2_mae_b256
# python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn --batchsize 128 --zdims 100 --checkpoint-every 2 --notify-every 2 --epoch 400 --triplet-weight 0.1 --resume  /lustre/lribeiro/output/topgan_ae_ebgan_r$EXP_ID/weights/epoch_00100/ -r $EXP_ID \
# >> /lustre/lribeiro/logs/$EXP_ID 2>&1 &

EXP_ID=svhn_gtriplet1_dtriplet1_ae1_auxclas0
python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn --batchsize 64 --zdims 100 --checkpoint-every 5 --notify-every 2 --epoch 300 --g-triplet-weight 1.0 --d-triplet-weight 1.0 --ae-weight 1. --aux-clas-weight 0. -r $EXP_ID >> /lustre/lribeiro/logs/$EXP_ID 2>&1 &

EXP_ID=svhn_gtriplet1_dtriplet1_ae0_auxclas0
python train.py --output /lustre/lribeiro/output --model topgan_ae_ebgan --dataset svhn --batchsize 64 --zdims 100 --checkpoint-every 5 --notify-every 2 --epoch 300 --g-triplet-weight 1.0 --d-triplet-weight 1.0 --ae-weight 0. --aux-clas-weight 0. -r $EXP_ID >> /lustre/lribeiro/logs/$EXP_ID 2>&1 &

wait