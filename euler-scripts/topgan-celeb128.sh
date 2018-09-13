#PBS -N celeb128
#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load cudnn/7.0
module load cuda-toolkit/9.0.176
module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

# EXP_ID=77
# python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset celeba-128 --batchsize 16 --embedding-dim 256 --checkpoint-every 500. --epoch 10000 --controlled-losses g_loss:1:hold:1000 d_loss:1:hold:1000 ae_loss:1 g_triplet:1:hold:300 d_triplet:1:hold:300 contrastive_ae_n:1 --metrics samples reconstruction --notify-every 10. --z-dims 100 --input-noise 0.1 --lr 1e-5 --n-filters-factor 64 -r test$EXP_ID >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1 &
# nvidia-smi >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1


# EXP_ID=78
# python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset celeba-128 --batchsize 16 --embedding-dim 256 --checkpoint-every 500. --epoch 10000 --controlled-losses g_loss:1:hold:1000 d_loss:1:hold:1000 ae_loss:1 g_triplet:1:hold:300 d_triplet:1:hold:300 contrastive_ae_n:1 --metrics samples reconstruction --notify-every 10. --z-dims 100 --input-noise 0.1 --lr 1e-4 --n-filters-factor 64 -r test$EXP_ID >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1 &
# nvidia-smi >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1

# EXP_ID=79
# python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-began --dataset celeba-128 --batchsize 16 --embedding-dim 128 --checkpoint-every 500. --epoch 10000 --controlled-losses g_loss:1:hold:500 d_loss:1:hold:500 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics samples reconstruction --notify-every 10. --z-dims 100 --input-noise 0.1 --lr 1e-4 --n-filters-factor 64 -r test$EXP_ID >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1 &
# nvidia-smi >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1

# EXP_ID=80
# python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-began --dataset celeba --batchsize 16 --embedding-dim 128 --checkpoint-every 500. --epoch 10000 --controlled-losses g_loss:1:hold:500 d_loss:1:hold:500 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics samples reconstruction --notify-every 10. --z-dims 100 --input-noise 0.1 --lr 1e-4 --n-filters-factor 64 -r test$EXP_ID >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1 &
# nvidia-smi >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1

# EXP_ID=81
# python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset celeba --batchsize 16 --embedding-dim 256 --checkpoint-every 500. --epoch 10000 --controlled-losses g_loss:1:hold:1000 d_loss:1:hold:1000 ae_loss:1 g_triplet:1:hold:300 d_triplet:1:hold:300 contrastive_ae_n:1 --metrics samples reconstruction --notify-every 10. --z-dims 100 --input-noise 0.1 --lr 1e-5 -r test$EXP_ID >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1 &
# nvidia-smi >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1

# EXP_ID=105
# python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-began --dataset celeba --batchsize 256 --embedding-dim 128 --checkpoint-every 500. --epoch 10000 --controlled-losses g_loss:0 d_loss:0 ae_loss:1 g_triplet:1 d_triplet:1 --metrics samples reconstruction --notify-every 10. --z-dims 100 --triplet-margin 32 --use-magan-equilibrium --input-noise 0.1 --lr 1e-4 -r test$EXP_ID >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1 &
# nvidia-smi >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1

EXP_ID=114
python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-began --dataset celeba-128 --batchsize 64 --embedding-dim 128 --checkpoint-every 500. --epoch 20000 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics samples --n-filters-factor 64 --notify-every 10. --z-dims 100 --triplet-margin 32 --use-magan-equilibrium --input-noise 0.0 --lr 1e-4 -r test$EXP_ID >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1 &
nvidia-smi >> /lustre/lribeiro/logs/output_test$EXP_ID 2>&1

wait