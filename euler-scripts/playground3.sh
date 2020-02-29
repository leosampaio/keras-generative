#PBS -N topgan3
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

# --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data

I=cvpr_5a_r4
python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r $I >> /lustre/lribeiro/logs/$I 2>&1 &

I=cvpr_5a_r5
python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r $I >> /lustre/lribeiro/logs/$I 2>&1 &


I=cvpr_5b_r4
python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model improved-wgan-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r $I >> /lustre/lribeiro/logs/$I 2>&1 &

I=cvpr_5b_r5
python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model improved-wgan-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r $I >> /lustre/lribeiro/logs/$I 2>&1 &


I=cvpr_5c_r4
python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model began-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r $I >> /lustre/lribeiro/logs/$I 2>&1 &

I=cvpr_5c_r5
python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model began-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r $I >> /lustre/lribeiro/logs/$I 2>&1 &

I=cvpr_5d_r4
python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics histogram-normality samples reconstruction mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --use-gradnorm --gradnorm-alpha 0. --slack-channel cvpr-experiments -r $I >> /lustre/lribeiro/logs/$I 2>&1 &

I=cvpr_5d_r5
python train.py --output /lustre/lribeiro/output --data-folder /lustre/lribeiro/data --model topgan-ae-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics histogram-normality samples reconstruction mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --use-gradnorm --gradnorm-alpha 0. --slack-channel cvpr-experiments -r $I >> /lustre/lribeiro/logs/$I 2>&1 &

wait