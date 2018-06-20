#PBS -N topgan-experiments
#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load cudnn/7.0
module load cuda-toolkit/9.0.176
module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

EXP_ID=1
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 0. 0. 0. --loss-control none none none none none --loss-control-epoch 1 1 1 1 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=2 
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 0. 1.0 0. --loss-control none none none hold-inc none --loss-control-epoch 1 1 1 100 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=3
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 0. 1.0 0. --loss-control none none none none none --loss-control-epoch 1 1 1 1 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=4
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 1.0 1.0 1.0 --loss-control none none none none none --loss-control-epoch 1 1 1 1 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=5
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 0.0 0.0 1.0 --loss-control none none none none none --loss-control-epoch 1 1 1 1 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=6_1
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 0. 0. 1. --loss-control hold hold hold hold none --loss-control-epoch 50 50 50 50 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=6_2
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 0.0 1.0 1. --loss-control hold hold hold hold-inc none --loss-control-epoch 50 50 50 100 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=6_3
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 0.0 1.0 1. --loss-control hold hold hold hold none --loss-control-epoch 50 50 50 50 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=6_4
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 1.0 1.0 1.0 --loss-control hold hold hold hold none --loss-control-epoch 50 50 50 50 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=7
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 0.0 0.0 1.0 1.0 0.0 --loss-control none none none none none --loss-control-epoch 1 1 1 1 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=8
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 1.0 1.0 1.0 --loss-control hold hold hold hold none --loss-control-epoch 100 100 50 50 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=9
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 1.0 1.0 1.0 --loss-control hold hold hold hold none --loss-control-epoch 50 50 100 100 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

EXP_ID=10
python train.py --output /lustre/lribeiro/output --model topgan_ae_mnist --dataset mnist --batchsize 32 --zdims 128 --checkpoint-every 25 --notify-every 1 --epoch 200 --embedding-dim 128 --loss-weights 1.0 1.0 1.0 1.0 2.0 --loss-control hold hold hold hold none --loss-control-epoch 50 50 100 100 1 --metrics tsne lda pca svm_eval samples ae_rec mmd s_inception_score svm_rbf_eval --metrics-every 1 --lr 5e-5 -r topgan_exp_$EXP_ID0 >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60

wait