#PBS -N topgan-vs-iwgan
#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load cudnn/7.0
module load cuda-toolkit/9.0.176
module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

gunicorn -b 0.0.0.0:5000 -w 1 --timeout 6000 run_inception_score_server:app >> /lustre/lribeiro/logs/output_server_3 2>&1 &

EXP_ID=31
python train.py --output /lustre/lribeiro/output --model improved-wgan-small --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 --wgan-n-critic 2 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=32
python train.py --output /lustre/lribeiro/output --model improved-wgan-small --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 --wgan-n-critic 2 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=33
python train.py --output /lustre/lribeiro/output --model improved-wgan-small --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 --wgan-n-critic 2 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=34
python train.py --output /lustre/lribeiro/output --model improved-wgan-dcgan --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 --wgan-n-critic 3 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=35
python train.py --output /lustre/lribeiro/output --model improved-wgan-dcgan --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 --wgan-n-critic 3 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=36
python train.py --output /lustre/lribeiro/output --model improved-wgan-dcgan --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 --wgan-n-critic 3 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=37
python train.py --output /lustre/lribeiro/output --model improved-wgan-began --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 --wgan-n-critic 3 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=38
python train.py --output /lustre/lribeiro/output --model improved-wgan-began --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 --wgan-n-critic 3 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=39
python train.py --output /lustre/lribeiro/output --model improved-wgan-began --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 --wgan-n-critic 3 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done