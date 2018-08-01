#PBS -N topgan-vs-began
#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load cudnn/7.0
module load cuda-toolkit/9.0.176
module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

gunicorn -b 0.0.0.0:5000 -w 1 --timeout 6000 run_inception_score_server:app >> /lustre/lribeiro/logs/output_server_2 2>&1 &

EXP_ID=21
python train.py --output /lustre/lribeiro/output --model began-small --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd reconstruction --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=22
python train.py --output /lustre/lribeiro/output --model began-small --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd reconstruction --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=23
python train.py --output /lustre/lribeiro/output --model began-small --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd reconstruction --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=24
python train.py --output /lustre/lribeiro/output --model began-dcgan --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd reconstruction --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=25
python train.py --output /lustre/lribeiro/output --model began-dcgan --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd reconstruction --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=26
python train.py --output /lustre/lribeiro/output --model began-dcgan --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd reconstruction --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=27
python train.py --output /lustre/lribeiro/output --model began-began --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd reconstruction --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=28
python train.py --output /lustre/lribeiro/output --model began-began --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd reconstruction --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=29
python train.py --output /lustre/lribeiro/output --model began-began --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd reconstruction --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done