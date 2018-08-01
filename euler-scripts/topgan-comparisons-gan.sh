#PBS -N vs-gan
#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load cudnn/7.0
module load cuda-toolkit/9.0.176
module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

gunicorn -b 0.0.0.0:5000 -w 1 --timeout 6000 run_inception_score_server:app >> /lustre/lribeiro/logs/output_server_6 2>&1 &
gunicorn_pid=$!

EXP_ID=61
python train.py --output /lustre/lribeiro/output --model topgan-small --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1 d_loss:1 ae_loss:0 g_triplet:0 d_triplet:0 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=62
python train.py --output /lustre/lribeiro/output --model topgan-small --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1 d_loss:1 ae_loss:0 g_triplet:0 d_triplet:0 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=63
python train.py --output /lustre/lribeiro/output --model topgan-small --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1 d_loss:1 ae_loss:0 g_triplet:0 d_triplet:0 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=64
python train.py --output /lustre/lribeiro/output --model topgan-dcgan --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1 d_loss:1 ae_loss:0 g_triplet:0 d_triplet:0 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=65
python train.py --output /lustre/lribeiro/output --model topgan-dcgan --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1 d_loss:1 ae_loss:0 g_triplet:0 d_triplet:0 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=66
python train.py --output /lustre/lribeiro/output --model topgan-dcgan --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1 d_loss:1 ae_loss:0 g_triplet:0 d_triplet:0 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=67
python train.py --output /lustre/lribeiro/output --model topgan-began --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1 d_loss:1 ae_loss:0 g_triplet:0 d_triplet:0 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=68
python train.py --output /lustre/lribeiro/output --model topgan-began --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1 d_loss:1 ae_loss:0 g_triplet:0 d_triplet:0 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=69
python train.py --output /lustre/lribeiro/output --model topgan-began --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1 d_loss:1 ae_loss:0 g_triplet:0 d_triplet:0 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done