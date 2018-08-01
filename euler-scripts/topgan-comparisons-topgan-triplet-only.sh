#PBS -N vs-toptriplet
#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load cudnn/7.0
module load cuda-toolkit/9.0.176
module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

gunicorn -b 0.0.0.0:5000 -w 1 --timeout 6000 run_inception_score_server:app >> /lustre/lribeiro/logs/output_server_7 2>&1 &
gunicorn_pid=$!

EXP_ID=71
python train.py --output /lustre/lribeiro/output --model topgan-small --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=72
python train.py --output /lustre/lribeiro/output --model topgan-small --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=73
python train.py --output /lustre/lribeiro/output --model topgan-small --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=74
python train.py --output /lustre/lribeiro/output --model topgan-dcgan --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=75
python train.py --output /lustre/lribeiro/output --model topgan-dcgan --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=76
python train.py --output /lustre/lribeiro/output --model topgan-dcgan --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=77
python train.py --output /lustre/lribeiro/output --model topgan-began --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=78
python train.py --output /lustre/lribeiro/output --model topgan-began --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=79
python train.py --output /lustre/lribeiro/output --model topgan-began --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Anomaly Detection Tests
# # # # # # # # # # # # # # # # # # # # # # # # # # # 
kill gunicorn_pid

EXP_ID=81
python train.py --output /lustre/lribeiro/output --model topgan-small --dataset mnist-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=82
python train.py --output /lustre/lribeiro/output --model topgan-small --dataset svhn-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=83
python train.py --output /lustre/lribeiro/output --model topgan-small --dataset cifar10-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=84
python train.py --output /lustre/lribeiro/output --model topgan-dcgan --dataset mnist-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=85
python train.py --output /lustre/lribeiro/output --model topgan-dcgan --dataset svhn-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=86
python train.py --output /lustre/lribeiro/output --model topgan-dcgan --dataset cifar10-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=87
python train.py --output /lustre/lribeiro/output --model topgan-began --dataset mnist-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=88
python train.py --output /lustre/lribeiro/output --model topgan-began --dataset svhn-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=89
python train.py --output /lustre/lribeiro/output --model topgan-began --dataset cifar10-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:0 d_loss:0 ae_loss:0 g_triplet:1 d_triplet:1 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done