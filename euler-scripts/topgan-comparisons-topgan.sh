#PBS -N vs-topgan
#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load cudnn/7.0
module load cuda-toolkit/9.0.176
module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

gunicorn -b 0.0.0.0:5000 -w 1 --timeout 6000 run_inception_score_server:app >> /lustre/lribeiro/logs/output_server_3 2>&1 &
gunicorn_pid=$!

EXP_ID=41
python train.py --output /lustre/lribeiro/output --model topgan-ae-small --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=42
python train.py --output /lustre/lribeiro/output --model topgan-ae-small --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=43
python train.py --output /lustre/lribeiro/output --model topgan-ae-small --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=44
python train.py --output /lustre/lribeiro/output --model topgan-ae-dcgan --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=45
python train.py --output /lustre/lribeiro/output --model topgan-ae-dcgan --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=46
python train.py --output /lustre/lribeiro/output --model topgan-ae-dcgan --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=47
python train.py --output /lustre/lribeiro/output --model topgan-ae-began --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=48
python train.py --output /lustre/lribeiro/output --model topgan-ae-began --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=49
python train.py --output /lustre/lribeiro/output --model topgan-ae-began --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_eval svm_rbf_eval tsne pca lda samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Anomaly Detection Tests
# # # # # # # # # # # # # # # # # # # # # # # # # # # 
kill gunicorn_pid

EXP_ID=51
python train.py --output /lustre/lribeiro/output --model topgan-ae-small --dataset mnist-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=52
python train.py --output /lustre/lribeiro/output --model topgan-ae-small --dataset svhn-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=53
python train.py --output /lustre/lribeiro/output --model topgan-ae-small --dataset cifar10-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=54
python train.py --output /lustre/lribeiro/output --model topgan-ae-dcgan --dataset mnist-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=55
python train.py --output /lustre/lribeiro/output --model topgan-ae-dcgan --dataset svhn-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=56
python train.py --output /lustre/lribeiro/output --model topgan-ae-dcgan --dataset cifar10-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=57
python train.py --output /lustre/lribeiro/output --model topgan-ae-began --dataset mnist-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=58
python train.py --output /lustre/lribeiro/output --model topgan-ae-began --dataset svhn-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=59
python train.py --output /lustre/lribeiro/output --model topgan-ae-began --dataset cifar10-anomaly --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --controlled-losses g_loss:1:hold:200 d_loss:1:hold:200 ae_loss:1 g_triplet:1:hold:50 d_triplet:1:hold:50 --metrics reconstruction svm_anomaly tsne pca samples --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done