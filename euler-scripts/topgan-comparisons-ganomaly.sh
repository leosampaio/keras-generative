#PBS -N topgan-vs-ganomaly
#PBS -l select=1:ngpus=1
#PBS -l walltime=336:00:00

module load cudnn/7.0
module load cuda-toolkit/9.0.176
module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

gunicorn -b 0.0.0.0:5000 -w 1 --timeout 6000 run_inception_score_server:app >> /lustre/lribeiro/logs/output_server_4 2>&1 &

EXP_ID=41
python train.py --output /lustre/lribeiro/output --model ganomaly-small --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=42
python train.py --output /lustre/lribeiro/output --model ganomaly-small --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=43
python train.py --output /lustre/lribeiro/output --model ganomaly-small --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=44
python train.py --output /lustre/lribeiro/output --model ganomaly-dcgan --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done

EXP_ID=45
python train.py --output /lustre/lribeiro/output --model ganomaly-dcgan --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=46
python train.py --output /lustre/lribeiro/output --model ganomaly-dcgan --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=47
python train.py --output /lustre/lribeiro/output --model ganomaly-began --dataset mnist --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=48
python train.py --output /lustre/lribeiro/output --model ganomaly-began --dataset svhn --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

EXP_ID=49
python train.py --output /lustre/lribeiro/output --model ganomaly-began --dataset cifar10 --batchsize 32 --z-dims 100 --checkpoint-every 50. --epoch 400 --embedding-dim 100 --metrics svm_anomaly tsne pca reconstruction samples m_inception_score mmd --notify-every 5. --lr 1e-4 -r tcomp_$EXP_ID >> /lustre/lribeiro/logs/output_$EXP_ID 2>&1 &
sleep 60
pids[$EXP_ID]=$!

for pid in ${pids[*]}; do
    wait $pid
done