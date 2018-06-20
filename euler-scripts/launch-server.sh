#PBS -N inception-score-server
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

uniq $PBS_NODEFILE > /home/lribeiro/projects/triplet-ali/server_for_inception_score.config
gunicorn -b 0.0.0.0:5000 -w 1 --timeout 6000 run_inception_score_server:app >> /lustre/lribeiro/logs/output_server 2>&1
