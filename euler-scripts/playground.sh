#PBS -N triplet_ealice_elcc_ds_stylel_r3
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00

export PATH="$PATH:/home/lribeiro/lib/cuda-9.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/lribeiro/lib/cuda-9.0/lib64"
/home/lribeiro/lib/cuda-9.0/lib64/libcudnn*

module load python/3.4.3
cd /home/lribeiro/projects/triplet-ali
source env/bin/activate

nvidia-smi > /lustre/lribeiro/output.txt

python train_cross_domain.py --model triplet_ealice_elcc_ds_stylel \
    --submodels ealice_shareable ealice_shareable \
    --dataset mnist-svhn \
    --zdims 512 --epoch 1000 --batchsize 256 \
    --triplet-margin 1.0 --triplet-weight 1.0 \
    --output /lustre/lribeiro/output/ \
    -r 3 --checkpoint-every 5 --n-layers-to-share 0 \
    --lr 1e-5\
>> /lustre/lribeiro/output.txt 2>&1