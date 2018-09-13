#PBS -N imagenet
#PBS -l ncpus=1
#PBS -l walltime=72:00:00

module load curl/7.50.3
cd /lustre/lribeiro/data/

curl  --output ILSVRC2012_img_train.tar >> /lustre/lribeiro/logs/output_imagenet -C - http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar 2>&1