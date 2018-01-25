for zdims in 32
do
    python3 train.py --zdims=${zdims} --output=out_bbc_${zdims} --model=ali --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=100 --epoch=10 --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc_${zdims}/ali/weights/epoch_00007
done

for zdims in 64
do
    python3 train.py --zdims=${zdims} --output=out_bbc_${zdims} --model=ali --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=100 --epoch=10 --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc_${zdims}/ali/weights/epoch_00005
done


for zdims in 16 32 64
do
    python3 train.py --zdims=${zdims} --output=out_bbc_${zdims} --model=ali --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=100 --epoch=20 --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc_${zdims}/ali/weights/epoch_00010
done

for zdims in 16 32 64
do
    python3 train.py --zdims=${zdims} --output=out_bbc_${zdims} --model=drali --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=100 --epoch=10 --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc_${zdims}/drali/weights/epoch_00005
done
