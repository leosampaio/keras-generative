
#for zdims in 128 
#do
    #python3 train.py --zdims=${zdims} --output=out_multi_${zdims} --model=ali --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10
    #python3 train.py --zdims=${zdims} --output=out_bbc_${zdims}   --model=ali  --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5  --batchsize=100 --epoch=5 

    #python3 train.py --zdims=${zdims} --output=out_multi_${zdims} --model=drali --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10
    #python3 train.py --zdims=${zdims} --output=out_bbc_${zdims}   --model=drali --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5   --batchsize=100 --epoch=5 --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc_128/drali/weights/epoch_00001
#done

#for zdims in 64
#do
    #python3 train.py --zdims=${zdims} --output=out_multi_${zdims} --model=ali --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10
    #python3 train.py --zdims=${zdims} --output=out_bbc_${zdims}   --model=ali  --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5  --batchsize=100 --epoch=5 

    #python3 train.py --zdims=${zdims} --output=out_multi_${zdims} --model=drali --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10
    #python3 train.py --zdims=${zdims} --output=out_bbc_${zdims}   --model=drali --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5   --batchsize=100 --epoch=5 
#done

#for zdims in 32
#do
    #python3 train.py --zdims=${zdims} --output=out_multi_${zdims} --model=ali --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10
    #python3 train.py --zdims=${zdims} --output=out_bbc_${zdims}   --model=ali  --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5  --batchsize=100 --epoch=5 

    #python3 train.py --zdims=${zdims} --output=out_multi_${zdims} --model=drali --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10
    #python3 train.py --zdims=${zdims} --output=out_bbc_${zdims}   --model=drali --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5   --batchsize=100 --epoch=5 
#done

for zdims in 16 
do
    #python3 train.py --zdims=${zdims} --output=out_multi_${zdims} --model=ali --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10 --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_multi_16/ali/weights/epoch_00004
    #python3 train.py --zdims=${zdims} --output=out_bbc_${zdims}   --model=ali  --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5  --batchsize=100 --epoch=5 --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc_16/ali/weights/epoch_00004

    #python3 train.py --zdims=${zdims} --output=out_multi_${zdims} --model=drali --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10
    python3 train.py --zdims=${zdims} --output=out_bbc_${zdims}   --model=drali --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5   --batchsize=100 --epoch=5 --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc_16/drali/weights/epoch_00001
done
