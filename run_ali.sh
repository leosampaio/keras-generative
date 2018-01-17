#python3 train.py --model=ali         --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10 --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/output/ali/weights/epoch_00004
#python3 train.py --model=dcgan       --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=50 --output=out_impr
#python3 train.py --model=improvedgan --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=50 --output=out_impr

#python3 train.py --model=ali         --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=100 --epoch=5 --output=out_bbc --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc/ali/weights/epoch_00001
#python3 train.py --model=dcgan       --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=20 --output=out_bbc --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc/dcgan/weights/epoch_00012
#python3 train.py --model=improvedgan --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=20 --output=out_bbc --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc/improved_gan/weights/epoch_00012

#python3 train.py --model=vae --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=100 --epoch=20 --output=out_bbc --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc/vae/weights/epoch_00001

#python3 train.py --model=drvae --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=100 --epoch=20 --output=out_bbc 
#python3 train.py --model=drali         --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=100 --epoch=5 --output=out_bbc 
#python3 train.py --model=drdcgan       --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=20 --output=out_bbc 
#python3 train.py --model=drimprovedgan --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=20 --output=out_bbc 

#python3 train.py --model=drali         --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10 --output=out_multi 
#python3 train.py --model=drdcgan       --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_multi/drdcgan/weights/epoch_00016
#python3 train.py --model=drimprovedgan --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi

#python3 train.py --model=vdcgan --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=10 --output=out_bbc  
#python3 train.py --model=vdimprovedgan --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=10 --output=out_bbc 
#python3 train.py --model=aae --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=10 --output=out_bbc 
#python3 train.py --model=binaae --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=10 --output=out_bbc --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc/baae/weights/epoch_00005
#python3 train.py --model=aae2 --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=10 --output=out_bbc 
#python3 train.py --model=draae2 --dataset=/home/alex/datasets/bbc_full_r_pr.hdf5 --batchsize=200 --epoch=10 --output=out_bbc 

#python3 train.py --model=ali         --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=100 --epoch=10 --output=out_multi 
#python3 train.py --model=dcgan       --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_multi/dcgan/weights/epoch_00025
#python3 train.py --model=improvedgan --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi

#python3 train.py --model=vdcgan --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_multi/vdcgan/weights/epoch_00013
#python3 train.py --model=vdimprovedgan --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi 
#python3 train.py --model=aae --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi --resume=/home/alex/Desktop/proj/sign-lang/keras-generative/out_multi/aae/weights/epoch_00025
#python3 train.py --model=binaae --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi 
#python3 train.py --model=aae2 --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi 
#python3 train.py --model=draae2 --dataset=/home/alex/datasets/multi_330k_r_pr.hdf5 --batchsize=200 --epoch=30 --output=out_multi
