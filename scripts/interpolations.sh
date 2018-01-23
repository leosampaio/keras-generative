export PYTHONPATH=$PYTHONPATH:/home/alex/Desktop/proj/sign-lang/keras-generative

for z_dims in 16 32 64 128
do
    python3 analysis/ali_sanity_check.py --z_dims=${z_dims} --weights=out_bbc_${z_dims}/ali/weights/epoch_00005
done