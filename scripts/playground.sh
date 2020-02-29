# TOPGAN Experiments

# 3e_r(1-5)

python train.py --model improved-wgan-mlp-synth-veegan --dataset synthetic-8ring --batchsize 128 --embedding-dim 128 --checkpoint-every 249. --epoch 500 --metrics synthetic-data-vis mode-coverage high-quality-ratio --notify-every 1. --send-every 20 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_3e_r

# 3f_r(1-5)

python train.py --model improved-wgan-mlp-synth-veegan --dataset synthetic-25grid --batchsize 128 --embedding-dim 128 --checkpoint-every 249. --epoch 500 --metrics synthetic-data-vis mode-coverage high-quality-ratio --notify-every 1. --send-every 20 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_3f_r

# 3g_r(1-5)

python train.py --model began-mlp-synth-veegan --dataset synthetic-8ring --batchsize 128 --embedding-dim 2 --checkpoint-every 249. --epoch 500 --metrics synthetic-data-vis mode-coverage high-quality-ratio --notify-every 1. --send-every 20 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_3g_r

# 3h_r(1-5)

python train.py --model began-mlp-synth-veegan --dataset synthetic-25grid --batchsize 128 --embedding-dim 2 --checkpoint-every 249. --epoch 500 --metrics synthetic-data-vis mode-coverage high-quality-ratio --notify-every 1. --send-every 20 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_3h_r

# 4(a-h)_r(1-5)
# 4a (5)

python train.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_4a_r

# 4b (5)

python train.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_4b_r

# 4c (5)

python train.py --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_4c_r

# 4d (5)

python train.py --model improved-wgan-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_4d_r

# 4e (5)

python train.py --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_4e_r

# 4f (5)

python train.py --model began-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_4f_r

# 4g (5)

python train.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. -r cvpr_4g_r

# 4h (1)

python train.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. -r cvpr_4h_r

# 5(a-h)_r(1-5)

# 5a
python train.py --model topgan-ae-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_5a_r

# 5b

python train.py --model improved-wgan-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_5b_r

# 5c

python train.py --model began-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments -r cvpr_5c_r

# 5d

python train.py --model topgan-ae-dcgan --dataset stacked-mnist --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 501 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics histogram-normality samples reconstruction mode-count-estimator clas-kl-divergence --notify-every 10. --send-every 2 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --use-gradnorm --gradnorm-alpha 0. --slack-channel cvpr-experiments -r cvpr_5d_r


