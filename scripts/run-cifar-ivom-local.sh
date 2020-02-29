# 4. Computing IvOM

# 4a (1)

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_dcgan_rcvpr_4a_r1_cifar10/weights/pi_24750080 -r cvpr_4a_ivom_r1

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_dcgan_rcvpr_4a_r2_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4a_ivom_r2

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_dcgan_rcvpr_4a_r3_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4a_ivom_r3

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_dcgan_rcvpr_4a_r4_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4a_ivom_r4

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_dcgan_rcvpr_4a_r5_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4a_ivom_r5

# 4b (5)

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_small2_rcvpr_4b_r1_cifar10/weights/pi_24750080 -r cvpr_4b_ivom_r1

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_small2_rcvpr_4b_r2_cifar10/weights/pi_24750080 -r cvpr_4b_ivom_r2

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_small2_rcvpr_4b_r3_cifar10/weights/pi_24750080 -r cvpr_4b_ivom_r3

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_small2_rcvpr_4b_r4_cifar10/weights/pi_24750080 -r cvpr_4b_ivom_r5

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/topgan_ae_small2_rcvpr_4b_r4_cifar10/weights/pi_24750080 -r cvpr_4b_ivom_r4

# 4c (1)

python eval-cifar-ivom.py --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_dcgan_rcvpr_4c_r1_cifar10/weights/pi_24750080 -r cvpr_4c_ivom_r1

python eval-cifar-ivom.py --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_dcgan_rcvpr_4c_r2_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4c_ivom_r2

python eval-cifar-ivom.py --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_dcgan_rcvpr_4c_r3_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4c_ivom_r3

python eval-cifar-ivom.py --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_dcgan_rcvpr_4c_r4_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4c_ivom_r4

python eval-cifar-ivom.py --model improved-wgan-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_dcgan_rcvpr_4c_r5_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4c_ivom_r5

# 4d (5)

python eval-cifar-ivom.py --model improved-wgan-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_small2_rcvpr_4d_r1_cifar10/weights/pi_24750080 -r cvpr_4d_ivom_r1

python eval-cifar-ivom.py --model improved-wgan-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_small2_rcvpr_4d_r2_cifar10/weights/pi_24750080 -r cvpr_4d_ivom_r2

python eval-cifar-ivom.py --model improved-wgan-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_small2_rcvpr_4d_r3_cifar10/weights/pi_24750080 -r cvpr_4d_ivom_r4

python eval-cifar-ivom.py --model improved-wgan-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_small2_rcvpr_4d_r5_cifar10/weights/pi_24750080 -r cvpr_4d_ivom_r5

python eval-cifar-ivom.py --model improved-wgan-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/improved_wgan_small2_rcvpr_4d_r3_cifar10/weights/pi_24750080 -r cvpr_4d_ivom_r3

# 4e (1)

python eval-cifar-ivom.py --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_dcgan_rcvpr_4e_r1_cifar10/weights/pi_24750080 -r cvpr_4e_ivom_r1


python eval-cifar-ivom.py --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_dcgan_rcvpr_4e_r2_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4e_ivom_r2

python eval-cifar-ivom.py --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_dcgan_rcvpr_4e_r3_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4e_ivom_r3

python eval-cifar-ivom.py --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_dcgan_rcvpr_4e_r4_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4e_ivom_r4

python eval-cifar-ivom.py --model began-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_dcgan_rcvpr_4e_r5_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4e_ivom_r5

# 4f (5)

python eval-cifar-ivom.py --model began-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_small2_rcvpr_4f_r1_cifar10/weights/pi_24750080 -r cvpr_4f_ivom_r1

python eval-cifar-ivom.py --model began-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_small2_rcvpr_4f_r2_cifar10/weights/pi_24750080 -r cvpr_4f_ivom_r2

python eval-cifar-ivom.py --model began-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_small2_rcvpr_4f_r3_cifar10/weights/pi_24750080 -r cvpr_4f_ivom_r3

python eval-cifar-ivom.py --model began-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_small2_rcvpr_4f_r4_cifar10/weights/pi_24750080 -r cvpr_4f_ivom_r4

python eval-cifar-ivom.py --model began-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:0 g_triplet:0 d_triplet:0 g_loss:1 d_loss:1 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --resume output/began_small2_rcvpr_4f_r5_cifar10/weights/pi_24750080 -r cvpr_4f_ivom_r5

# 4g (5)

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_dcgan_rcvpr_4g_r1_cifar10/weights/pi_24750080 -r cvpr_4g_ivom_r1

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_dcgan_rcvpr_4g_r2_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4g_ivom_r2

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_dcgan_rcvpr_4g_r3_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4g_ivom_r3

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_dcgan_rcvpr_4g_r4_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4g_ivom_r4

python eval-cifar-ivom.py --model topgan-ae-dcgan --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_dcgan_rcvpr_4g_r5_cifar10/weights/pi_24750080/pi_24750080 -r cvpr_4g_ivom_r5


# 4h (1)

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_small2_rcvpr_4h_r1_cifar10/weights/pi_24750080 -r cvpr_4h_ivom_r1

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_small2_rcvpr_4h_r2_cifar10/weights/pi_24750080 -r cvpr_4h_ivom_r2

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_small2_rcvpr_4h_r3_cifar10/weights/pi_24750080 -r cvpr_4h_ivom_r3

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_small2_rcvpr_4h_r4_cifar10/weights/pi_24750080 -r cvpr_4h_ivom_r4

python eval-cifar-ivom.py --model topgan-ae-small2 --dataset cifar10 --batchsize 128 --embedding-dim 512 --checkpoint-every 99. --epoch 500 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics samples reconstruction --notify-every 5. --send-every 5 --z-dims 100 --input-noise 0.00 --triplet-margin 1 --lr 1e-4 --slack-channel cvpr-experiments --use-gradnorm --gradnorm-alpha 0. --resume output/topgan_ae_small2_rcvpr_4h_r5_cifar10/weights/pi_24750080 -r cvpr_4h_ivom_r5


mkdir -p output/topgan_ae_dcgan_rcvpr_4a_r2_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:output/topgan_ae_dcgan_rcvpr_4a_r2_cifar10/weights/pi_24750080/pi_24750080 output/topgan_ae_dcgan_rcvpr_4a_r2_cifar10/weights/pi_24750080/
mkdir -p output/topgan_ae_dcgan_rcvpr_4a_r3_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4a_r3_cifar10/weights/pi_24750080 output/topgan_ae_dcgan_rcvpr_4a_r3_cifar10/weights/pi_24750080/
mkdir -p output/topgan_ae_dcgan_rcvpr_4a_r4_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4a_r4_cifar10/weights/pi_24750080 output/topgan_ae_dcgan_rcvpr_4a_r4_cifar10/weights/pi_24750080/
mkdir -p output/topgan_ae_dcgan_rcvpr_4a_r5_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4a_r5_cifar10/weights/pi_24750080 output/topgan_ae_dcgan_rcvpr_4a_r5_cifar10/weights/pi_24750080/
mkdir -p output/improved_wgan_dcgan_rcvpr_4c_r2_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/improved_wgan_dcgan_rcvpr_4c_r2_cifar10/weights/pi_24750080 output/improved_wgan_dcgan_rcvpr_4c_r2_cifar10/weights/pi_24750080/
mkdir -p output/improved_wgan_dcgan_rcvpr_4c_r3_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/improved_wgan_dcgan_rcvpr_4c_r3_cifar10/weights/pi_24750080 output/improved_wgan_dcgan_rcvpr_4c_r3_cifar10/weights/pi_24750080/
mkdir -p output/improved_wgan_dcgan_rcvpr_4c_r4_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/improved_wgan_dcgan_rcvpr_4c_r4_cifar10/weights/pi_24750080 output/improved_wgan_dcgan_rcvpr_4c_r4_cifar10/weights/pi_24750080/
mkdir -p output/improved_wgan_dcgan_rcvpr_4c_r5_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/improved_wgan_dcgan_rcvpr_4c_r5_cifar10/weights/pi_24750080 output/improved_wgan_dcgan_rcvpr_4c_r5_cifar10/weights/pi_24750080/
mkdir -p output/began_dcgan_rcvpr_4e_r2_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/began_dcgan_rcvpr_4e_r2_cifar10/weights/pi_24750080 output/began_dcgan_rcvpr_4e_r2_cifar10/weights/pi_24750080/
mkdir -p output/began_dcgan_rcvpr_4e_r3_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/began_dcgan_rcvpr_4e_r3_cifar10/weights/pi_24750080 output/began_dcgan_rcvpr_4e_r3_cifar10/weights/pi_24750080/
mkdir -p output/began_dcgan_rcvpr_4e_r4_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/began_dcgan_rcvpr_4e_r4_cifar10/weights/pi_24750080 output/began_dcgan_rcvpr_4e_r4_cifar10/weights/pi_24750080/
mkdir -p output/began_dcgan_rcvpr_4e_r5_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/began_dcgan_rcvpr_4e_r5_cifar10/weights/pi_24750080 output/began_dcgan_rcvpr_4e_r5_cifar10/weights/pi_24750080/
mkdir -p output/topgan_ae_dcgan_rcvpr_4g_r2_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4g_r2_cifar10/weights/pi_24750080 output/topgan_ae_dcgan_rcvpr_4g_r2_cifar10/weights/pi_24750080/
mkdir -p output/topgan_ae_dcgan_rcvpr_4g_r3_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4g_r3_cifar10/weights/pi_24750080 output/topgan_ae_dcgan_rcvpr_4g_r3_cifar10/weights/pi_24750080/
mkdir -p output/topgan_ae_dcgan_rcvpr_4g_r4_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4g_r4_cifar10/weights/pi_24750080 output/topgan_ae_dcgan_rcvpr_4g_r4_cifar10/weights/pi_24750080/
mkdir -p output/topgan_ae_dcgan_rcvpr_4g_r5_cifar10/weights/pi_24750080
scp -r euler.cemeai.icmc.usp.br:/lustre/lribeiro/output/topgan_ae_dcgan_rcvpr_4g_r5_cifar10/weights/pi_24750080 output/topgan_ae_dcgan_rcvpr_4g_r5_cifar10/weights/pi_24750080/