# Toy Datasets

## Experiments with TOPGAN

2a.

*[topgan-ae-mlp-synth-veegan_rcvpr_2a_zdim100_Lgtriplet10none1_Ldloss00zero1_Lgloss00zero1_Ldtriplet10none1_Lgradnorm10none1_Lgmean10none1_Laeloss10skip10_Lgstd10none1]*
`train.py --model topgan-ae-mlp-synth-veegan --dataset synthetic-8ring --batchsize 128 --embedding-dim 2 --checkpoint-every 500. --epoch 10000 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics histogram-normality synthetic-data-vis mode-coverage high-quality-ratio --notify-every 1. --send-every 20 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --use-gradnorm --gradnorm-alpha 0. --distance-metric l2 --slack-channel cvpr-experiments -r cvpr_2a`

[Metrics] Image #39240064 Epoch #981.0016:  histogram-normality: plotted, high-quality-ratio: 0.9181, mode-coverage: 6, synthetic-data-vis: plotted, g_triplet: 0.44954821467399597*1.0, d_loss: 1.3664592504501343*0.0, g_loss: 1.413039207458496*0.0, d_triplet: 0.7662857174873352*1.0, gradnorm: 1.6555033922195435*1.0, g_mean: 0.2986336946487427*1.0, ae_loss: 0.03165677934885025*1.3476957082748413, g_std: 0.6675713658332825*1.0,

2a.2.

*[topgan-ae-mlp-synth-veegan_rcvpr_2a_2_zdim100_Lgtriplet10none1_Ldtriplet10none1_Lgstd10none1_Laeloss10skip10_Ldloss00zero1_Lgradnorm10none1_Lgmean10none1_Lgloss00zero1]*
`train.py --model topgan-ae-mlp-synth-veegan --dataset synthetic-8ring --batchsize 128 --embedding-dim 2 --checkpoint-every 500. --epoch 1000 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics histogram-normality synthetic-data-vis mode-coverage high-quality-ratio --notify-every 1. --send-every 20 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --use-gradnorm --gradnorm-alpha 0. --distance-metric l2 --slack-channel cvpr-experiments -r cvpr_2a_2`

[Metrics] Image #39240064 Epoch #981.0016:  histogram-normality: plotted, mode-coverage: 8, high-quality-ratio.2: 0.9221, synthetic-data-vis: plotted, g_triplet: 0.4772491753101349*1.0, d_triplet: 0.8203639984130859*1.0, g_std: 0.8166853785514832*1.0, ae_loss: 0.03087085299193859*2.3717188835144043, d_loss: 1.386521816253662*0.0, gradnorm: 0.42712128162384033*1.0, g_mean: 0.2836274206638336*1.0, g_loss: 1.3860632181167603*0.0,

2c.

*[topgan-ae-mlp-synth-veegan_rcvpr_2c_zdim100_Lgradnorm10none1_Lgloss00zero1_Laeloss10skip10_Ldloss00zero1_Lgtriplet10none1_Lgmean10none1_Ldtriplet10none1_Lgstd10none1]*
`train.py --model topgan-ae-mlp-synth-veegan --dataset synthetic-25grid --batchsize 128 --embedding-dim 2 --checkpoint-every 500. --epoch 10000 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics histogram-normality synthetic-data-vis mode-coverage high-quality-ratio --notify-every 1. --send-every 20 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --use-gradnorm --gradnorm-alpha 0. --distance-metric l2 --slack-channel cvpr-experiments -r cvpr_2c`

[Metrics] Image #61312512 Epoch #981.000192:  high-quality-ratio: 0.8609, synthetic-data-vis: plotted, histogram-normality: plotted, mode-coverage: 12, gradnorm: 1.408745288848877*1.0, g_loss: 1.3807713985443115*0.0, ae_loss: 1.4212900400161743*0.1763075590133667, d_loss: 1.3948484659194946*0.0, g_triplet: 1.102617621421814*1.0, g_mean: 1.0225642919540405*1.0, d_triplet: 0.38817691802978516*1.0, g_std: 0.9408827424049377*1.0,

2c.2.

*[topgan-ae-mlp-synth-veegan_rcvpr_2c_2_zdim100_Ldloss00zero1_Lgmean10none1_Lgradnorm10none1_Lgtriplet10none1_Ldtriplet10none1_Lgstd10none1_Lgloss00zero1_Laeloss10skip10]*
`train.py --model topgan-ae-mlp-synth-veegan --dataset synthetic-25grid --batchsize 128 --embedding-dim 2 --checkpoint-every 500. --epoch 10000 --controlled-losses ae_loss:1:skip:1 g_triplet:1 d_triplet:1 g_loss:0 d_loss:0 --metrics histogram-normality synthetic-data-vis mode-coverage high-quality-ratio --notify-every 1. --send-every 20 --z-dims 100 --input-noise 0.0 --triplet-margin 1 --lr 1e-4 --use-gradnorm --gradnorm-alpha 0. --distance-metric l2 --slack-channel cvpr-experiments -r cvpr_2c_2`

...waiting

3.a

[Metrics] Epoch #980:  g_loss: 358.27142333984375, d_loss: 1.0723927021026611, mode-coverage: 8, high-quality-ratio: 0.8349358974358975,

3.b

[Metrics] Epoch #980:  g_loss: 361.17498779296875, mode-coverage: 25, d_loss: 0.9520033597946167, high-quality-ratio: 0.6629607371794872,

3.c

[Metrics] Image #39240064 Epoch #981.0016:  mode-coverage: 2, histogram-normality: plotted, high-quality-ratio: 0.9894, synthetic-data-vis: plotted, g_loss: 12.127714157104492*1.0, g_mean: 0.18841075897216797*1.0, g_std: 12.423895835876465*1.0, ae_loss: 0.54290771484375*0.0, d_loss: 0.7710078358650208*1.0, g_triplet: 4.7684407234191895*0.0, d_triplet: 5.122031211853027*0.0,

3.d

[Metrics] Image #61312512 Epoch #981.000192:  synthetic-data-vis: plotted, high-quality-ratio: 0.9765, histogram-normality: plotted, mode-coverage: 1, ae_loss: 7.688571929931641*0.0, g_std: 6.3930344581604*1.0, g_loss: 18.204601287841797*1.0, g_triplet: 2.762660026550293*0.0, g_mean: 0.7686401605606079*1.0, d_loss: 0.24280837178230286*1.0, d_triplet: 2.3658080101013184*0.0,

3.e

[Metrics] Image #39240064 Epoch #981.0016:  mode-coverage: 8, synthetic-data-vis: plotted, high-quality-ratio: 0.8129, gp_loss: 2.134618029003832e-07*1.0, d_loss: 0.0*1.0, g_loss: 0.9999998807907104*1.0,

3.f

[Metrics] Image #61312512 Epoch #981.000192:  mode-coverage: 23, high-quality-ratio: 0.0175, synthetic-data-vis: plotted, d_loss: 0.0*1.0, g_loss: 0.9999998807907104*1.0, gp_loss: 5.385218937448144e-09*1.0,

3.g

[Metrics] Image #39240064 Epoch #981.0016:  mode-coverage: 5, synthetic-data-vis: plotted, high-quality-ratio: 0.1452, gd_ratio: 0.0*1.0, g_loss: 87.3388900756836*1.0, convergence_measure: 87.33953094482422*1.0, d_loss: 0.0028620101511478424*1.0,

3.h

[Metrics] Image #61312512 Epoch #981.000192:  synthetic-data-vis: plotted, mode-coverage: 16, high-quality-ratio: 0.0277, g_loss: 5.863367080688477*1.0, gd_ratio: 0.0*1.0, d_loss: 0.005535909906029701*1.0, convergence_measure: 5.865691184997559*1.0,