import tensorflow as tf
import numpy as np
import os
import math
from time import gmtime, strftime
import os
import argparse

from keras import backend as K
from keras import optimizers
from keras import layers
import models
from datasets import load_dataset
from models.utils import plot_metrics

from core.notifyier import notify_with_message, notify_with_image


def get_inference_via_optimization(model, data):

    total_iters = 10000

    input_x = layers.Input(shape=model.input_shape)
    latent_z = K.variable(np.random.normal(size=(model.batchsize, model.z_dims)))
    x_hat = model.f_Gx(latent_z)
    mse = K.mean(K.square(x_hat - input_x))
    opt = optimizers.RMSprop(lr=1e-2)
    updates = opt.get_updates([latent_z], [], mse)

    ivom_trainer = K.function([input_x], [x_hat, mse], updates=updates)

    random_real_data, _ = data.get_random_fixed_batch(n=model.batchsize)

    for b in range(total_iters):
        x_hat, loss = ivom_trainer([random_real_data])
        print('[Metrics] Training Classifier for Mode Estimation... B{}/{}: loss = {}'.format(b, total_iters, loss), end='\r')
    message = '[Metrics] Training Classifier for Mode Estimation... B{}/{}: loss = {}'.format(b, total_iters, loss)
    print(message)

    imgs = np.zeros((len(random_real_data)*2,) + random_real_data.shape[1:])
    imgs[0::2] = random_real_data
    imgs[1::2] = x_hat
    imgs = np.clip(imgs, 0., 1.)

    figname = 'output/eval_cifar_{}.png'.format(model.experiment_id)
    plot_metrics(figname,
                 metrics_list=[imgs],
                 iterations_list=[1],
                 metric_names=['IvOM'],
                 types=['image-grid'],
                 legend=True,
                 figsize=8,
                 wspace=0.15)

    notify_with_message(message, model.experiment_id)
    notify_with_image(figname, model.experiment_id)

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')
    parser.add_argument('--z-dims', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test-mode', action='store_true')
    parser.add_argument('--is-conditional', action='store_true')
    parser.add_argument('--aux-classifier', action='store_true')
    parser.add_argument('--share-decoder-and-generator', action='store_true')
    parser.add_argument('--label-smoothing', default=0.0, type=float)
    parser.add_argument('--input-noise', default=0.0, type=float)
    parser.add_argument('--run-id', '-r', required=True)
    parser.add_argument('--checkpoint-every', default='1.', type=str)
    parser.add_argument('--notify-every', default='1.', type=str)
    parser.add_argument('--send-every', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dis-loss-control', default=1., type=float)
    parser.add_argument('--triplet-weight', default=1., type=float)
    parser.add_argument('--embedding-dim', default=256, type=int)
    parser.add_argument('--isolate-d-classifier', action='store_true')
    parser.add_argument('--controlled-losses', type=str, nargs='+',
                        help="strings in format loss_name:weight:control_type:pivot_epoch")
    parser.add_argument('--metrics', type=str, nargs='+',
                        help="selection of metrics you want to calculate")
    parser.add_argument('--wgan-n-critic', default=5, type=int)
    parser.add_argument('--began-gamma', default=0.5, type=float)
    parser.add_argument('--triplet-margin', default=1., type=float)
    parser.add_argument('--n-filters-factor', default=32, type=int)
    parser.add_argument('--use-began-equilibrium', action='store_true')
    parser.add_argument('--use-alignment-layer', action='store_true')
    parser.add_argument('--use-simplified-triplet', action='store_true')
    parser.add_argument('--data-folder', default='datasets/files')
    parser.add_argument('--use-magan-equilibrium', action='store_true')
    parser.add_argument('--topgan-enforce-std-dev', action='store_true')
    parser.add_argument('--topgan-use-data-trilet-regularization', action='store_true')
    parser.add_argument('--use-began-loss', action='store_true')
    parser.add_argument('--use-gradnorm', action='store_true')
    parser.add_argument('--use-sigmoid-triplet', action='store_true')
    parser.add_argument('--online-mining', default=None, type=str)
    parser.add_argument('--online-mining-ratio', default=4, type=int)
    parser.add_argument('--gradnorm-alpha', default=0.5, type=float)
    parser.add_argument('--distance-metric', default='l2', type=str)
    parser.add_argument('--slack-channel', type=str, default="random")
    parser.add_argument('--use-quadruplet', action='store_true')
    parser.add_argument('--generator-mining', action='store_true')

    args = parser.parse_args()

    # select gpu and limit resources if applicable
    if 'tensorflow' == K.backend():
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(args.gpu)
        set_session(tf.Session(config=config))

    # make output directory if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # load datasets
    dataset = load_dataset(args.dataset)

    model = models.get_model_by_name(args.model)(
        input_shape=dataset.shape[1:],
        **vars(args)
    )

    if args.resume:
        model.load_model(args.resume)

    get_inference_via_optimization(model, dataset)

if __name__ == '__main__':
    main()
