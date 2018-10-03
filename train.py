import os
import argparse

from keras import backend as K
import models
from datasets import load_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    parser.add_argument('--gradnorm-alpha', default=0.5, type=float)

    args = parser.parse_args()

    # select gpu and limit resources if applicable
    if 'tensorflow' == K.backend():
        import tensorflow as tf
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

    model.main_loop(dataset, epochs=args.epoch, batchsize=args.batchsize)


if __name__ == '__main__':
    main()
