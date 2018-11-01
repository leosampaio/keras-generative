import tensorflow as tf
import itertools
import numpy as np
from tqdm import tqdm
import collections
import matplotlib
matplotlib.use('Agg')
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

from core.notifyier import notify_with_image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

params = {
    'batch_size': 500,
    'latent_dim': 2,
    'eps_dim': 1,
    'input_dim': 254,
    'n_layer_disc': 2,
    'n_hidden_disc': 128,
    'n_layer_gen': 2,
    'n_hidden_gen': 128,
    'n_layer_inf': 2,
    'n_hidden_inf': 128,
}


def create_distribution(batch_size, num_components=25, num_features=2, **kwargs):
    cat = ds.Categorical(tf.zeros(num_components, dtype=np.float32))
    mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                   range(-4, 5, 2))], dtype=np.float32)

    s = 0.05
    sigmas = [np.array([s, s]).astype(np.float32) for i in range(num_components)]
    components = list((ds.MultivariateNormalDiag(mu, sigma)
                       for (mu, sigma) in zip(mus, sigmas)))
    data = ds.Mixture(cat, components)
    return data.sample(batch_size)


def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return st.StochasticTensor(
        ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs))


def normal_mixture(shape, **kwargs):
    return create_distribution(shape[0], 25, shape[1], **kwargs)


def generative_network(batch_size, latent_dim, input_dim, n_layer, n_hidden, eps=1e-6, X=None):
    with tf.variable_scope("generative"):

        z = normal_mixture([batch_size, latent_dim], name="p_z")
        h = slim.fully_connected(z, n_hidden, activation_fn=tf.nn.relu)
        h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
        p = slim.fully_connected(h, input_dim, activation_fn=None)
        x = st.StochasticTensor(ds.Normal(p * tf.ones(input_dim), 1 * tf.ones(input_dim), name="p_x"))
    return [x, z]


def inference_network(x, latent_dim, n_layer, n_hidden, eps_dim):
    eps = standard_normal([x.get_shape().as_list()[0], eps_dim], name="eps").value()
    h = tf.concat([x, eps], 1)
    with tf.variable_scope("inference"):
        h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
        h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
        z = slim.fully_connected(h, latent_dim, activation_fn=None, scope="q_z")
    return z


def data_network(x, z, n_layers=2, n_hidden=128, activation_fn=None):
    h = tf.concat([x, z], 1)
    with tf.variable_scope('discriminator'):
        h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])

tf.reset_default_graph()

x = tf.random_normal([params['batch_size'], params['input_dim']])

p_x, p_z = generative_network(params['batch_size'], params['latent_dim'], params['input_dim'],
                              params['n_layer_gen'], params['n_hidden_gen'])

q_z = inference_network(x, params['latent_dim'], params['n_layer_inf'], params['n_hidden_inf'],
                        params['eps_dim'])


log_d_prior = data_network(p_x, p_z, n_layers=params['n_layer_disc'],
                           n_hidden=params['n_hidden_disc'])
log_d_posterior = data_network(x, q_z, n_layers=params['n_layer_disc'],
                               n_hidden=params['n_hidden_disc'])


disc_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_posterior, labels=tf.ones_like(log_d_posterior)) +
    tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_prior, labels=tf.zeros_like(log_d_prior)))


recon_likelihood_prior = p_x.distribution.log_prob(x)
recon_likelihood = tf.reduce_sum(graph_replace(recon_likelihood_prior, {p_z: q_z}), [1])


gen_loss = tf.reduce_mean(log_d_posterior) - tf.reduce_mean(recon_likelihood)

qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
opt = tf.train.AdamOptimizer(1e-3, beta1=.5)

train_gen_op = opt.minimize(gen_loss, var_list=qvars + pvars)
train_disc_op = opt.minimize(disc_loss, var_list=dvars)

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

fs = []

total_batch = 1000

#  Training cycle
for epoch in tqdm(range(100)):
    xx = np.vstack([sess.run(q_z) for _ in range(5)])
    yy = np.vstack([sess.run(p_z) for _ in range(5)])

    ax = plt.figure(figsize=(5, 5), facecolor='w')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    cmap = cm.tab10
    category_labels = len(xx)*[0]+len(yy)*[1]
    norm = colors.Normalize(vmin=np.min(category_labels), vmax=np.max(category_labels))
    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapped_colors = cmapper.to_rgba(category_labels)
    unique_labels = list(set(category_labels))
    lines = ax.scatter(xx[:, 0]+yy[:, 0], xx[:, 1]+yy[:, 1],
                       color=mapped_colors,
                       label=unique_labels)

    plt.savefig("output/veegan.png", dpi=150, bbox_inches='tight')
    notify_with_image("output/veegan.png", "veegan_2D_grid")

#     Loop over all batches
    for i in range(total_batch):
        _ = sess.run([[gen_loss, disc_loss], train_gen_op, train_disc_op])
