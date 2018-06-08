
'''
Created by: https://github.com/altosaar/variational-autoencoder/
Edited by : Eunsong Kang
Tensorflow : TF 1.1
'''

import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as distributions
import scipy.io
import time
import matplotlib.pyplot as plt
tf.set_random_seed(42)
np.random.seed(42)

"Data Load"
data = scipy.io.loadmat('../datapath/data_file')["variable"]
# data.shape = (subjects, timescans,  ROIs)
arr = data.reshape(-1, data.shape[-1])
# arr.shape = (samples, ROIs)

# Input information
input_dim = arr.shape[1]
samples_for_data = arr.shape[0]

sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor
flags = tf.app.flags

# Must set before training
flags.DEFINE_string('logdir', '../upload_test', 'Directory for logs')
flags.DEFINE_integer('latent_dim', 10, 'Latent dimensionality of model')
flags.DEFINE_integer('batch_size', 10, 'Minibatch size')
flags.DEFINE_integer('print_every', 100, 'Print every n')
flags.DEFINE_integer('hidden_size', 20, 'Hidden size for neural networks')
flags.DEFINE_integer('n_iterations', 100, 'number of iterations')
flags.DEFINE_float('param', 0.001, 'L1 regularization')

flags.DEFINE_string('train', True, 'Train or test')
flags.DEFINE_string('hidden_layer', False, 'Hidden layer or not')

FLAGS = flags.FLAGS


def inference_network(x, latent_dim, hidden_size, layers, trainornot):
    """Construct an inference network parametrizing a Gaussian.

    Args:
      x: A batch of MCI data subjects.
      latent_dim: The latent dimensionality.
      hidden_size: The size of the neural net hidden layers.

    Returns:
      mu: Mean parameters for the variational family Normal
      sigma: Standard deviation parameters for the variational family Normal
    """
    if layers == True:
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            net = slim.flatten(x)
            net = slim.fully_connected(net, hidden_size, trainable=trainornot)
            net = slim.fully_connected(net, hidden_size, trainable=trainornot)
            gaussian_params = slim.fully_connected(net, latent_dim * 2, activation_fn=None, trainable=trainornot)

            # The mean parameter is unconstrained
            mu = gaussian_params[:, :latent_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, latent_dim:])

    elif layers == False:

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            net = slim.flatten(x)
            gaussian_params = slim.fully_connected(net, latent_dim * 2,
                                                   activation_fn=None,
                                                   trainable=trainornot,
                                                    weights_regularizer=slim.l1_regularizer(FLAGS.param))

            # The mean parameter is unconstrained
            mu = gaussian_params[:, :latent_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, latent_dim:])
    else:
        raise("Only True and False in the layer parameter")

    return mu, sigma


def generative_network(z, hidden_size, layers, trainornot):
    """Build a generative network parametrizing the likelihood of the data

    Args:
      z: Samples of latent variables
      hidden_size: Size of the hidden state of the neural net

    Returns:
      mu: Mean parameters for the variational family Normal
      sigma: Standard deviation parameters for the variational family Normal
    """
    if layers == True:
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            net = slim.fully_connected(z, hidden_size, trainable=trainornot)
            net = slim.fully_connected(net, hidden_size, trainable=trainornot)
            gaussian_params = slim.fully_connected(net, input_dim * 2,
                                                   activation_fn=None,
                                                   trainable=trainornot)
            mu = gaussian_params[:, :input_dim]
            sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, input_dim:])
            mu = tf.reshape(mu, [-1, 116])
            sigma = tf.reshape(sigma, [-1, 116])

    elif layers == False:
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            gaussian_params = slim.fully_connected(z, input_dim * 2,
                                                   activation_fn=None,
                                                   trainable=trainornot,
                                                   weights_regularizer=slim.l1_regularizer(FLAGS.param))

            mu = gaussian_params[:, :input_dim]
            sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, input_dim:])
            mu = tf.reshape(mu, [-1, 116])
            sigma = tf.reshape(sigma, [-1, 116])
    else:
        raise ("...")


    return mu, sigma


def train():

    """ Input placeholders"""
    with tf.name_scope('ROIs'):
        x = tf.placeholder(tf.float32, [None, input_dim])

    with tf.variable_scope('variational'):
        q_mu, q_sigma = inference_network(x=x,
                                          latent_dim=FLAGS.latent_dim,
                                          hidden_size=FLAGS.hidden_size,
                                          layers=FLAGS.hidden_layer,
                                          trainornot=FLAGS.train)

        p_z = distributions.MultivariateNormalDiag(loc=np.zeros(FLAGS.latent_dim, dtype=np.float32),
                                                   scale_diag=np.ones(FLAGS.latent_dim, dtype=np.float32))



        with st.value_type(st.SampleValue()):
            # The variational distribution is a Normal with mean and standard
            # deviation given by the inference network
            q_z = st.StochasticTensor(distributions.MultivariateNormalDiag(loc=q_mu, scale_diag=q_sigma))

    with tf.variable_scope('generative'):
        # The likelihood is Gaussian-distributed with parameter mu given by the generative network
        p_x_given_z_mu, p_x_given_z_sigma = generative_network(z=q_z,
                                                               hidden_size=FLAGS.hidden_size,
                                                               layers=FLAGS.hidden_layer,
                                                               trainornot=FLAGS.train)

        p_x_given_z = distributions.MultivariateNormalDiag(loc=p_x_given_z_mu, scale_diag=p_x_given_z_sigma)


    with tf.variable_scope('generative', reuse=True):
        z_input = tf.placeholder(tf.float32, [None, FLAGS.latent_dim])
        p_x_given_z_mu_2, p_x_given_z_sigma_2 = generative_network(z=z_input,
                                                                   hidden_size=FLAGS.hidden_size,
                                                                   layers=FLAGS.hidden_layer,
                                                                   trainornot=FLAGS.train)
        p_x_given_z_2 = distributions.MultivariateNormalDiag(loc=p_x_given_z_mu_2, scale_diag=p_x_given_z_sigma)
        prior_predictive = p_x_given_z_2.copy()

        prior_predictive_inp_sample = prior_predictive.sample()


    # Build the evidence lower bound (ELBO) or the negative loss

    # For no regularization term
    # kl = distributions.kl(q_z.distribution, p_z)
    # expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), -1)
    # elbo = tf.reduce_sum(expected_log_likelihood - kl, 0)

    kl = distributions.kl(q_z.distribution, p_z)
    reg_variables = slim.losses.get_regularization_losses()
    reg_variables_sum = tf.reduce_sum(reg_variables)
    expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), -1)
    reg_expected_log_likelihood = expected_log_likelihood + reg_variables_sum
    elbo = tf.reduce_sum(reg_expected_log_likelihood - kl, 0)

    # Optimization
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(-elbo)
    tf.summary.scalar("ELBO", elbo)

    # Merge all the summaries
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Run training
    sess = tf.InteractiveSession()
    sess.run(init_op)

    print('Saving TensorBoard summaries and images to: %s' % FLAGS.logdir)
    train_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    randidx = np.random.permutation(np.arange(samples_for_data, dtype=np.uint32))
    saver = tf.train.Saver()

    #Batchsize
    cur_epoch = 0
    for i in range((FLAGS.n_iterations * samples_for_data)// FLAGS.batch_size):
        offset = (i) % (samples_for_data // FLAGS.batch_size)
        np_x = arr[randidx[offset * FLAGS.batch_size:(offset + 1) * FLAGS.batch_size]].reshape(-1, input_dim).copy()
        sess.run(train_op, {x: np_x})

        t0 = time.time()
        if i % FLAGS.print_every == 0:
            np_elbo, summary_str = sess.run([elbo, summary_op], {x: np_x})
            train_writer.add_summary(summary_str, i)
            print('Iteration: {0:d} ELBO: {1:.3f} Examples/s: {2:.3e}'.format(i, np_elbo / FLAGS.batch_size, FLAGS.batch_size * FLAGS.print_every / (time.time() - t0)))

        if cur_epoch != int((i * FLAGS.batch_size)/ samples_for_data):
            # print("Saved in path", saver.save(sess, os.path.join(FLAGS.logdir, "%02d.ckpt" % (cur_epoch))))
            randidx = np.random.permutation(samples_for_data)
        cur_epoch = int((i * FLAGS.batch_size) / samples_for_data)
        t0 = time.time()
    saver.save(sess, os.path.join(FLAGS.logdir, 'savedmodel_final.ckpt'))



def main(_):
    if ~tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.MakeDirs(FLAGS.logdir)
    train()


if __name__ == '__main__':
    tf.app.run()