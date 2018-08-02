"""
    This code modifies the "TensorFlow-Slim image classification model library",
    Please visit https://github.com/tensorflow/models/tree/master/research/slim
    for more detailed usage.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import *

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def get_init_fn(sess):
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    ck_global_step = 0

    if FLAGS.restore_pretrain:
        ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
        if ckpt and ckpt.model_checkpoint_path:
            variables_to_restore = get_variables_to_restore()
            restorer = tf.train.Saver(variables_to_restore)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            restorer.restore(sess, os.path.join(FLAGS.restore_path, ckpt_name))
            ck_pretrain_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' %
                  (ckpt.model_checkpoint_path, ck_pretrain_step))
        else:
            print('No checkpoint Found, Please provide pretrained checkpoint')
            return

    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint with relative path.
            variables_to_restore = tf.trainable_variables()
            restorer = tf.train.Saver(variables_to_restore)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            restorer.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt_name))
            ck_global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' %
                  (ckpt.model_checkpoint_path, ck_global_step))
        else:
            print('No checkpoint Found, Start training from step= %d' % ck_global_step)

    return ck_global_step


def get_variables_to_restore():
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    scopes = [FLAGS.restore_scopes]  #
    scope_variables_to_restore = []
    var_list = tf.trainable_variables()
    for restore_scope in scopes:
        variables = [var for var in var_list if restore_scope in var.name]
        scope_variables_to_restore.extend(variables)

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in scope_variables_to_restore:    # slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return variables_to_restore


def get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    var_list = tf.trainable_variables()
    exclusions = []
    if FLAGS.trainable_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.trainable_exclude_scopes.split(',')]
    variables_to_train = []
    for var in var_list:    # slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_train.append(var)
    return variables_to_train


def configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def add_variables_summaries(learning_rate):
    summaries = []
    for variable in slim.get_model_variables():
        summaries.append(tf.summary.histogram(variable.op.name, variable))
    summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
    return summaries


def activation_summaries(end_points):
    for end_point in end_points:
        x = end_points[end_point]
        tf.summary.histogram('activations/' + end_point, x)
        tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x))


def print_train_info():
    # print main information for training
    current_path = os.getcwd()
    print("Dataset: %s" % FLAGS.dataset_name)
    print('Classes: %d' % FLAGS.num_classes)
    print("Data Path: %s" % FLAGS.dataset_dir)
    print("Current Path: %s" % current_path)
    print("Train Epochs: %d" % FLAGS.num_epochs)
    print("Optimizer: %s" % FLAGS.optimizer)
    print("Learning Rate: %.5f" % FLAGS.learning_rate)
    print("Batch Size: %d" % FLAGS.batch_size)
    print("LSTM Dropout Keep Prob: %.2f" % FLAGS.lstm_dropout_keep_prob)


