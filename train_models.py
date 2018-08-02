"""
    Generic training script that trains a model using a given dataset.
    This code modifies the "TensorFlow-Slim image classification model library",
    Please visit https://github.com/tensorflow/models/tree/master/research/slim
    for more detailed usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from modules import *
from configuration import *


slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            # print(g)
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def _tower_loss(network_fn, images, labels, input_seqs, input_masks):
    """Calculate the total loss on a single tower running the model."""

    # Get image features, text features, and joint embeddings
    image_features, _ = build_image_features(network_fn, images)
    text_features, _ = build_text_features(input_seqs, input_masks)

    image_embeddings = build_joint_embeddings(image_features, scope='image_joint_embedding')
    text_embeddings = build_joint_embeddings(text_features, scope='text_joint_embedding')

    loss, cmpm_loss, cmpc_loss, i2t_loss, t2i_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    cmpm_loss = tf.cast(cmpm_loss, tf.float32)
    cmpc_loss = tf.cast(cmpc_loss, tf.float32)
    i2t_loss = tf.cast(i2t_loss, tf.float32)
    t2i_loss = tf.cast(t2i_loss, tf.float32)

    # CMPM loss
    if FLAGS.CMPM:
        i2t_loss, t2i_loss, pos_avg_dist, neg_avg_dist = \
            cmpm_loss_compute(text_embeddings, image_embeddings, labels)

        cmpm_loss = i2t_loss + t2i_loss

        tf.summary.scalar('cmpm_i2t_loss', i2t_loss)
        tf.summary.scalar('cmpm_t2i_loss', t2i_loss)
        tf.summary.scalar('cmpm_loss', cmpm_loss)
        tf.summary.scalar('pos_avg_dist', pos_avg_dist)
        tf.summary.scalar('neg_avg_dist', neg_avg_dist)

    # CMPC loss
    if FLAGS.CMPC:
        ipt_loss, tpi_loss, image_precision, text_precision = \
            cmpc_loss_compute(text_embeddings, image_embeddings, labels)

        cmpc_loss = ipt_loss + tpi_loss

        tf.summary.scalar('cmpc_ipt_loss', ipt_loss)
        tf.summary.scalar('cmpc_tpi_loss', tpi_loss)
        tf.summary.scalar('cmpc_loss', cmpc_loss)
        tf.summary.scalar('image_precision', image_precision)
        tf.summary.scalar('text_precision', text_precision)

    loss = cmpc_loss + cmpm_loss

    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + reg_loss, name='total_loss')
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg_loss')
    loss_averages_op = loss_averages.apply([loss] + [total_loss])

    tf.summary.scalar('loss_raw', loss)
    tf.summary.scalar('loss_avg', loss_averages.average(loss))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    return total_loss, cmpm_loss, cmpc_loss, i2t_loss, t2i_loss


def train():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.split_name, FLAGS.dataset_dir)

        ###########################
        # Select the CNN network  #
        ###########################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=None,
            weight_decay=FLAGS.weight_decay,
            is_training=True)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = configure_learning_rate(dataset.num_samples, global_step)
            optimizer = configure_optimizer(learning_rate)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            examples_per_shard = 1024
            min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=min_queue_examples + 3 * FLAGS.batch_size,
                common_queue_min=min_queue_examples)
            [image, label, text_id, text] = provider.get(['image', 'label', 'caption_ids', 'caption'])

            train_image_size = network_fn.default_image_size
            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            # This function splits the text into an input sequence and a target sequence,
            # where the target sequence is the input sequence right-shifted by 1. Input and
            # target sequences are batched and padded up to the maximum length of sequences
            # in the batch. A mask is created to distinguish real words from padding words.
            # Note that the target sequence is used if performing caption generation
            seq_length = tf.shape(text_id)[0]
            input_length = tf.expand_dims(tf.subtract(seq_length, 1), 0)
            input_seq = tf.slice(text_id, [0], input_length)
            target_seq = tf.slice(text_id, [1], input_length)
            input_mask = tf.ones(input_length, dtype=tf.int32)

            images, labels, input_seqs, target_seqs, input_masks, texts, text_ids = tf.train.batch(
                [image, label, input_seq, target_seq, input_mask, text, text_id],
                batch_size=FLAGS.batch_size,
                capacity=2 * FLAGS.num_preprocessing_threads * FLAGS.batch_size,
                dynamic_pad=True,
                name="batch_and_pad")

            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels, input_seqs, target_seqs, input_masks, texts, text_ids],
                capacity=16 * deploy_config.num_clones,
                num_threads=FLAGS.num_preprocessing_threads,
                dynamic_pad=True,
                name="perfetch_and_pad")

            images, labels, input_seqs, target_seqs, input_masks, texts, text_ids = batch_queue.dequeue()

        images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
        labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)
        input_seqs_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=input_seqs)
        target_seqs_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=target_seqs)
        input_masks_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=input_masks)
        texts_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=texts)
        text_ids_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=text_ids)

        tower_grads = []
        for k in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.name_scope('tower_%d' % k) as scope:
                    with tf.variable_scope(tf.get_variable_scope()):

                        loss, cmpm_loss, cmpc_loss, i2t_loss, t2i_loss = \
                            _tower_loss(network_fn, images_splits[k], labels_splits[k],
                                        input_seqs_splits[k], input_masks_splits[k])

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

                        # Variables to train.
                        variables_to_train = get_variables_to_train()
                        grads = optimizer.compute_gradients(loss, var_list=variables_to_train)

                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = _average_gradients(tower_grads)

        # Add a summary to track the learning rate and precision.
        summaries.append(tf.summary.scalar('learning_rate', learning_rate))

        # Add histograms for histogram and trainable variables.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Apply the gradients to adjust the shared variables.
        grad_updates = optimizer.apply_gradients(grads, global_step=global_step)
        update_ops.append(grad_updates)

        # Group all updates to into a single train op.
        train_op = tf.group(*update_ops)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=config)
        sess.run(init)

        ck_global_step = get_init_fn(sess)
        print_train_info()

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(
            os.path.join(FLAGS.log_dir),
            graph=sess.graph)

        num_steps_per_epoch = int(dataset.num_samples / FLAGS.batch_size)
        max_number_of_steps = FLAGS.num_epochs * num_steps_per_epoch

        for step in xrange(max_number_of_steps):
            step += int(ck_global_step)
            # check the training data
            # simages, slabels, sinput_seqs, starget_seqs, sinput_masks, stexts, stext_ids = \
            # sess.run([images_splits[0], labels_splits[0], input_seqs_splits[0], target_seqs_splits[0],
            #           input_masks_splits[0], texts_splits[0], text_ids_splits[0]])
            # save_images(simages[:8], [1, 8], './{}/{:05d}.png'.format(FLAGS.train_samples_dir, step))
            # import pdb
            # pdb.set_trace()

            _, total_loss_value, cmpm_loss_value, cmpc_loss_value, i2t_loss_value, t2i_loss_value = \
                sess.run([train_op, loss, cmpm_loss, cmpc_loss, i2t_loss, t2i_loss])

            assert not np.isnan(cmpm_loss_value), 'Model diverged with cmpm_loss = NaN'
            assert not np.isnan(cmpc_loss_value), 'Model diverged with cmpc_loss = NaN'
            assert not np.isnan(total_loss_value), 'Model diverged with total_loss = NaN'

            if step % 10 == 0:
                format_str = ('%s: step %d, cmpm_loss = %.2f, cmpc_loss = %.2f, '
                              'i2t_loss = %.2f, t2i_loss = %.2f')
                print(format_str % (FLAGS.dataset_name, step, cmpm_loss_value, cmpc_loss_value,
                                    i2t_loss_value, t2i_loss_value))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.ckpt_steps == 0 or (step + 1) == max_number_of_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
