"""
    Generic evaluation script that evaluates a model using a given dataset.
    This code modifies the "TensorFlow-Slim image classification model library",
    Please visit https://github.com/tensorflow/models/tree/master/research/slim
    for more detailed usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import json
import itertools
import collections
from datetime import datetime
import numpy as np
import os.path
import sys
import scipy.io as sio
from collections import defaultdict

import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from utils import *
from modules import *

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


def merge_image_features(all_image_features, all_labels, all_filenames):
    """
        Merge features from the same image.
    """
    merged = list(itertools.chain.from_iterable(all_filenames))
    with open(os.path.join(FLAGS.eval_dir, 'filenames.txt'), 'w') as myfile:
        myfile.write('\n'.join(merged))
    myfile.close()
    dup_list = sorted(list_duplicates(merged), key=lambda x: x[1])
    num_images = len(dup_list)
    avg_image_features = []
    avg_labels = []
    for i in range(num_images):
        single_image_features = [all_image_features[k] for k in dup_list[i][1]]
        single_labels = [all_labels[k] for k in dup_list[i][1]]

        avg_image_features.append(sum(single_image_features)/len(single_image_features))
        avg_labels.append(sum(single_labels) / len(single_labels))

    return avg_image_features, avg_labels


def save_array(features, labels, name):
    np_features = np.asarray(features)
    np_features = np.reshape(np_features, [len(features), -1])
    np_labels = np.asarray(labels)
    np_labels = np.reshape(np_labels, len(labels))

    # save .npy
    feature_filename = "%s/%s_%s_features.npy" % (FLAGS.eval_dir, FLAGS.split_name, name)
    np.save(feature_filename, np_features)
    label_filename = "%s/%s_%s_labels.npy" % (FLAGS.eval_dir, FLAGS.split_name, name)
    np.save(label_filename, np_labels)

    # save .mat
    feature_filename = "%s/%s_%s_features.mat" % (FLAGS.eval_dir, FLAGS.split_name, name)
    sio.savemat(feature_filename, {'feature': np_features})
    label_filename = "%s/%s_%s_labels.mat" % (FLAGS.eval_dir, FLAGS.split_name, name)
    sio.savemat(label_filename, {'label': np_labels})


def _extract_once(image_embeddings, caption_embeddings, labels, filenames, images, num_examples, saver):
    """Extract Features.
    """
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                    saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt_name))
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Succesfully loaded model from %s at step=%s.' %
                      (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return

            if FLAGS.max_num_batches:
                    num_batches = FLAGS.max_num_batches
            else:
                    num_batches = int(math.ceil(num_examples / float(FLAGS.batch_size)))
            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                step = 0
                all_image_features, all_caption_features, all_labels, all_filenames = [], [], [], []
                print("Current Path: %s" % os.getcwd())
                print('%s: starting extracting features on (%s).' % (datetime.now(), FLAGS.split_name))
                while step < num_batches and not coord.should_stop():
                    step += 1
                    sys.stdout.write('\r>> Extracting %s image %d/%d [%d examples]' %
                                     (FLAGS.split_name, step, num_batches, num_examples))
                    sys.stdout.flush()
                    eval_image_embeddings, eval_caption_embeddings, eval_labels, eval_filenames = sess.run(
                        [image_embeddings, caption_embeddings, labels, filenames])
                    eval_image_features = np.reshape(eval_image_embeddings, [eval_image_embeddings.shape[0], -1])
                    eval_caption_features = np.reshape(eval_caption_embeddings, [eval_caption_embeddings.shape[0], -1])
                    all_image_features.append(eval_image_features)
                    all_caption_features.append(eval_caption_features)
                    all_labels.append(eval_labels)
                    all_filenames.append(eval_filenames)

                #  save features and labels
                avg_image_features, avg_labels = merge_image_features(all_image_features, all_labels, all_filenames)

                save_array(avg_image_features, avg_labels, 'image')
                save_array(all_caption_features, all_labels, 'caption')
                print("Done!\n")

            except Exception as e:
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=None,
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label, caption_id, caption, filename] = \
            provider.get(['image', 'label', 'caption_ids', 'caption', 'filename'])

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        caption_length = tf.shape(caption_id)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)
        input_seq = tf.slice(caption_id, [0], input_length)
        target_seq = tf.slice(caption_id, [1], input_length)
        input_mask = tf.ones(input_length, dtype=tf.int32)

        images, labels, input_seqs, target_seqs, input_masks, captions, caption_ids, filenames = \
            tf.train.batch([image, label, input_seq, target_seq, input_mask, caption, caption_id, filename],
                           batch_size=FLAGS.batch_size,
                           num_threads=FLAGS.num_preprocessing_threads,
                           capacity=5 * FLAGS.batch_size,
                           dynamic_pad=True)

        ####################
        # Define the model #
        ####################
        image_features, _ = build_image_features(network_fn, images)
        text_features, _ = build_text_features(input_seqs, input_masks)

        image_embeddings = build_joint_embeddings(image_features, scope='image_joint_embedding')
        text_embeddings = build_joint_embeddings(text_features, scope='text_joint_embedding')

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)
        _extract_once(image_embeddings, text_embeddings, labels, filenames, images, dataset.num_samples, saver)
