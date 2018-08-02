"""
    Generic evaluation script that evaluates a model using a given dataset.
    This code modifies the "TensorFlow-Slim image classification model library",
    Please visit https://github.com/tensorflow/models/tree/master/research/slim
    for more detailed usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import test_models
from utils import *

slim = tf.contrib.slim


tf.app.flags.DEFINE_string('dataset_name', 'coco',
                           'The name of the dataset to load.')

tf.app.flags.DEFINE_string('split_name', 'test',
                           'The name of the train/test split.')

tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string('checkpoint_dir', None,
                           'The directory where the model was written to or an absolute path to a '
                           'checkpoint file.')

tf.app.flags.DEFINE_string('eval_dir', 'results',
                           'Directory where the results are saved to.')

tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1',
                           'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string('model_scope', 'MobileNetV1',
                           'The scope name of the architecture to evaluate.')

tf.app.flags.DEFINE_integer('num_classes', None,
                            'The number of classes.')

tf.app.flags.DEFINE_integer('max_num_batches', None,
                            'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_integer('batch_size', 1,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('feature_size', 512,
                            "joint Feature size.")

tf.app.flags.DEFINE_float('weight_decay', 0.00004,
                          'The weight decay on the model weights.')

tf.app.flags.DEFINE_boolean('is_training', False,
                            'Training or testing.')

tf.app.flags.DEFINE_string('preprocessing_name', None,
                           'The name of the preprocessing to use. If left '
                           'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 1,
                            'The number of threads used to create the batches.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          'The decay to use for the moving average.'
                          'If left as None, then moving averages are not used.')

#########################
#     LSTM Settings     #
#########################

tf.app.flags.DEFINE_integer('embedding_size', 512,
                            """Embedding size.""")

tf.app.flags.DEFINE_integer('num_lstm_units', 512,
                            """Number of LSTM units.""")

tf.app.flags.DEFINE_integer('vocab_size', 12000,
                            """Vocabulary Size.""")

tf.app.flags.DEFINE_float('lstm_dropout_keep_prob', 0.7,
                          """dropout keep prob.""")

#########################

FLAGS = tf.app.flags.FLAGS


def main(_):
    # create folders
    mkdir_if_missing(FLAGS.eval_dir)
    # test
    test_models.evaluate()


if __name__ == '__main__':
    tf.app.run()
