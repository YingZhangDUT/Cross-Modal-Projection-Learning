"""
    convert data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import coco
import flickr30k
import pedes

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("image_dir", None,
                       "image directory.")

tf.flags.DEFINE_string("text_dir", None,
                       "Training captions file.")

tf.flags.DEFINE_string("split_dir", None,
                       "Output data directory.")

tf.flags.DEFINE_string("output_dir", None,
                       "Output data directory.")

tf.flags.DEFINE_string("dataset_name", "flickr30k",
                       "Datset name.")

tf.flags.DEFINE_integer("min_word_count", 3,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")

tf.flags.DEFINE_string("word_counts_output_file", "word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_string("word_to_idx_file", "word_to_idx.pkl",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("train_shards", 1,
                        "Number of shards in training TFRecord files.")

tf.flags.DEFINE_integer("val_shards", 1,
                        "Number of shards in validation TFRecord files.")

tf.flags.DEFINE_integer("test_shards", 1,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")


def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset name with --dataset_name')

    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")

    if FLAGS.dataset_name == 'flickr30k':
        flickr30k.run()
    elif FLAGS.dataset_name == 'coco':
        coco.run()
    elif FLAGS.dataset_name == 'pedes':
        pedes.run()
    else:
        raise ValueError(
            'dataset_name [%s] was not recognized.' % FLAGS.dataset_name)


if __name__ == '__main__':
    tf.app.run()
