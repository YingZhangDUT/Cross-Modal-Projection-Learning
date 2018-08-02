"""
    Provide dataset given split name.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from datasets.sequence_example_decoder import TFSEquenceExampleDecoder

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*'

datasets_map = {'flickr30k': {'train': 145000, 'val': 5070, 'test': 5000},
                'coco': {'train': 414113, 'val': 152634, 'test': 25010},
                'pedes': {'train': 68126, 'val': 6158, 'test': 6156},
                }

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and num_classes',
    'filename': 'The name of the image',
    'split': 'The split of the image, whether for traing/val/test',
    'caption_ids': 'The id of all the caption words',
    'caption': 'Text description of the image',
    'caption_len': 'The number of caption words',
}


def get_split(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      name: the name of the dataset.
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in datasets_map[name]:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    context_keys_to_features = {
        "image/data": tf.FixedLenFeature([], dtype=tf.string),
        "image/format": tf.FixedLenFeature([], dtype=tf.string, default_value='jpeg'),
        "image/label": tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        "image/filename": tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        "image/split": tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }

    sequence_keys_to_features = {
        "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "image/caption": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    items_to_handlers = {
        "image": slim.tfexample_decoder.Image(image_key="image/data", format_key="image/format", channels=3),
        "label": slim.tfexample_decoder.Tensor("image/label"),
        "filename": slim.tfexample_decoder.Tensor("image/filename"),
        "split": slim.tfexample_decoder.Tensor("image/split"),
        "caption_ids": slim.tfexample_decoder.Tensor("image/caption_ids"),
        "caption": slim.tfexample_decoder.Tensor("image/caption"),
        "caption_len": slim.tfexample_decoder.ItemHandlerCallback(
            keys=["image/caption"],
            func=lambda x: tf.size(x["image/caption"]))
    }

    decoder = TFSEquenceExampleDecoder(
        context_keys_to_features, sequence_keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=datasets_map[name][split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    """Given a dataset name and a split_name returns a Dataset.

    Args:
      name: String, the name of the dataset.
      split_name: A train/test/val split name.
      dataset_dir: The directory where the dataset files are stored.
      file_pattern: The file pattern to use for matching the dataset source files.
      reader: The subclass of tf.ReaderBase. If left as `None`, then the default
        reader defined by each dataset is used.

    Returns:
      A `Dataset` class.

    Raises:
      ValueError: If the dataset `name` is unknown.
    """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return get_split(name,
                     split_name,
                     dataset_dir,
                     file_pattern,
                     reader)
