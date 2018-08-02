"""
    Contains a factory for building various models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import resnet_v1
from nets import mobilenet_v1

slim = tf.contrib.slim

networks_map = {'mobilenet_v1': mobilenet_v1.mobilenet_v1,
                'resnet_v1_152': resnet_v1.resnet_v1_152,
                }

arg_scopes_map = {'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                  'resnet_v1_152': resnet_v1.resnet_arg_scope,
                  }


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
          num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images, scope=None):
        arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, scope=scope)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
