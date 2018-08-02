"""
    Contains utilities for converting datasets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import nltk.tokenize
from collections import Counter
from collections import namedtuple
from datetime import datetime
import threading
import string
import numpy as np
import sys
from scipy import misc
import random

from utils import *


ImageMetadata = namedtuple("ImageMetadata",
                           ["label", "filename", "captions", "split"])


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.

        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


class ImageEncoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG encoding.
        self._image = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(self._image)

    def encode_jpeg(self, image):
        encode_jpeg = self._sess.run(self._encode_jpeg,
                                     feed_dict={self._image: image})
        return encode_jpeg


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(image, encoder, decoder, vocab):
    """Builds a SequenceExample proto for an image-caption pair.

    Args:
      image: An ImageMetadata object.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.

    Returns:
      A SequenceExample proto.
    """
    # with tf.gfile.FastGFile(image.filename, "r") as f:
    #     encoded_image = f.read()

    # in case the dataset has gray images
    image_data = misc.imread(image.filename) 
    if len(image_data.shape) == 2:
        print(image.filename)
        image_data = np.dstack((image_data, image_data, image_data))
    # print(image.filename)
    encoded_image = encoder.encode_jpeg(image_data)

    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return

    context = tf.train.Features(feature={
        "image/data": _bytes_feature(encoded_image),
        "image/label": _int64_feature(image.label),
        "image/filename": _bytes_feature(image.filename),
        "image/split": _bytes_feature(image.split),
    })

    assert len(image.captions) == 1
    caption = image.captions[0]
    caption_ids = [vocab.word_to_id(word) for word in caption]

    feature_lists = tf.train.FeatureLists(feature_list={
        "image/caption": _bytes_feature_list(caption),
        "image/caption_ids": _int64_feature_list(caption_ids)
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def process_caption(caption):
    """Processes a caption string into a list of tokenized words.

    Args:
      caption: A string caption.
    Returns:
      A list of strings; the tokenized caption.
    """
    tokens = str(caption).lower().translate(None, string.punctuation).split()
    tokenized_caption = add_start_end(tokens)
    return tokenized_caption


def add_start_end(tokens, start_word="<START>", end_word="<END>"):
    """ Add  start and end words for a caption string

    Args:
      tokens: original tokenized caption
      start_word: word to indicate start of a caption sentence
      end_word: word to indicate end of a caption sentence
    Returns:
      token_caption: tokenized caption
    """
    token_caption = [start_word]
    token_caption.extend(tokens)
    token_caption.append(end_word)

    return token_caption


def create_vocab(captions, min_word_count, word_counts_output_file, word_to_idx_file):
    """Creates the vocabulary of word to word_id.

    The vocabulary is saved to disk in a text file of word counts. The id of each
    word in the file is its corresponding 0-based line number.

    Args:
      captions: A list of lists of strings.
      min_word_count: filter out the words occurs less than min times.
      word_counts_output_file: txt file to record the word counts.
      word_to_idx_file: file to find the idx given a word.
    Returns:
      A Vocabulary object.
    """
    print("Creating vocabulary...")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(word_counts_output_file, "w") as f:
        f.write("Total words: %d \n" % len(counter))
        f.write("Words in vocabulary: %d \n" % len(word_counts))
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))

    print("Wrote vocabulary file:", word_counts_output_file)

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id)

    # save word dictionary to word_to_idx.pkl,
    # which could be used for pretrained word embeddings like Glove
    word_to_idx = {u'<NULL>': 0}
    idx = 1
    for word, count in word_counts:
        word_to_idx[word] = idx
        idx += 1
    pickle(word_to_idx, word_to_idx_file)

    return vocab


def process_image_files(thread_index, ranges, split_name, images, encoder, decoder, vocab,
                        output_dir, num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.

    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      split_name: A train/test/val split name.
      images: List of ImageMetadata.
      encoder: An ImageEncoder object.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.
      output_dir: output directory.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    myfile = open(osp.join(output_dir, '%s_tfrecords.txt' % split_name), 'w')
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (split_name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, encoder, decoder, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                myfile.write("%s %d %s \n" % (image.filename, image.label, image.captions[0]))
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()
        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()
    myfile.close()


def process_dataset(split_name, images, vocab, output_dir, num_shards=1, num_threads=1):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
      split_name: A train/test/val split name.
      images: List of ImageMetadata.
      vocab: A Vocabulary object.
      output_dir: output path.
      num_shards: Integer number of shards for the output files.
    """
    # Break up each image into a separate entity for each caption.
    images = [ImageMetadata(image.label, image.filename, [caption], image.split)
              for image in images for caption in image.captions]

    # Shuffle the ordering of training images. Make the randomization repeatable.
    if split_name == 'train':
        random.seed(12345)
        random.shuffle(images)
    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()
    encoder = ImageEncoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, split_name, images, encoder, decoder, vocab, output_dir, num_shards)
        t = threading.Thread(target=process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in %s." %
          (datetime.now(), len(images), split_name))

