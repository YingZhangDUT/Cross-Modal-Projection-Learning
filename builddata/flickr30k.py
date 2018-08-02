"""
    Converts Flickr30k data into TFRecords
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from dataset_utils import *

FLAGS = tf.flags.FLAGS


def process_metadata(split_name, caption_data, image_dir):
    """Process the captions and combine the data into a list of ImageMetadata.
    Args:
      split_name: A train/test/val split name.
      caption_data: caption file containing caption annotations.
      image_dir: Directory containing the image files.
    Returns:
      A list of ImageMetadata.
    """
    print("Processing image-text...")
    id_to_captions = {}
    image_metadata = []
    num_captions = 0
    count = 0
    for img in caption_data:
        count += 1
        label = img["imgid"]
        filename = os.path.join(image_dir, img["filename"])
        # print(filename)
        assert os.path.exists(filename)
        captions = [add_start_end(c["tokens"]) for c in img["sentences"]]
        id_to_captions.setdefault(label, [])
        id_to_captions[label].append(captions)
        split = img["split"]
        assert split == split_name
        image_metadata.append(ImageMetadata(label, filename, captions, split))
        num_captions += len(captions)

    num_examples = len(caption_data)
    num_classes = len(id_to_captions)

    print("Finished processing %d captions for %d images of %d identities in %s" %
          (num_captions, num_examples, num_classes, split_name))

    # Write out the data preparation information.
    output_file = '%s/%s_data_info.txt' % (FLAGS.output_dir, split_name)
    with tf.gfile.FastGFile(output_file, "w") as f:
        f.write("Finished processing %d captions for %d images of %d identities in %s." %
                (num_captions, num_examples, num_classes, split_name))

    return image_metadata


def run():
    """Convert the original image and caption into tfrecords.
    """
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # process captions
    with tf.gfile.FastGFile(FLAGS.text_dir, "r") as f:
        caption_data = json.load(f)

    test_caption_data = [img for img in caption_data['images'] if img['split'] == 'test']
    val_caption_data = [img for img in caption_data['images'] if img['split'] == 'val']
    train_caption_data = [img for img in caption_data['images'] if img['split'] == 'train']

    train_captions = [add_start_end(c["tokens"]) for annotation in train_caption_data for c in annotation["sentences"]]

    # Create vocabulary from the captions.
    vocab = create_vocab(train_captions,
                         min_word_count=FLAGS.min_word_count,
                         word_counts_output_file=os.path.join(FLAGS.output_dir, FLAGS.word_counts_output_file),
                         word_to_idx_file=os.path.join(FLAGS.output_dir, FLAGS.word_to_idx_file))

    # create ImageMetadata
    val_dataset = process_metadata("val", val_caption_data, FLAGS.image_dir)
    test_dataset = process_metadata("test", test_caption_data, FLAGS.image_dir)
    train_dataset = process_metadata("train", train_caption_data, FLAGS.image_dir)

    # convert to TFRecords
    process_dataset("val", val_dataset, vocab, output_dir=FLAGS.output_dir,
                    num_shards=FLAGS.val_shards, num_threads=FLAGS.num_threads)
    process_dataset("test", test_dataset, vocab, output_dir=FLAGS.output_dir,
                    num_shards=FLAGS.test_shards, num_threads=FLAGS.num_threads)
    process_dataset("train", train_dataset, vocab, output_dir=FLAGS.output_dir,
                    num_shards=FLAGS.train_shards, num_threads=FLAGS.num_threads)

