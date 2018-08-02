import numpy as np
from argparse import ArgumentParser
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import os.path as osp

from retrieval_eval import *
from datetime import datetime


def _get_test_data(result_dir):
    Text_X = np.load(osp.join(result_dir, 'test_caption_features.npy'))
    Text_Y = np.load(osp.join(result_dir, 'test_caption_labels.npy'))
    Image_X = np.load(osp.join(result_dir, 'test_image_features.npy'))
    Image_Y = np.load(osp.join(result_dir, 'test_image_labels.npy'))
    return Text_X, Text_Y, Image_X, Image_Y


def _eval_retrieval(PX, PY, GX, GY):

    # D_{i, j} is the distance between the ith array from PX and the jth array from GX.
    D = pairwise_distances(PX, GX, metric=args.method, n_jobs=-2)
    Rank = np.argsort(D, axis=1)

    # Evaluation
    recall_1 = recall_at_k(Rank, PY, GY, k=1)  # Recall @ K
    print "{:8}{:8.2%}".format('Recall@1', recall_1)

    recall_5 = recall_at_k(Rank, PY, GY, k=5)  # Recall @ K
    print "{:8}{:8.2%}".format('Recall@5', recall_5)

    recall_10 = recall_at_k(Rank, PY, GY, k=10)  # Recall @ K
    print "{:8}{:8.2%}".format('Recall@10', recall_10)

    map_value = mean_average_precision(Rank, PY, GY)  # Mean Average Precision
    print "{:8}{:8.2%}".format('MAP', map_value)

    return recall_1, recall_5, recall_10, map_value


def main(args):
    Text_X, Text_Y, Image_X, Image_Y= _get_test_data(args.result_dir)

    if args.method == 'cosine':
        # L2 normalization
        Text_X = preprocessing.normalize(Text_X, norm='l2', axis=1)
        Image_X = preprocessing.normalize(Image_X, norm='l2', axis=1)

    mytime = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    text_file = open(osp.join(args.result_dir, "retrieval-%s.txt" % mytime), "w")
    print("Text-to-Image Evaluation...")
    recall_1, recall_5, recall_10, map_value = _eval_retrieval(Text_X, Text_Y, Image_X, Image_Y)
    text_file.write("Text-to-Image Evaluation \n")
    text_file.write("{:8}{:8.2%}\n".format('Recall@1', recall_1))
    text_file.write("{:8}{:8.2%}\n".format('Recall@5', recall_5))
    text_file.write("{:8}{:8.2%}\n".format('Recall@10', recall_10))
    text_file.write("{:8}{:8.2%}\n".format('MAP', map_value))

    print("Image-to-Text Evaluation...")
    recall_1, recall_5, recall_10, map_value = _eval_retrieval(Image_X, Image_Y, Text_X, Text_Y)
    text_file.write("Image-to-Text Evaluation \n")
    text_file.write("{:8}{:8.2%}\n".format('Recall@1', recall_1))
    text_file.write("{:8}{:8.2%}\n".format('Recall@5', recall_5))
    text_file.write("{:8}{:8.2%}\n".format('Recall@10', recall_10))
    text_file.write("{:8}{:8.2%}\n".format('MAP', map_value))
    text_file.close()


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Metric learning and evaluate performance")
    parser.add_argument('result_dir',
                        help="Result directory. Containing extracted features and labels. "
                             "CMC curve will also be saved to this directory.")
    parser.add_argument('--method', choices=['euclidean', 'cosine'],
                        default='cosine')
    args = parser.parse_args()
    main(args)
