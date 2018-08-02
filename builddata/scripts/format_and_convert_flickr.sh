#!/bin/bash
#
GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$1
#
#train: Finished processing 145000 captions for 29000 images of 29000 identities
#val: Finished processing 5070 captions for 1014 images of 1014 identities
#test: Finished processing 5000 captions for 1000 images of 1000 identities

IMAGE_DIR=/home/zhangying/Documents/Dataset/Flickr30K/images
TEXT_DIR=/home/zhangying/Documents/Dataset/Flickr30K/annotations/dataset_flickr30k.json
OUTPUT_DIR=/home/zhangying/Documents/Dataset/TFRecords/flickr30k
DATASET_NAME=flickr30k

echo "Building the TFRecords for Flickr30K dataset..."

python convert_data.py \
    --dataset_name=${DATASET_NAME} \
    --image_dir=${IMAGE_DIR} \
    --text_dir=${TEXT_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --min_word_count=3 \
    --train_shards=32

echo "Done!"
