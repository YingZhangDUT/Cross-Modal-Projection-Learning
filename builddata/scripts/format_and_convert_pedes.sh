#!/bin/bash
#
GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$1
#
# train: Finished processing 68126 captions for 34054 images of 11003 identities
# val: Finished processing 6158 captions for 3078 images of 1000 identities
# test: Finished processing 6156 captions for 3074 images of 1000 identities

IMAGE_DIR=/home/zhangying/Documents/Dataset/CUHK-PEDES/imgs
TEXT_DIR=/home/zhangying/Documents/Dataset/CUHK-PEDES/reid_raw.json
OUTPUT_DIR=/home/zhangying/Documents/Dataset/TFRecords/pedes
DATASET_NAME=pedes

echo "Building the TFRecords for CUHK-PEDES dataset..."

python convert_data.py \
    --image_dir=${IMAGE_DIR} \
    --text_dir=${TEXT_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --dataset_name=${DATASET_NAME} \
    --min_word_count=3

echo "Done!"
