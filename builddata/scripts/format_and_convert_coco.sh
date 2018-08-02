#!/bin/bash
#
GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$1
#
# train: Finished processing 414113 captions for 82783 images of 82783 identities
# val: Finished processing 152634 captions for 30504 images of 30504 identities
# test: Finished processing 25010 captions for 5000 images of 5000 identities

IMAGE_DIR=/home/zhangying/Documents/Dataset/COCO/image
TEXT_DIR=/home/zhangying/Documents/Dataset/COCO/annotations/dataset_coco.json
OUTPUT_DIR=/home/zhangying/Documents/Dataset/TFRecords/coco
DATASET_NAME=coco

echo "Building the TFRecords for MSCOCO dataset..."

python convert_data.py \
    --image_dir=${IMAGE_DIR} \
    --text_dir=${TEXT_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --dataset_name=${DATASET_NAME} \
    --min_word_count=3 \
    --train_shards=32

echo "Done!"
