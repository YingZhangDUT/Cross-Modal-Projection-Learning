#!/bin/bash
#
GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$1

# Where the dataset is saved to.
DATASET_DIR=/home/zhangying/Documents/Dataset/TFRecords/coco

# Where the checkpoint and logs saved to.
DATASET_NAME=coco
SAVE_NAME=coco_resnet152_cmpm
CKPT_DIR=${SAVE_NAME}/checkpoint
RESULT_DIR=${SAVE_NAME}/results

MODEL_NAME=resnet_v1_152
MODEL_SCOPE=resnet_v1_152

SPLIT_NAME=test

for i in $(seq 0 20)
do
    # Run evaluation.
    python test_image_text.py \
      --checkpoint_dir=${CKPT_DIR} \
      --eval_dir=${RESULT_DIR} \
      --dataset_name=${DATASET_NAME} \
      --split_name=${SPLIT_NAME} \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${MODEL_NAME} \
      --model_scope=${MODEL_SCOPE} \
      --preprocessing_name=${MODEL_NAME} \
      --batch_size=1

    echo "Evaluating with Cosine Distance..."
    python2 evaluation/bidirectional_eval.py ${RESULT_DIR} --method cosine

    echo "Waiting..."
    sleep 45m

done
