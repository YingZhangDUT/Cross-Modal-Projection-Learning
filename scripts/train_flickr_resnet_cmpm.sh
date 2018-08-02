#!/bin/bash
#
GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$1

# Where the dataset is saved to.
DATASET_DIR=/home/zhangying/Documents/Dataset/TFRecords/flickr30k
RESTORE_PATH=/home/zhangying/Documents/PretrainedModels/resnet_v1_152

# Where the checkpoint and logs will be saved to.
DATASET_NAME=flickr30k
SAVE_NAME=flickr30k_resnet152_cmpm
CKPT_DIR=${SAVE_NAME}/checkpoint
LOG_DIR=${SAVE_NAME}/logs
SAMPLE_DIR=${SAVE_NAME}/train_samples

SPLIT_NAME=train

# Model setting
MODEL_NAME=resnet_v1_152
MODEL_SCOPE=resnet_v1_152
RESTORE_SCOPES=resnet_v1_152
RESTORE_EXCLUDE=resnet_v1_152/Logits
TRAIN_EXCLUDE=resnet_v1_152


# Run training stage 1 with fixing Resnet-152
python train_image_text.py \
    --checkpoint_dir=${CKPT_DIR} \
    --log_dir=${LOG_DIR} \
    --train_samples_dir=${SAMPLE_DIR} \
    --dataset_name=${DATASET_NAME}\
    --dataset_split_name=${SPLIT_NAME} \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --model_scope=${MODEL_SCOPE} \
    --preprocessing_name=${MODEL_NAME} \
    --restore_pretrain=True \
    --restore_path=${RESTORE_PATH} \
    --restore_scopes=${RESTORE_SCOPES} \
    --checkpoint_exclude_scopes=${RESTORE_EXCLUDE} \
    --trainable_exclude_scopes=${TRAIN_EXCLUDE} \
    --num_epochs=15 \
    --ckpt_steps=5000 \
    --batch_size=32 \
    --num_classes=31783 \
    --optimizer=adam \
    --learning_rate=0.0002 \
    --opt_epsilon=1e-8 \
    --CMPM=True \
    --CMPC=False

# Run training stage 2 with training the whole model
python train_image_text.py \
    --checkpoint_dir=${CKPT_DIR} \
    --log_dir=${LOG_DIR} \
    --train_samples_dir=${SAMPLE_DIR} \
    --dataset_name=${DATASET_NAME}\
    --dataset_split_name=${SPLIT_NAME} \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --model_scope=${MODEL_SCOPE} \
    --preprocessing_name=${MODEL_NAME} \
    --restore_scopes=${RESTORE_SCOPES} \
    --restore_pretrain=False \
    --num_epochs=30 \
    --ckpt_steps=5000 \
    --batch_size=32 \
    --num_classes=31783 \
    --optimizer=adam \
    --learning_rate=0.00002 \
    --opt_epsilon=1e-8 \
    --CMPM=True \
    --CMPC=False
