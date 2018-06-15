#!/usr/bin/env sh
set -e

echo "convert:............."
CDATASET_DIR=F:/python/mySSD/voc/mydata/
OUTPUT_DIR=F:/python/mySSD/voc/mydata/output/
python my_tf_convert_data.py \
     --dataset_name=plastic \
     --dataset_dir=${CDATASET_DIR} \
     --output_name=plastic_train \
     --output_dir=${OUTPUT_DIR}
	
 echo "Eval:............."
 EVAL_DIR=./log/
 DATASET_DIR=./voc/output
 CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_300x300_iter_120000.ckpt
 python eval_ssd_network.py \
     --eval_dir=${EVAL_DIR} \
     --dataset_dir=${DATASET_DIR} \
     --dataset_name=pascalvoc_2012 \
     --dataset_split_name=train \
     --model_name=ssd_300_vgg \
     --checkpoint_path=${CHECKPOINT_PATH} \
     --batch_size=4
	
# echo "train:.................."
# DATASET_DIR=./voc/mydata/output
# TRAIN_DIR=./test/
# CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
# python train_plastic.py \·
#     --train_dir=${TRAIN_DIR} \
#     --dataset_dir=${DATASET_DIR} \
#     --dataset_name=plastic \
# 	--num_classes=2 \
#     --max_number_of_steps=4000 \
#     --dataset_split_name=train \
#     --model_name=ssd_300_vgg \
#     --checkpoint_path=${CHECKPOINT_PATH} \
#     --checkpoint_model_scope=vgg_16 \
#     --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
#     --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
#     --save_summaries_secs=60 \
#     --save_interval_secs=200 \
#     --weight_decay=0.0005 \
#     --optimizer=momentum \
#     --momentum=0.9 \
#     --learning_rate=0.0001 \
#     --end_learning_rate=0.0001 \
#     --learning_rate_decay_type=fixed \
#     --learning_rate_decay_factor=0.94 \
#     --batch_size=16
	
# DATASET_DIR=./voc/output
# TRAIN_DIR=./voclog/
# CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
# python train_ssd_network.py \
#     --train_dir=${TRAIN_DIR} \
#     --dataset_dir=${DATASET_DIR} \
#     --dataset_name=pascalvoc_2012 \
#     --max_number_of_steps=200 \
#     --dataset_split_name=train \
#     --model_name=ssd_300_vgg \
#     --checkpoint_path=${CHECKPOINT_PATH} \
#     --checkpoint_model_scope=vgg_16 \
#     --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
#     --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
#     --save_summaries_secs=60 \
#     --save_interval_secs=200 \
#     --weight_decay=0.0005 \
#     --optimizer=momentum \
#     --learning_rate=0.001 \
#     --learning_rate_decay_factor=0.94 \
#     --batch_size=8

DATASET_DIR=./voc/mydata/output
TRAIN_DIR=./test_log_finetune/
CHECKPOINT_PATH=./test/model.ckpt-5000
python train_plastic.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=plastic \
    --num_classes=2 \
    --max_number_of_steps=5000 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=ssd_300_vgg \
    --save_summaries_secs=60 \
    --save_interval_secs=200 \
    --weight_decay=0.0005 \
    --optimizer=momentum \
    --momentum=0.9 \
    --learning_rate=0.00001 \
    --learning_rate_decay_type=fixed \
    --learning_rate_decay_factor=0.94 \
    --batch_size=8


# DATASET_DIR=./voc/output
# TRAIN_DIR=./voclog/
# CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
# python train_ssd_network.py \
#     --train_dir=${TRAIN_DIR} \
#     --dataset_dir=${DATASET_DIR} \
#     --dataset_name=pascalvoc_2012 \
#     --dataset_split_name=train \
#     --model_name=ssd_300_vgg \
#     --checkpoint_path=${CHECKPOINT_PATH} \
#     --save_summaries_secs=60 \
#     --save_interval_secs=200 \
#     --weight_decay=0.0005 \
#     --optimizer=adam \
#     --learning_rate=0.001 \
#     --batch_size=16
echo "Done."