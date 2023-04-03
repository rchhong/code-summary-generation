#!/bin/bash
python train_line_summaries.py \
    --do_train \
    --dataset conala-corpus/conala-train.json \
    --output_dir $OUTPUT_DIR \
    --task "code-summary" \
    --num_train_epochs $TRAIN_EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE