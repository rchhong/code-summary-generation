#!/bin/bash
python train_line_summaries.py \
    --do_eval \
    --model $OUTPUT_DIR \
    --dataset $DATASET_DIR/conala-test.json \
    --output_dir $OUTPUT_DIR \
    --task "code-summary" \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE