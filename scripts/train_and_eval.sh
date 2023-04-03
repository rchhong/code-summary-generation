#!/bin/bash
# Output directory of model
export OUTPUT_DIR="output"
# Number of training outputs
export TRAIN_EPOCHS=4
# Batch size when training
export TRAIN_BATCH_SIZE=8
# Batch size for evaluation (due to HuggingFace Transformer caching)
export EVAL_BATCH_SIZE=8
# Location of dataset
export DATASET_DIR="conala-corpus"

./scripts/train.sh
./scripts/eval.sh

unset OUTPUT_DIR
unset TRAIN_EPOCHS
unset TRAIN_BATCH_SIZE
unset EVAL_BATCH_SIZE
unset DATASET_DIR