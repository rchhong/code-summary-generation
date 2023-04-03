#!/bin/bash
export OUTPUT_DIR="output"
export TRAIN_EPOCHS=4
export TRAIN_BATCH_SIZE=8
export EVAL_BATCH_SIZE=8

./scripts/train.sh
./scripts/eval.sh

unset OUTPUT_DIR
unset TRAIN_EPOCHS
unset TRAIN_BATCH_SIZE
unset EVAL_BATCH_SIZE