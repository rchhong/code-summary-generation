import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# This function preprocesses the CoNoLA dataset.  Tokenizes snippets and produces natural language
def prepare_dataset_code_summary(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    examples_with_prompts = list(map(lambda x: "Summarize Python: " + x, examples['snippet']))
    tokenized_examples = tokenizer(
        examples_with_prompts,
        text_target=examples['rewritten_intent'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors = 'pt'
    )

    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_rouge_and_bleu(eval_preds: EvalPrediction, tokenizer):
    encoded_predictions = eval_preds.predictions

    decoded_predictions = tokenizer.batch_decode(encoded_predictions, skip_special_tokens=True)

    labels = np.where(eval_preds.label_ids != -100, eval_preds.label_ids, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_predictions = [pred.strip() for pred in decoded_predictions]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    print(decoded_predictions)
    print(decoded_labels)

    return {
        **rouge.compute(predictions=decoded_predictions,
                               references=decoded_labels),
        **bleu.compute(predictions=decoded_predictions,
                               references=decoded_labels)
    }

# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py