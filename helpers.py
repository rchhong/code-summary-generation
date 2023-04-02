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


# This function preprocesses the CoNoLA dataset.  Tokenizes snippets and produces natural language
def prepare_dataset_code_summary(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    examples_with_prompts = list(map(lambda x: "Summarize Python: " + x, examples['snippet']))
    tokenized_examples = tokenizer(
        examples_with_prompts,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors = 'pt'
    )

    tokenized_examples['label'] = tokenizer(
        examples['rewritten_intent'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors = 'pt'
    )['input_ids']

    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_rouge_and_bleu(eval_preds: EvalPrediction, tokenizer):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    argmax = np.argmax(eval_preds.predictions[0], axis = -1)

    decoded_predictions = [tokenizer.decode(x) for x in argmax]
    decoded_labels = [tokenizer.decode(x) for x in eval_preds.label_ids]

    return {
        **rouge.compute(predictions=decoded_predictions,
                               references=decoded_labels),
        **bleu.compute(predictions=decoded_predictions,
                               references=decoded_labels)
    }

# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py