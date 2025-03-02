import datasets
from transformers import pipeline, SummarizationPipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# from helpers import prepare_dataset_code_summary, compute_rouge_and_bleu
# from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import json
from torch import Tensor
import torch
import math
import datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq,\
    AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, HfArgumentParser
# from helpers import prepare_dataset_code_summary, compute_rouge_and_bleu
import os
import json

def generate_summaries(dataset, model, tokenizer, batch_size=8):
    # Set up inference pipeline
    device = torch.device('cuda') if torch.has_cuda else torch.device('cpu')

    output_file = open("codeT5_summaries.jsonl", "w")
    # tokenizer = AutoTokenizer.from_pretrained('./single_line_summarization_model', use_fast=True)

    model = model.to(device)
    NUM_BATCHES = int(math.ceil(len(dataset) / batch_size))
    for batch_no in tqdm(range(NUM_BATCHES)):
        batch = dataset[batch_no * batch_size : min(len(dataset), (batch_no + 1) * batch_size )]
        input = {
            'input_ids': Tensor(batch['input_ids']).long().to(device),
        }
        tokenized_summaries = model.generate(**input)
        decoded_summaries = tokenizer.batch_decode(tokenized_summaries, skip_special_tokens=True)
        cleaned_summaries = [s.strip() for s in decoded_summaries]

        # Write out to file
        for gold_summary, summary in list(zip(batch['docstring'], cleaned_summaries)):
            output_file.write(json.dumps({"summary": gold_summary, "generated_summary": summary}) + "\n")

    output_file.close()

def _preprocess_dataset(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length
    inputTokens = [" ".join(ex) for ex in examples['code_tokens']]

    tokenized_examples = tokenizer(
        inputTokens,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    docstrings = []
    for d in examples["docstring_tokens"]:
        no_space_tokens = [".", ")", ":", "/"]
        docstring = ""
        for doc_token in d:
            if len(docstring) == 0:
                docstring += doc_token
            else:
                if doc_token in no_space_tokens or docstring[-1] == "(":
                    docstring += doc_token
                else:
                    docstring += " " + doc_token
        docstrings.append(docstring)

    return {**tokenized_examples, "docstring": docstrings}


def main():
    NUM_PREPROCESSING_WORKERS = 2

    dataset = datasets.load_dataset('json', data_files="codesearchnet-corpus/test/python_test_0.jsonl")
    dataset = dataset['train']

    MODEL_NAME = "Salesforce/codet5-base-multi-sum"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    preprocess_dataset = lambda exs: _preprocess_dataset(exs, tokenizer)

    dataset = dataset.map(
        preprocess_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=dataset.column_names
    )

    generate_summaries(dataset, model, tokenizer)

if __name__ == "__main__":
    main()