import openai
import argparse
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch import Tensor
import math
from tqdm import tqdm

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_summary_with_backoff(curr_batch):
    # return openai.Completion.create(**kwargs)
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai.Completion.create(
        model="text-davinci-003",
        prompt=curr_batch,
        temperature=0.7,
        max_tokens=48,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

def generate_gpt_summaries():
    # GPT Summarization
    output_file = open("gpt_summaries.jsonl", "w")

    BATCH_SIZE = 20

    with open("joined_file.jsonl", "r") as f:
        curr_batch = []
        for raw_json in f:
            if (len(curr_batch) == BATCH_SIZE):
                summaries = list(map(lambda x: x['summary'], curr_batch))
                line_summaries = list(map(lambda x: x["line_summary"], curr_batch))

                response = get_summary_with_backoff(line_summaries)

                generated_summaries = [""] * BATCH_SIZE
                for choice in response.choices:
                    generated_summaries[choice.index] = choice.text.strip()

                # Write summaries to file
                result = [{"summary": summaries[i], "gpt_summary": generated_summaries[i]} for i in range(len(summaries))]
                for r in result:
                    output_file.write(json.dumps(r) + "\n")

                curr_batch = []

            tmp = json.loads(raw_json)
            summary, line_summary = tmp['summary'], tmp['lines']

            line_summary = "\n".join(line_summary)
            line_summary = "Summarize the code description in a couple of sentences:\n\n" + line_summary

            curr_batch.append({"summary" : summary, "line_summary" : line_summary})
    output_file.close()

def _preprocess_dataset(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length
    inputTokens = [".  ".join(ex) for ex in examples['lines']]

    tokenized_examples = tokenizer(
        inputTokens,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    return {**tokenized_examples, "summary": examples["summary"]}

def generate_bart_summaries():
    # BART Summarization

    output_file = open("bart_summaries.jsonl", "w")
    NUM_PREPROCESSING_WORKERS = 2
    dataset = datasets.load_dataset('json', data_files="./joined_file.jsonl")
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
    BATCH_SIZE = 8

    device = torch.device('cuda') if torch.has_cuda else torch.device('cpu')
    # tokenizer = AutoTokenizer.from_pretrained('./single_line_summarization_model', use_fast=True)

    model = model.to(device)
    NUM_BATCHES = int(math.ceil(len(dataset) / BATCH_SIZE))
    for batch_no in tqdm(range(NUM_BATCHES)):
        batch = dataset[batch_no * BATCH_SIZE : min(len(dataset), (batch_no + 1) * BATCH_SIZE )]
        input = {
            'input_ids': Tensor(batch['input_ids']).long().to(device),
        }
        tokenized_summaries = model.generate(**input)
        decoded_summaries = tokenizer.batch_decode(tokenized_summaries, skip_special_tokens=True)
        cleaned_summaries = [s.strip() for s in decoded_summaries]

        # Write out to file
        for gold_summary, summary in list(zip(batch['summary'], cleaned_summaries)):
            output_file.write(json.dumps({"summary": gold_summary, "generated_summary": summary}) + "\n")

    output_file.close()

def main(args):
    if(args.model == "gpt"):
        generate_gpt_summaries()
    elif(args.model == "bart"):
        generate_bart_summaries()
    else:
        raise Exception("Invalid Model Type")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='FullSummaryGenerator',
                        description='Generates natural-text summaries based on the line-by-line summaries found in joined_file.jsonl',
                        epilog='Text at the bottom of help')

    parser.add_argument("-m", "--model", choices=['gpt', 'bart'], action='store')
    main(parser.parse_args())
