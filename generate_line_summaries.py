import datasets
from transformers import pipeline, SummarizationPipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# from helpers import prepare_dataset_code_summary, compute_rouge_and_bleu
# from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import json
from torch import Tensor
import torch
import math

NUM_PREPROCESSING_WORKERS = 2
class CustomSummaryGenerationPipeline(SummarizationPipeline):

    def preprocess(self, inputs):
        max_seq_length = self.tokenizer.model_max_length

        # print(examples_with_prompts)
        tokenized_examples = self.tokenizer(
            inputs  ,
            truncation=True,
            max_length=max_seq_length,
            padding='max_length',
            return_tensors = 'pt'
        )
        return tokenized_examples

    def _forward(self, model_inputs):
        outputs = self.model.generate(**model_inputs)
        return {"encoded_pred": outputs}

    def postprocess(self, model_outputs):
        # print(model_outputs)
        encoded_predictions = model_outputs["encoded_pred"]
        decoded_predictions = self.tokenizer.batch_decode(encoded_predictions, skip_special_tokens=True)
        # decoded_predictions = [pred.strip() for pred in decoded_predictions]
        return decoded_predictions

def generate_line_by_line_summaries(dataset, path_to_model, batch_size = 8):
    # Set up inference pipeline
    device = torch.device('cuda') if torch.has_cuda else torch.device('cpu')

    model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model)
    output_file = open("line_summaries.jsonl", "w")

    tokenizer = AutoTokenizer.from_pretrained('./single_line_summarization_model', use_fast=True)

    NUM_BATCHES = int(math.ceil(len(dataset) / batch_size))
    for batch_no in tqdm(range(NUM_BATCHES)):
        batch = dataset[batch_no * batch_size : min(len(dataset), (batch_no + 1) * batch_size )]
        input = {
            'input_ids': Tensor(batch['input_ids']).long().to(device),
            'attention_mask': Tensor(batch['attention_mask']).long().to(device)
        }
        tokenized_line_summaries = model.generate(**input)
        decoded_line_summaries = tokenizer.batch_decode(tokenized_line_summaries, skip_special_tokens=True)
        cleaned_line_summaries = [s.strip() for s in decoded_line_summaries]
        # Write out to file

        for id, index, num_lines, line_summary in list(zip(batch['id'], batch['index'], batch['num_lines'], cleaned_line_summaries)):
            output_file.write(json.dumps({"id" : id, "index" : index, "lines": num_lines, "line_summary":  line_summary}) + "\n")

    output_file.close()


PREFIX = "Summarize Python:"
def _preprocess_dataset(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length
    examples_with_prompts= ["Summarize Python:" + ex for ex in examples['code']]

    tokenized_examples = tokenizer(
        examples_with_prompts,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors = 'pt'
    )

    return {**tokenized_examples, "id": examples["id"], "index": examples["index"], "num_lines": examples["lines"]}

def main():
    # Generate line-by-line summaries
    dataset = datasets.load_dataset('json', data_files="codesearchnet-corpus/python/final/flattened/test/python_test_0.jsonl")
    dataset = dataset['train']

    tokenizer = AutoTokenizer.from_pretrained('./single_line_summarization_model', use_fast=True)
    preprocess_dataset = lambda exs: _preprocess_dataset(exs, tokenizer)

    dataset = dataset.map(
        preprocess_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=dataset.column_names
    )


    generate_line_by_line_summaries(dataset, './single_line_summarization_model')


if __name__ == "__main__":
    main()