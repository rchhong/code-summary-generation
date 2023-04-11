import openai
import datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq,\
    AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, HfArgumentParser
# from helpers import prepare_dataset_code_summary, compute_rouge_and_bleu
from tqdm import tqdm
import os
import json

NUM_PREPROCESSING_WORKERS = 2

# This function preprocesses the CODESEARCHNET dataset.  Tokenizes snippets and produces natural language
def prepare_dataset_code_summary(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    examples_with_prompts = list(map(lambda x: "Summarize Python: " + x, map(lambda x: x, examples['code'])))
    tokenized_examples = tokenizer(
        examples_with_prompts,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors = 'pt'
    )

    return tokenized_examples

def generate_line_by_line_summaries(path_to_single_line_model):
    # Read in original model

    # Here we select the right model fine-tuning head
    model_class = AutoModelForSeq2SeqLM
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(path_to_single_line_model)
    tokenizer = AutoTokenizer.from_pretrained(path_to_single_line_model, use_fast=True)

    # For each code chunk, tokenize each line and pass it through the model
    prepare_dataset = lambda exs: prepare_dataset_code_summary(exs, tokenizer)

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    dataset = dataset.filter(lambda ex: ex['rewritten_intent'] != None)

    dataset_featurized = dataset.map(
            prepare_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=dataset.column_names
        )

    for ex in tqdm()



async def main():


    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.Model.list()

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Summarize this for a second-grade student:\n\nJupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets in the Solar System combined. Jupiter is one of the brightest objects visible to the naked eye in the night sky, and has been known to ancient civilizations since before recorded history. It is named after the Roman god Jupiter.[19] When viewed from Earth, Jupiter can be bright enough for its reflected light to cast visible shadows,[20] and is on average the third-brightest natural object in the night sky after the Moon and Venus.",
        temperature=0.7,
        max_tokens=64,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )


if __name__ == "main":
    main()