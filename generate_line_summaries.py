import openai
import datasets
from transformers import SummarizationPipeline, pipeline
# from helpers import prepare_dataset_code_summary, compute_rouge_and_bleu
# from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import os
import json

NUM_PREPROCESSING_WORKERS = 2
def generate_line_by_line_summaries(dataset, path_to_model):
    # Set up inference pipeline
    my_pipeline = pipeline(
        task = "summarization",
        model = path_to_model,
        # pipeline_class = CustomSummaryGenerationPipeline
    )

    output_file = open("line_summaries.jsonl", "w")

    for out in tqdm(dataset):
        id, index, raw_line, num_lines = out['id'], out['index'], out['code'], out['lines']
        line_summaries = my_pipeline(raw_line)[0]["summary_text"]
        # Write out to file
        # store.append(out)
        output_file.write(json.dumps({"id" : id, "index" : index, "lines": num_lines, "line_summary":  line_summaries}) + "\n")
    output_file.close()




def main():
    # Generate line-by-line summaries
    dataset = datasets.load_dataset('json', data_files="codesearchnet-corpus/python/final/flattened/test/python_test_0.jsonl")
    dataset = dataset['train']

    generate_line_by_line_summaries(dataset, './single_line_summarization_model')


if __name__ == "__main__":
    main()