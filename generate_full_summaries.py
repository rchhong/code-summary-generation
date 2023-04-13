import openai
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, SummarizationPipeline, pipeline
# from helpers import prepare_dataset_code_summary, compute_rouge_and_bleu
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import os

NUM_PREPROCESSING_WORKERS = 2
class CustomSummaryGenerationPipeline(SummarizationPipeline):
    def preprocess(self, inputs):
        max_seq_length = self.tokenizer.model_max_length

        examples_with_prompts = list(map(lambda x: "Summarize Python: " + x, inputs))
        # print(examples_with_prompts)
        tokenized_examples = self.tokenizer(
            examples_with_prompts,
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

def generate_line_by_line_summaries(dataset, path_to_model):
    # Set up inference pipeline
    my_pipeline = pipeline(
        task = "summarization",
        model = path_to_model,
        pipeline_class = CustomSummaryGenerationPipeline
    )

    store = []
    for out in tqdm(my_pipeline(KeyDataset(dataset, 'code'))):
        store.append(out)

    return store



async def main():
    # Generate line-by-line summaries
    dataset = datasets.load_dataset('json', data_files="./codesearchnet-corpus/python/final/jsonl/train/python_train_0.jsonl")
    dataset = dataset['train']

    line_by_line_summaries = generate_line_by_line_summaries(dataset, './single_line_summarization_model')

    # GPT Summarization
    openai.api_key = os.getenv("OPENAI_API_KEY")
    summaries = []
    for line_summary in line_by_line_summaries:
        line_summary = "\n".join(line_summary)
        gpt_response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Summarize in a couple of sentences:\n\n" + line_summary,
            temperature=0.7,
            max_tokens=128,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        summaries.append(gpt_response['choices'][0]['text'].strip())

    # Write summaries to file


if __name__ == "main":
    main()