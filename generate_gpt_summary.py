import openai
import os
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

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

def main():
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

if __name__ == "__main__":
    main()
