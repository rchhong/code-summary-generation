import openai
import os
import json

async def main():
    # GPT Summarization
    openai.api_key = os.getenv("OPENAI_API_KEY")
    output_file = open("gpt_summaries.jsonl", "w")
    with open("joined_file.jsonl", "r") as f:
        for raw_json in f:
            tmp = json.loads(raw_json)
            summary, line_summary = tmp['summary'], tmp['lines']
            line_summary = "\n".join(line_summary)

            # Get GPT Summary
            gpt_response = await openai.Completion.create(
                model="text-davinci-003",
                prompt="Summarize in a couple of sentences:\n\n" + line_summary,
                temperature=0.7,
                max_tokens=128,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            output = {
                "summary" : summary,
                "gpt_summary": gpt_response['choices'][0]['text'].strip()
            }

            # Write summaries to file
            output_file.write(json.dumps(output) + "\n")
    output_file.close()

if __name__ == "__main__":
    main()