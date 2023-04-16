from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

sum_file = "gpt_summaries_demo.jsonl"
input_file = open(sum_file)
average_bleu_score = 0
average_rouge1_score = 0
average_rougeL_score = 0
total_lines = 0
chencherry = SmoothingFunction()

for line in input_file:
    content = json.loads(line)
    summary = content["summary"]
    gpt_summary = content["gpt_summary"]

    bleu_score = sentence_bleu([summary.split()], gpt_summary.split(), smoothing_function=chencherry.method1, weights=(0.25, 0.25, 0.25, 0.25))
    rouge_score = scorer.score(summary, gpt_summary)

    average_bleu_score += bleu_score
    average_rouge1_score += rouge_score["rouge1"].fmeasure
    average_rouge1_score += rouge_score["rougeL"].fmeasure
    total_lines += 1

average_bleu_score /= total_lines
average_rouge1_score /= total_lines
average_rougeL_score /= total_lines
print(f"Average Bleu Score: {average_bleu_score}")
print(f"Average Rouge1 Score: {average_rouge1_score}")
print(f"Average RougeL Score: {average_rougeL_score}")

input_file.close()