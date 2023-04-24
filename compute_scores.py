from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json
import argparse

def main(args):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    if(args.model == "gpt"):
        sum_file = "gpt_summaries.jsonl"
    elif(args.model == "bart"):
        sum_file = "bart_summaries.jsonl"
    elif(args.model == "codet5"):
        sum_file = "codet5_summaries.jsonl"
    else:
        raise Exception("Invalid Model")

    input_file = open(sum_file)
    average_bleu_score = 0
    average_rouge1_score = 0
    average_rougeL_score = 0
    total_lines = 0
    chencherry = SmoothingFunction()

    for line in input_file:
        content = json.loads(line)
        summary = content["summary"]
        if(args.model == "gpt"):
            generated_summary = content["gpt_summary"]
        elif(args.model == "bart" or args.model == "codet5"):
            generated_summary = content["generated_summary"]

        bleu_score = sentence_bleu([summary.split()], generated_summary.split(), smoothing_function=chencherry.method1, weights=(0.25, 0.25, 0.25, 0.25))
        rouge_score = scorer.score(summary, generated_summary)

        average_bleu_score += bleu_score
        average_rouge1_score += rouge_score["rouge1"].fmeasure
        average_rouge1_score += rouge_score["rougeL"].fmeasure
        total_lines += 1

    average_bleu_score /= total_lines
    average_rouge1_score /= total_lines
    average_rougeL_score /= total_lines
    print(f"Model: {args.model}")
    print(f"Average Bleu Score: {average_bleu_score}")
    print(f"Average Rouge1 Score: {average_rouge1_score}")
    print(f"Average RougeL Score: {average_rougeL_score}")

    input_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='FullSummaryGenerator',
                        description='Generates natural-text summaries based on the line-by-line summaries found in joined_file.jsonl',
                        epilog='Text at the bottom of help')

    parser.add_argument("-m", "--model", choices=['gpt', 'bart', "codet5"], action='store')
    main(parser.parse_args())