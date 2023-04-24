# code-summary-generation
Final Project for CS 388: Natural Language Processing.  Explores utilizes a 2 stage pipeline to generate natural language summaries of Python code.

## Prerequisites
```
transformers==4.23.0
datasets==2.10.0
evaluate
sentencepiece
tqdm
torch
absl-py
rouge_score
nltk
```

## Installation
### Without Conda
```
# Create environment and install prequisites
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Get dataset
scripts/setup.sh

# Make bash script executable
chmod +x scripts/*
```

### Conda
```
# Create environment and install prequisites
conda create --name nlp-final python
conda activate nlp-final

pip install -r requirements.txt

# Get dataset
scripts/setup.sh

# Make bash script executable
chmod +x scripts/*
```

## Fine-Tuning + Evaluation of Translation Module
```
./scripts/train_and_eval.sh
```
To tune hyperparameters and environment variables, please edit `./scripts/train_and_eval.sh`.  You should not have to adjust the directories, but the hyperparameters can be adjusted at will.

## Generating the line-by-line summaries
Required for the GPT-summarization and the BART abalation study.  Warning: takes a very long time to run this.
```
./scripts/generate_line_summaries.sh
```

## Generating GPT-Assisted Line-by-line-based Summaries
```
./scripts/generate_gpt.sh
```

## Generating BART-Assisted Line-by-line-based Summaries
```
./scripts/generate_bart.sh
```

## Generating CodeT5 Control Summaries
```
./scripts/generate_codet5.sh
```


To tune hyperparameters and environment variables, please edit `./scripts/train_and_eval.sh`.  You should not have to adjust the directories, but the hyperparameters can be adjusted at will.

## Computing Metrics
Substitute the model that you wish to evaluate the quality of the summaries of for the model argument.

'''
python compute_scores --model {gpt, bart, codet5}
'''