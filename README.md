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

## Generating GPT-Assisted Summaries
```
./scripts/train_and_eval.sh
```
To tune hyperparameters and environment variables, please edit `./scripts/train_and_eval.sh`.  You should not have to adjust the directories, but the hyperparameters can be adjusted at will.
