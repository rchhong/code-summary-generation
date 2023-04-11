import os
import json
from tqdm import tqdm

file = open("./codesearchnet-corpus/python/final/jsonl/train/python_train_0.jsonl")
lines = file.readlines()
file.close()

newFile = open("./python_train_0_FLATTENED.jsonl", "w")

for i, line in tqdm(enumerate(lines)):
    content = json.loads(line)
    for code_line in content["code"]:
        example = {
            "example": i,
            "code": code_line
        }
        newFile.write(json.dumps(example))

newFile.close()