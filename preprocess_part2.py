import os
import json
from tqdm import tqdm

parentPath = "./codesearchnet-corpus/python/final/"
sourceDir = "jsonl"
destDir = "flattened"
subdirectories = [
    'test', 
    'train', 
    'valid'
]

if not os.path.exists(os.path.join(parentPath, destDir)):
    os.mkdir(os.path.join(parentPath, destDir))

for d in subdirectories:
    folder = os.path.join(parentPath, sourceDir, d)
    
    if not os.path.exists(os.path.join(parentPath, destDir, d)):
        os.mkdir(os.path.join(parentPath, destDir, d))

    for filename in os.listdir(folder):
        if not filename.endswith("jsonl"):
            continue
        f = open(os.path.join(folder, filename))
        lines = f.readlines()
        f.close()

        newFile = open(os.path.join(parentPath, destDir, d, filename), "w")
        fileNum = filename.split(".")[0].split("_")[-1]
        
        lineCounts = []
        for i, line in tqdm(enumerate(lines)):
            content = json.loads(line)
            code_lines = len(content["code"])
            lineCounts.append(code_lines)
            for code_line_index, code_line in enumerate(content["code"]):
                example = {
                    "id": f"{fileNum}-{i}",
                    "index": code_line_index,
                    "lines": code_lines,
                    "code": code_line
                }
                newFile.write(json.dumps(example) + "\n")

        # with open(os.path.join(parentPath, destDir, d, filename.split(".")[0] + "_LineCounts.json"), "w") as lcFile:
        #     lcFile.write(json.dumps(lineCounts))

        newFile.close()