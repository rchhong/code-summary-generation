import json

source_file = "codesearchnet-corpus/python/final/jsonl/test/python_test_0.jsonl"
f = open(source_file)
source_lines = f.readlines()
f.close()

source_lines_parsed = []
for l in source_lines:
    source_lines_parsed.append(json.loads(l))

joined_file = open("joined_file.jsonl", "w")
with open("line_summaries.jsonl", "r") as f:
    joined = {}
    for line in f:
        line_content = json.loads(line)
        id = line_content["id"]
        ex_index = id.split("-")[-1]
        line_index = line_content["index"]

        if not (id in joined):
            actual_summary = source_lines_parsed[int(ex_index)]["summary"]
            joined[id] = {"summary": actual_summary, "lines": [None for _ in range(len(source_lines_parsed[int(ex_index)]["code"]))]}
        joined[id]["lines"][int(line_index)] = line_content["line_summary"]

        if not (None in joined[id]["lines"]):
            joined_file.write(json.dumps(joined[id]) + "\n")
            del joined[id]

joined_file.close()
