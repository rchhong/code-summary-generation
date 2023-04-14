import json
import os
import token, tokenize
from collections import deque
from tqdm import tqdm
import argparse

parentPath = "./codesearchnet-corpus/python/final/jsonl"
directories = [
    'test',
    'train',
    'valid'
]

def remove_comments(source):
    mod = open("code_strip", "w")

    prev_toktype = token.INDENT
    first_line = None
    last_lineno = -1
    last_col = 0

    tokgen = tokenize.generate_tokens(source.readline)
    for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
        if 0:   # Change to if 1 to see the tokens fly by.
            print("%10s %-14s %-20r %r" % (
                tokenize.tok_name.get(toktype, toktype),
                "%d.%d-%d.%d" % (slineno, scol, elineno, ecol),
                ttext, ltext
                ))
        if slineno > last_lineno:
            last_col = 0
        if scol > last_col:
            mod.write(" " * (scol - last_col))
        if toktype == token.STRING and prev_toktype == token.INDENT:
            # Docstring
            # mod.write("#--")
            pass
        elif toktype == tokenize.COMMENT:
            # Comment
            # mod.write("##\n")
            pass
        else:
            mod.write(ttext)
        prev_toktype = toktype
        last_col = ecol
        last_lineno = elineno

# def printlist(l):
#     for i, line in enumerate(l):
#         print(line)
#         if i != len(l) - 1:
#             print("-" * 70)

def part1():
    for d in directories:
        folder = os.path.join(parentPath, d)
        for filename in os.listdir(folder):
            f = open(os.path.join(folder, filename))
            lines = f.readlines()
            f.close()

            f = open(os.path.join(folder, filename), "w")

            i = 0
            for entry in lines:
                entry_json = json.loads(entry)

                # remove comments
                with open("code", "w") as code:
                    code.write(entry_json["code"])

                try:
                    with open("code") as code:
                        remove_comments(code)
                except:
                    continue

                code = open("code_strip")
                code_lines = code.readlines()
                code.close()

                os.remove("code")
                os.remove("code_strip")

                code_lines_deque = deque()

                # remove empty lines
                for l in code_lines:
                    if l.strip() == "":
                        continue
                    else:
                        code_lines_deque.append(l)

                # rule: if ends with ":" or "[" or "{" or "(", include next line
                # rule: if ends with "\" or ",", include next line (strip)
                # rule: if line starts with ".", include it (strip)
                # rule: if line starts with "]" or "}" or ")", include it
                skip = False
                updated_code_lines = []
                while(len(code_lines_deque) > 0):
                    current_line = []
                    stripNext = False
                    while(True):
                        if len(code_lines_deque) == 0:
                            skip = True
                            break

                        code_line = code_lines_deque.popleft()
                        if stripNext:
                            code_line = code_line.lstrip()
                        stripNext = False
                        current_line.append(code_line)

                        if current_line[-1].strip()[-1] in [":", "[", "{", "("]:
                            continue
                        if current_line[-1].strip()[-1] in ["\\", ","]:
                            current_line[-1] = code_line.rstrip() + " "
                            stripNext = True
                            continue
                        if len(code_lines_deque) >= 1 and code_lines_deque[0].strip()[0] == ".":
                            current_line[-1] = code_line.rstrip()
                            stripNext = True
                            continue
                        if len(code_lines_deque) >= 1 and code_lines_deque[0].strip()[0] in ["]", "}", ")"]:
                            continue
                        break
                    if skip:
                        break
                    updated_code_lines.append("".join(current_line))
                if skip:
                    continue

                no_space_tokens = [".", ")", ":", "/"]
                summary = ""
                for doc_token in entry_json["docstring_tokens"]:
                    if len(summary) == 0:
                        summary += doc_token
                    else:
                        if doc_token in no_space_tokens or summary[-1] == "(":
                            summary += doc_token
                        else:
                            summary += " " + doc_token

                updated_entry = {}
                updated_entry["summary"] = summary
                updated_entry["code"] = updated_code_lines
                updated_entry_json = json.dumps(updated_entry)
                f.write(updated_entry_json + "\n")

                i += 1
                print(f"Completed line {i}/{len(lines)} in file {filename} in folder {d}\r", end="")

            f.close()

def part2():
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part1", help="tokenize code search net", action="store_true")
    parser.add_argument("--part2", help="flatten code search net", action="store_true")
    args = parser.parse_args()

    if args.part1 or args.part2:
        if args.part1:
            part1()
        if args.part2:
            part2()
    else:
        part1()
        part2()

if __name__ == "__main__":
    main()
