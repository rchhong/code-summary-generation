#!/bin/bash

echo "Downloading CoNoLa Corpus"
wget -O "conola-corpus.zip" "http://www.phontron.com/download/conala-corpus-v1.1.zip"

unzip "./conola-corpus.zip"
rm "conola-corpus.zip"

echo "Downloading CodeSearchNet"
wget -O "codesearchnet-corpus.zip" "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip"

mkdir codesearchnet-corpus
mv codesearchnet-corpus.zip codesearchnet-corpus/codesearchnet-corpus.zip
cd codesearchnet-corpus
unzip "./codesearchnet-corpus.zip"
rm "./codesearchnet-corpus.zip"

cd ./python/final/jsonl/test
for filename in ./*; do
    gzip -d $filename
done

cd ../train
for filename in ./*; do
    gzip -d $filename
done

cd ../valid
for filename in ./*; do
    gzip -d $filename
done

cd ../../../../..
# python3 preprocess.py