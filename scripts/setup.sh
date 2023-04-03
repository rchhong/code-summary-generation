#!/bin/bash

echo "Downloading CoNoLa Corpus"
wget -O "conola-corpus.zip" "http://www.phontron.com/download/conala-corpus-v1.1.zip"

unzip "./conola-corpus.zip"
rm "conola-corpus.zip"
