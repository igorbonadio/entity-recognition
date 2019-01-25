#!/usr/bin/env bash

curl -o dataset/glove_s300.zip http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s300.zip

unzip dataset/glove_s300.zip -d dataset

rm dataset/glove_s300.zip

pip install -r requirements.txt