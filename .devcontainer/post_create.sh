#!/usr/bin/env bash

set -a
source .env
set +a

git clone https://github.com/google/gemma_pytorch.git
python -m pip install --upgrade pip
pip install -r requirements.txt
huggingface-cli download $MODEL_NAME --local-dir ./models/$MODEL_NAME --token $HF_READ_TOKEN
