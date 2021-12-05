#!/bin/bash
data_path="$1"
model_path="$2"
python3 train.py --data_path "$1" --model_path "$2"
