#!/bin/bash
model_directory="$1"
input_file_path="$2"
output_file_path="$3"
python3 test.py --model_directory "$1" --input_file "$2" --output_file "$3"