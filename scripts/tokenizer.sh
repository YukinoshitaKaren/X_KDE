#!/bin/bash
source your_environment

WORKSPACE=your_work_path
cd $WORKSPACE

python sft/tokenizer.py --input_file your_input_file --tokenizer_path your_tokenizer_path