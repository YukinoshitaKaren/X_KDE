#!/bin/bash
source your_environment

WORKSPACE=your_work_path
cd $WORKSPACE

MODEL=your_model
DATAPATH=your_test_data
echo $MODEL1

python3  ./eval/eval.py --device "0" --source "en" --target "en" --model $MODEL --data_path $DATAPATH &
# python3  ./eval/eval.py --device "1" --source "en" --target "zh" --model $MODEL --data_path $DATAPATH &
# python3  ./eval/eval.py --device "2" --source "zh" --target "en" --model $MODEL --data_path $DATAPATH &
# python3  ./eval/eval.py --device "3" --source "zh" --target "zh" --model $MODEL --data_path $DATAPATH &