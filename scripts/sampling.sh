#!/bin/bash
source your_environment

WORKSPACE=your_work_path
cd $WORKSPACE
MODEL=your_model_path
DATAPATH=your_data_path
DATANAME=your_data_name

datapath=./ORPO/result/vllm/$DATANAME.json
INPUT_DIR=./ORPO/result/vllm/pre/
OUTPUT=./ORPO/result/vllm/process/$DATANAME.json

python3 ./ORPO/src/sampling_vllm.py --model $MODEL --data_path $DATAPATH

num_proc=8
data_len=2400
processes=()
for ((i = 0; i < num_proc; i++)); do
    begin_index=$((i * data_len))
    CUDA_VISIBLE_DEVICES=$i python ./ORPO/src/batched.py --begin_index "$begin_index" --data_length "$data_len" --data_file "$datapath" &
    echo "Process $i Begin!"
    processes+=($!)
done

for pid in "${processes[@]}"; do
    wait "$pid"
done

python3 ./ORPO/src/merge.py --input_dir $INPUT_DIR --output_file $OUTPUT
python3 ./ORPO/src/exact.py --input_dir $OUTPUT