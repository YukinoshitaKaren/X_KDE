#!/bin/bash
source your_environment

WORKSPACE=your_work_path
cd $WORKSPACE

CONFIG_FILES=("example.json")

for CONFIG in "${CONFIG_FILES[@]}"; do
  echo "Running with config: $CONFIG"
  accelerate launch --config_file=./ORPO/Alignment/deepspeed/deepspeed_zero2_dpo.yaml --num_processes 4 ./ORPO/Alignment/run_orpo.py --training_config=$CONFIG 
done
