#!/bin/bash
source your_environment

WORKSPACE=your_work_path
cd $WORKSPACE

torchrun --nproc_per_node=4 --master_port=20001 ./LLaMA-Factory/src/train.py \
    --deepspeed ./LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn auto \
    --model_name_or_path your_model_path \
    --dataset your_data \
    --dataset_dir ./LLaMA-Factory/data \
    --template llama2 \
    --finetuning_type full \
    --output_dir ./LLaMA-Factory/saves \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 2560 \
    --save_steps 20000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16
