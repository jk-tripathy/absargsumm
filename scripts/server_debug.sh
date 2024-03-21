#!/usr/bin/env bash

$HOME/pegasus-bridle/wrapper.sh python main.py \
    --wandb_project gsum_test \
    --accelerator gpu \
    --log_step 16 \
    --max_steps 1000 \
    --warmup_steps 100 \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --dataset_limit 1024 \
    --longtext_column article \
    --shorttext_column highlights \
    --guidance_type gsum \
    --batch_size 8 \
