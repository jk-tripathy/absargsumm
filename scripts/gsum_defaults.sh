#!/usr/bin/env bash

$HOME/pegasus-bridle/wrapper.sh python main.py \
    --wandb_project gsum_defaults \
    --accelerator gpu \
    --log_step 100 \
    --max_steps 200000 \
    --warmup_steps 15000 \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --longtext_column article \
    --shorttext_column highlights \
    --guidance_type gsum \
    --batch_size 50 \
    --stage fit \
