#!/usr/bin/env bash

python main.py \
    --wandb_project gsum_test \
    --accelerator gpu \
    --log_step 16 \
    --max_steps 1000 \
    --warmup_steps 100 \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --dataset_limit 512 \
    --longtext_column article \
    --shorttext_column highlights \
    --guidance_type gsum \
    --batch_size 4 \
