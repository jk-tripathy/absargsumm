#!/usr/bin/env bash

python main.py \
    --wandb_project gsum_test \
    --accelerator gpu \
    --max_steps 20000 \
    --warmup_steps 2000 \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --dataset_limit 2000 \
    --longtext_column article \
    --shorttext_column highlights \
    --guidance_type gsum \
    --batch_size 4 \
    --stage fit \
