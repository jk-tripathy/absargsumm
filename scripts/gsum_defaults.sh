#!/usr/bin/env bash

$HOME/pegasus-bridle/wrapper.sh python main.py \
    --wandb_project gsum_defaults \
    --accelerator gpu \
    --max_steps 200000 \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --longtext_column article \
    --shorttext_column highlights \
    --guidance gsum \
    --batch_size 100 \
    --stage fit \
    --guidance gsum \
    --encoder_learning_rate 0.002 \
    --decoder_learning_rate 0.2 \
    --warmup_steps 15000 \
