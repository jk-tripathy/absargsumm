#!/usr/bin/env bash

python main.py \
    --wandb_project gsum_test \
    --accelerator gpu \
    --max_steps 20000 \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --dataset_limit 2000 \
    --longtext_column article \
    --shorttext_column highlights \
    --guidance gsum \
    --batch_size 4 \
    --stage fit \
    --guidance gsum \
    --encoder_learning_rate 0.002 \
    --decoder_learning_rate 0.2 \
    --warmup_steps 2000 \
