#!/usr/bin/env bash

python main.py \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --dataset_limit 32 \
    --shorttext_column article \
    --summary_column highlights \
    --guidance gsum \
    --batch_size 4 \
    --stage fit \
