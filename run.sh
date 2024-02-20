#!/usr/bin/env bash

python main.py \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --dataset_limit 32 \
    --longtext_column article \
    --shorttext_column highlights \
    --guidance gsum \
    --batch_size 4 \
    --stage fit \
