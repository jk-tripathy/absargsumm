#!/usr/bin/env bash

# Preprocess the CNN/DailyMail dataset
python data/dataset.py \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --longtext_column article \
    --shorttext_column highlights \
