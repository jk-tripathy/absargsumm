#!/usr/bin/env bash

python -m spacy download en_core_web_sm
# Preprocess the CNN/DailyMail dataset
python data/dataset.py \
    --dataset cnn_dailymail \
    --dataset_variant 3.0.0 \
    --longtext_column article \
    --shorttext_column highlights \
    --guidance_type gsum \
