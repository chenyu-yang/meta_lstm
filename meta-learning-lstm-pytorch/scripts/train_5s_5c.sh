#!/bin/bash
#
# For 5-shot, 5-class training
# Hyper-parameters follow https://github.com/twitter/meta-learning-lstm

python main.py --mode train \
               --n-shot 500 \
               --n-eval 500 \
               --n-class 5 \
               --input-size 4 \
               --input-dim 10000 \
               --hidden-size 20 \
               --lr 1e-3 \
               --episode 50000 \
               --episode-val 100 \
               --epoch 8 \
               --batch-size 2 \
               --grad-clip 0.25 \
               --bn-momentum 0.95 \
               --bn-eps 1e-3 \
               --data WGM \
               --data-root C:/Users/22739/Downloads/frame_classification/frame_classification/data_processed \
               --pin-mem True \
               --log-freq 50 \
               --val-freq 1000 \
               --csv_train C:/Users/22739/Downloads/frame_classification/frame_classification/data_processed/train.csv \
               --csv_val C:/Users/22739/Downloads/frame_classification/frame_classification/data_processed/val.csv \
               --csv_test C:/Users/22739/Downloads/frame_classification/frame_classification/data_processed/test.csv