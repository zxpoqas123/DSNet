#!/bin/bash
epochs=50
lr=1e-3
loss_type=CE
optimizer_type=Adam
batch_size=32
gpu_id=0

set -ue

# training the network
seed=42
#feature='egemaps'
feature='FBank'
#feature='WavLM'
#feature='mfcc'
max_len=6
condition=all
dataset_type=MSP-IMPROV

python test.py --seed $seed --device-number $gpu_id --feature $feature \
            --max-len $max_len --lr $lr --batch-size $batch_size \
            --optimizer-type $optimizer_type --dataset-type $dataset_type --condition $condition 