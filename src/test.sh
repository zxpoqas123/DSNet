#!/bin/bash
epochs=80
lr=1e-3
loss_type=CE
optimizer_type=Adam
batch_size=32
gpu_id=1

set -ue

# training the network
seed=12
#feature='egemaps'
feature='FBank'
#feature='WavLM'
#feature='mfcc'
max_len=6
condition=all
dataset_type=IEMOCAP_4
mode=sdser

python test.py --seed $seed --device-number $gpu_id --feature $feature \
            --max-len $max_len --lr $lr --batch-size $batch_size --mode $mode \
            --optimizer-type $optimizer_type --dataset-type $dataset_type --condition $condition

dataset_type=MSP-IMPROV
mode=sdser

python test.py --seed $seed --device-number $gpu_id --feature $feature \
            --max-len $max_len --lr $lr --batch-size $batch_size --mode $mode \
            --optimizer-type $optimizer_type --dataset-type $dataset_type --condition $condition             