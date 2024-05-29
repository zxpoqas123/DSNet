#!/bin/bash
epochs=80
lr=1e-3
loss_type=CE
optimizer_type=Adam
batch_size=32
gpu_id=2

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

python test_MSP2IEMOCAP.py --seed $seed --device-number $gpu_id --feature $feature \
            --max-len $max_len --lr $lr --batch-size $batch_size \
            --optimizer-type $optimizer_type --dataset-type $dataset_type --condition $condition 

dataset_type=IEMOCAP_4

python test_IEMOCAP2MSP.py --seed $seed --device-number $gpu_id --feature $feature \
            --max-len $max_len --lr $lr --batch-size $batch_size \
            --optimizer-type $optimizer_type --dataset-type $dataset_type --condition $condition             