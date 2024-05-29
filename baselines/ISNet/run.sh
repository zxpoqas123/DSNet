#!/bin/bash
epochs=40
lr=1e-3
loss_type=CE
optimizer_type=Adam
batch_size=32
gpu_id=0

set -ue

# training the network
#feature='egemaps'
feature='FBank'
#feature='WavLM'
#feature='mfcc'
#feature='spectrogram'
max_len=6
condition=all
seed=42
dataset_type=MSP-IMPROV
for mode in isnet
do
for fold in {0..2}
do
    store_root=./seed_$seed/$mode+$dataset_type+$max_len\s+$feature+lr$lr+batch_size$batch_size+$loss_type+$optimizer_type/
    echo "============training fold $fold============"
    python ./train.py \
        --seed $seed \
        --max-len $max_len \
        --lr $lr \
        --epochs $epochs \
        --fold $fold \
        --root $store_root \
        --loss-type $loss_type \
        --batch-size $batch_size \
        --dataset-type $dataset_type \
        --optimizer-type $optimizer_type \
        --device-number $gpu_id \
        --feature $feature \
        --condition $condition \
        --mode $mode
done
done