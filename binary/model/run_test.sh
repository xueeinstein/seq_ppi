#!/bin/bash
emb=$1
model=$2
gpu=$3

cd lasagna
for species in "mouse" "fly" "worm" "ecoli" "yeast"
do
    CUDA_VISIBLE_DEVICES=${gpu} python3 rcnn_test.py ../data/pairs/${species}_test.tsv -1 results/human_${model}.txt \
                        ../${emb}/${species}_test.h5 100 \
                        ../../../data/${species}_dict.tsv results/test_human_${model}_${species}.txt
done
