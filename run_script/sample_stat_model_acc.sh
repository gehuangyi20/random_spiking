#!/bin/bash
dataset=$1

echo '["0.csv", "1.csv", "2.csv", "3.csv"]' > "result_$dataset/models_acc_sample/list.json"
echo '[{"name": "standard", "dir": "standard"},
{"name": "rs1_nodropout", "dir": "rs1_nodropout"}]' > "result_$dataset/models_acc_sample/config.json"

python3 RsNet/verify_models_sum.py --dir "result_$dataset/models_acc_sample/" \
--config config.json --list list.json --output summary.csv

