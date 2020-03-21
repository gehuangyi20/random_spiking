#!/bin/bash
dataset=$1

mkdir -p result_"${dataset}"/data_sum/sample_single
mkdir -p result_"${dataset}"/data_sum/sample_multi
cp -a run_script/"${dataset}"/sample_white_box_single.json result_"${dataset}"/data_sum/sample_single/
cp -a run_script/"${dataset}"/sample_white_box_multi.json result_"${dataset}"/data_sum/sample_multi/

cd RsNet/
python3 compute_adv_diff_summary_mthd.py --dir=../result_"${dataset}"/data_sum/sample_single/ --config=sample_white_box_single.json -j 10
python3 compute_adv_diff_summary_mthd.py --dir=../result_"${dataset}"/data_sum/sample_multi/ --config=sample_white_box_multi.json -j 10
