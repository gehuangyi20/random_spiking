#!/bin/bash
dataset=$1

python3 RsNet/gen_script_att_and_verify.py --dir=run_script/"${dataset}"/ \
-c sample_verify_trans_single.json -o sh_sample_verify_trans_single_
python3 RsNet/gen_script_att_and_verify.py --dir=run_script/"${dataset}"/ \
-c sample_verify_trans_multi.json -o sh_sample_verify_trans_multi_

bash run_script/"${dataset}"/sh_sample_verify_trans_multi_rs1_nodropout.sh
bash run_script/"${dataset}"/sh_sample_verify_trans_multi_standard.sh

bash run_script/"${dataset}"/sh_sample_verify_trans_single_rs1_nodropout.sh
bash run_script/"${dataset}"/sh_sample_verify_trans_single_standard.sh
