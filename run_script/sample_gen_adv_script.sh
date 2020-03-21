#!/bin/bash
dataset=$1

cp -a "${dataset}"_data/tmpl_att_data "${dataset}"_data/sample_attack_l2_nodropout
cp -a "${dataset}"_data/tmpl_att_data "${dataset}"_data/sample_attack_l2_rs1_nodropout

cp -a "${dataset}"_data/tmpl_att_data "${dataset}"_data/sample_attack_multi_l2_nodropout
cp -a "${dataset}"_data/tmpl_att_data "${dataset}"_data/sample_attack_multi_l2_rs1_nodropout

python3 RsNet/gen_script_att_and_verify.py --dir=run_script/"${dataset}"/ -c sample_att_single.json -o sh_sample_att_
python3 RsNet/gen_script_att_and_verify.py --dir=run_script/"${dataset}"/ -c sample_att_multi.json -o sh_sample_att_
