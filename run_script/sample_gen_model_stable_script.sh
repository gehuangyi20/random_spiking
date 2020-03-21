#!/bin/bash
dataset=$1

python3 RsNet/gen_script_att_and_verify.py --dir=run_script/"${dataset}"/ -c sample_stability_jpeg.json -o sh_sample_stability_
python3 RsNet/gen_script_att_and_verify.py --dir=run_script/"${dataset}"/ -c sample_stability_noise.json -o sh_sample_stability_

mkdir -p result_"${dataset}"/models_jpeg_sample/
mkdir -p result_"${dataset}"/models_noise_sample/


jpeg_config='{
    "model_name": ["standard", "rs1_nodropout"],
    "model_name_st": ["Standard", "RS-1"],
    "l2": [
        [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    ],
    "xticks": [0, 20, 40, 60, 80, 100],
    "xticklabels": [100, 80, 60, 40, 20, 0]
}'

echo ${jpeg_config} > "result_${dataset}/models_jpeg_sample/config.json"


if [ "$dataset" == "mnist" ] ;
then
    noise_config='{
        "model_name": ["standard", "rs1_nodropout"],
        "model_name_st": ["Standard", "RS-1"],
        "l2": [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5]
        ]
    }'
else
    noise_config='{
        "model_name": ["standard", "rs1_nodropout"],
        "model_name_st": ["Standard", "RS-1"],
        "l2": [
            [0, 0.5, 1, 1.5, 2, 2.5],
            [0, 0.5, 1, 1.5, 2, 2.5]
        ]
    }'
fi

echo ${noise_config} > "result_${dataset}/models_noise_sample/config.json"
