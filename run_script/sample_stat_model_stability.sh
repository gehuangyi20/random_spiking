#!/bin/bash
dataset=$1

if [ "$dataset" == "mnist" ] ;
then

    python3 RsNet/rand_dir_summary.py --config=config.json --output=mnist --dir=result_mnist/models_noise_sample \
    --marker_size=5 --markeredgewidth=0.9 --capsize=2 --legend_col=9 -H 2.8 -W 4 --y_low=0.945 --y_high=1.003 \
    --y_unchg='Unchanged Predictions (\%)' --y_unchg_scale='percent' --x_text='Amount of Guassian Noise Added ($L_2$ distance)'

    python3 RsNet/rand_dir_summary.py --config=config.json --reverse_x=yes --output=mnist \
    --dir=result_mnist/models_jpeg_sample --marker_size=5 --markeredgewidth=0.9 --capsize=2 --legend_col=9 -H 2.8 -W 4 \
    --y_low=0.945 --y_high=1.003 --y_unchg='Unchanged Predictions (\%)' --y_unchg_scale='percent' --x_text='JPEG Compression Quality'

elif [ "$dataset" == "fashion" ];
then

    python3 RsNet/rand_dir_summary.py --config=config.json --output=fashion --dir=result_fashion/models_noise_sample \
    --marker_size=6 --legend_col=9 -H 2.8 -W 4 --y_low=-0.02 --y_high=1.02 \
    --y_unchg='Unchanged Predictions (\%)' --y_unchg_scale='percent' --x_text='Amount of Guassian Noise Added ($L_2$ distance)'

    python3 RsNet/rand_dir_summary.py --config=config.json --reverse_x=yes --output=fashion \
    --dir=result_fashion/models_jpeg_sample --marker_size=6 --legend_col=9 -H 2.8 -W 4 \
    --y_low=-0.02 --y_high=1.02 --y_unchg='Unchanged Predictions (\%)' --y_unchg_scale='percent' --x_text='JPEG Compression Quality'

else

    python3 RsNet/rand_dir_summary.py --config=config.json --output=cifar --dir=result_cifar/models_noise_sample \
    --marker_size=6 --legend_col=10 -H 2.8 -W 4 --y_low=-0.02 --y_high=1.02 \
    --y_unchg='Unchanged Predictions (\%)' \--y_unchg_scale='percent' --x_text='Amount of Guassian Noise Added ($L_2$ distance)'

    python3 RsNet/rand_dir_summary.py --config=config.json --reverse_x=yes --output=cifar \
    --dir=result_cifar/models_jpeg_sample --marker_size=6 --legend_col=10 -H 2.8 -W 4 \
    --y_low=-0.02 --y_high=1.02 --y_unchg='Unchanged Predictions (\%)' --y_unchg_scale='percent' --x_text='JPEG Compression Quality'

fi
