#!/bin/bash
dataset=$1

dir_name=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')
mkdir -p "/tmp/$dir_name"

if [ "$dataset" == "mnist" ] ;
then
    wget -c https://github.com/gehuangyi20/random_spiking/releases/download/sample_model/result_mnist.zip -O "/tmp/$dir_name"/result_mnist.zip
    unzip "/tmp/$dir_name"/result_mnist.zip -d .
else
    wget -c https://github.com/gehuangyi20/random_spiking/releases/download/sample_model/"${dataset}"_wrn_28_10_standard.zip -O "/tmp/$dir_name"/"${dataset}"_wrn_28_10_standard.zip
    wget -c https://github.com/gehuangyi20/random_spiking/releases/download/sample_model/"${dataset}"_wrn_28_10_rs1_nodropout.zip -O "/tmp/$dir_name"/"${dataset}"_wrn_28_10_rs1_nodropout.zip
    mkdir -p result_"${dataset}"/models_sample/
    unzip "/tmp/$dir_name/${dataset}"_wrn_28_10_standard.zip -d result_"${dataset}"/models_sample/
    unzip "/tmp/$dir_name/${dataset}"_wrn_28_10_rs1_nodropout.zip -d result_"${dataset}"/models_sample/
fi

rm -r "/tmp/$dir_name"
