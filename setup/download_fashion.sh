#!/bin/bash
output_dir=$1

if [ -z "$1" ] ;
then
    output_dir=fashion_data/data
fi

echo $output_dir
dir_name=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')

mkdir -p "/tmp/$dir_name"
wget -c http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -O "/tmp/$dir_name"/train-img.gz
wget -c http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -O "/tmp/$dir_name"/train-label.gz
wget -c http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -O "/tmp/$dir_name"/test-img.gz
wget -c http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -O "/tmp/$dir_name"/test-label.gz

echo  "/tmp/$dir_name"

echo '{"test-img": "test-img.gz",
"test-count": 10000, "name": "data",
"train-count": 60000, "train-label": "train-label.gz",
"train-img": "train-img.gz", "test-label": "test-label.gz"}' > "/tmp/$dir_name/config.json"

mkdir -p $output_dir
mv "/tmp/${dir_name}/"* $output_dir

rm -r "/tmp/$dir_name"

tmpl_dir=$output_dir/../tmpl_att_data
target_dir=../$(basename $output_dir)
mkdir -p $tmpl_dir
cp -a $output_dir/config.json $tmpl_dir/config.json
ln -sf "$target_dir"/train-img.gz $tmpl_dir/train-img.gz
ln -sf "$target_dir"/train-label.gz $tmpl_dir/train-label.gz
ln -sf "$target_dir"/test-img.gz $tmpl_dir/test-img.gz
ln -sf "$target_dir"/test-label.gz $tmpl_dir/test-label.gz
