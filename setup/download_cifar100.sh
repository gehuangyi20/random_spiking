#!/bin/bash
output_dir=$1

if [ -z "$1" ] ;
then
    output_dir=cifar100_data/data
fi

echo $output_dir
dir_name=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')
wget -c https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -P "/tmp/$dir_name"
tar -xvzf "/tmp/${dir_name}/cifar-100-binary.tar.gz" -C "/tmp/$dir_name"

python3 setup/convert_cifar100_to_mnist_format.py "/tmp/${dir_name}/cifar-100-binary/" no $output_dir

rm -r "/tmp/$dir_name"

tmpl_dir=$output_dir/../tmpl_att_data
target_dir=../$(basename $output_dir)
mkdir -p $tmpl_dir
cp -a $output_dir/config.json $tmpl_dir/config.json
ln -sf "$target_dir"/train-img.gz $tmpl_dir/train-img.gz
ln -sf "$target_dir"/train-label.gz $tmpl_dir/train-label.gz
ln -sf "$target_dir"/test-img.gz $tmpl_dir/test-img.gz
ln -sf "$target_dir"/test-label.gz $tmpl_dir/test-label.gz
