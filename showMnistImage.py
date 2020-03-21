import json
import sys
import gzip
import math
import os
import hashlib
import argparse
import pickle
import numpy as np
import tensorflow as tf

from PIL import Image
from RsNet.setup_mnist import MNIST
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta, model_cifar10L_meta
from RsNet.tf_config import setup_visibile_gpus, gpu_config

argc = len(sys.argv)
if argc < 6:
    print('usage: showMnistImage [dir] [name] [start_idx] [count] [train/test/attack/etc] '
          '[column optional, default 30] [dataset, default mnist] [step] [duplicate] [out_filename] [show_hash] '
          '[margin] [encrypt_config_path]')
    sys.exit()

parser = argparse.ArgumentParser(description='show adversarial images')
parser.add_argument('--dir', help='save data directory, required', type=str, default=None)
parser.add_argument('--data_dir', help='orig data directory, required', type=str, default=None)
parser.add_argument('--att_name', help='save adv data name, required', type=str, default=None)
parser.add_argument('--start_idx', help='start idx of all saved advs', type=int, default=0)
parser.add_argument('--count', help='how many adv examples are read', type=int, default=100)
parser.add_argument('--set_name', help='set name [mnist, fashion, cifar10, imagenet], required', type=str, default=None)
parser.add_argument('--col', help='how many images display on each row', type=int, default=30)
parser.add_argument('--step', help='walk step k, ex: read one image, and skip k-1 images', type=int, default=1)
parser.add_argument('--duplicate', help='show an image k times', type=int, default=1)
parser.add_argument('--output', help='output filename', type=str, default='')
parser.add_argument('--show_hash', help='whether to show the hash value of the output image', type=str, default='no')
parser.add_argument('--margin', help='margin space between two adjacent images', type=int, default=0)
parser.add_argument('--is_test', help='reference data is test data or training data.',
                    type=str, default='yes')
parser.add_argument('--show_diff', help='whether to show orig image and diff',
                    type=str, default='yes')
parser.add_argument('--batch_size', help='batch_size', type=int, default=500)


args = parser.parse_args()

real_dir = args.dir
att_name = args.att_name
start_idx = args.start_idx
count = args.count
set_name = args.set_name
is_test = args.is_test == 'yes'
col_len = args.col
step = args.step
duplicate = args.duplicate
out_filename = args.output
show_hash = args.show_hash == 'yes'
margin = args.margin
batch_size = args.batch_size
show_diff = args.show_diff == 'yes'

if set_name == 'mnist':
    model_meta = model_mnist_meta
    is_color = False
elif set_name == 'fashion':
    model_meta = model_mnist_meta
    is_color = False
elif set_name == "cifar10":
    model_meta = model_cifar10_meta
    is_color = True
elif set_name == "cifar10L":
    model_meta = model_cifar10L_meta
    is_color = False
else:
    model_meta = None
    print("invalid data set name %s" % att_name)
    exit(0)

print("%s %s %d" % (real_dir, start_idx, count))

config_fp = open(os.path.join(real_dir, "config.json"), "rb")
json_str = config_fp.read()
config_fp.close()
config = json.loads(json_str.decode())

count = min([count * step, config[att_name + '-count'] - start_idx])
count = math.ceil(count / step)

if count <= 0:
    print("start index (%d) is larger than the number of available images (%d)."
          % (start_idx, config[att_name + '-count']))
    exit()

count = count * duplicate
img_size = model_meta.width * model_meta.height * model_meta.channel
img_width = model_meta.width
img_height = model_meta.height
img_mode = "RGB" if is_color else "L"
default_color = (255, 255, 255) if is_color else 255
filename = os.path.join(real_dir, (config[att_name + "-img"]))
attack_idx = os.path.join(real_dir, att_name + '-idx.pkl')
fp = gzip.open(filename, "rb")
fp_attack_idx = open(attack_idx, 'rb')
print(img_size)
fp.seek(16 + start_idx*img_size)
data_idx = pickle.load(fp_attack_idx)

label_file = os.path.join(real_dir, (config[att_name + "-label"]))
fp_label = gzip.open(label_file, "rb")
fp_label = gzip.open(label_file, "rb")
fp_label.seek(8 + start_idx)

big_im = Image.new(img_mode, ((img_width+margin) * col_len * (3 if show_diff else 1) - margin,
                              (img_height+margin) * math.ceil(count/col_len)-margin), default_color)

col_width = len(str(col_len)) + 1
col_tmpl = "%" + str(col_width) + "d"
for i in range(col_len):
    print(col_tmpl % i, end='')

print()

for i in range(col_len*col_width):
    print("-", end='')

print()

setup_visibile_gpus(None)


with tf.Session(config=gpu_config) as sess:
    with tf.device('/cpu:0'):
        # load the original training and testing data
        data = MNIST(args.data_dir, '', model_meta=model_meta, normalize=False, batch_size=batch_size)

        data_type = 'test' if is_test else 'train'
        data_idx = data_idx[start_idx//step: start_idx//step + count]
        argsort_adv_img_idx = np.argsort(data_idx)
        back_argsort_adv_img_idx = np.argsort(argsort_adv_img_idx)
        data_ref, _, _ = data.get_data_by_idx(data_idx[argsort_adv_img_idx], data_type=data_type)
        data_ref = data_ref[back_argsort_adv_img_idx]
        data_ref = np.repeat(data_ref, duplicate, axis=0)

for i in range(0, count):
    row = math.floor(i/col_len)
    col = i % col_len * (3 if show_diff else 1)
    if i % duplicate == 0:
        buf = fp.read(img_size)
        buf_label = int.from_bytes(fp_label.read(1), byteorder='little', signed=False)

        # skip path
        if step > 1:
            fp.read(img_size*(step-1))
            fp_label.read(step-1)
    if show_diff:
        im_ref = Image.frombuffer(img_mode, (img_width, img_height), data_ref[i].astype(np.uint8), 'raw', img_mode, 0, 1)
        big_im.paste(im_ref, ((img_width + margin) * col, (img_height + margin) * row))
        im = Image.frombuffer(img_mode, (img_width, img_height), buf, 'raw', img_mode, 0, 1)
        big_im.paste(im, ((img_width + margin) * (col+1), (img_height + margin) * row))
        
        diff_buf = np.frombuffer(buf, np.uint8)
        diff_buf = np.reshape(diff_buf, [1, model_meta.width, model_meta.height, model_meta.channel])
        im_diff = np.round((diff_buf - data_ref[i]) / 255*127.5 + 127.5).astype(np.uint8)
        im_diff = Image.frombuffer(img_mode, (img_width, img_height), im_diff, 'raw', img_mode, 0, 1)
        big_im.paste(im_diff, ((img_width + margin) * (col+2), (img_height + margin) * row))
    else:
        im = Image.frombuffer(img_mode, (img_width, img_height), buf, 'raw', img_mode, 0, 1)
        big_im.paste(im, ((img_width + margin)*col, (img_height + margin)*row))
    print(col_tmpl % buf_label, end='')

    if col // (3 if show_diff else 1) + 1 == col_len:
        print()

if count % col_len:
    print()

fp.close()
fp_label.close()
fp_attack_idx.close()

if show_hash:
    im_data = big_im.tobytes()
    md5_sum = hashlib.md5()
    md5_sum.update(im_data)
    digest = md5_sum.hexdigest()
    print("md5 digest: ", digest)

if out_filename:
    if not os.path.exists(os.path.dirname(out_filename)):
        os.makedirs(os.path.dirname(out_filename))
    big_im.save(out_filename)
else:
    big_im.show()
