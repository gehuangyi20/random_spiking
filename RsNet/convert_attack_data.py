import os
import gzip
import json
import argparse
import numpy as np
import RsNet.utils as utils
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta
from RsNet.tf_config import CHANNELS_FIRST

parser = argparse.ArgumentParser(description='convert attack data from mnist gz format to pkl format')

requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-d', '--data_dir', help='attack dir', type=str, required=True)
requiredNamed.add_argument('-n', '--data_name', help='attack name', type=str, required=True)
requiredNamed.add_argument('-a', '--attack_name', help='attack file name', type=str, required=True)
requiredNamed.add_argument('--output_dir', help='output directory', type=str, required=True)
requiredNamed.add_argument('--output_file', help='output filename', type=str, required=True)
requiredNamed.add_argument('-s', '--set_name', help='attack set name', type=str, required=True)
requiredNamed.add_argument('--data_format', help='data_format', type=str, required=True)

args = parser.parse_args()

set_name = args.set_name
if set_name == 'mnist':
    model_meta = model_mnist_meta
elif set_name == 'fashion':
    model_meta = model_mnist_meta
elif set_name == "cifar10":
    model_meta = model_cifar10_meta
else:
    model_meta = None
    MODEL = None
    print("invalid data set name %s" % set_name)
    exit(0)

img_size = model_meta.width * model_meta.height * model_meta.channel

real_dir = os.path.join(args.data_dir, args.data_name)
config_fp = open(os.path.join(real_dir, "config.json"), "rb")
json_str = config_fp.read()
config_fp.close()
config = json.loads(json_str.decode())

count = int(config[args.attack_name + '-count'] / 3)

attack_adv = os.path.join(real_dir, config[args.attack_name + "-adv-img"])
fp_attack_adv = gzip.open(attack_adv, "rb")

# load data
buf = fp_attack_adv.read(count * img_size * 4)
data_attack_adv = np.frombuffer(buf, dtype=np.float32)
data_attack_adv = data_attack_adv.reshape(count, model_meta.width, model_meta.height, model_meta.channel)

if args.data_format == CHANNELS_FIRST:
    data_attack_adv = data_attack_adv.transpose([0, 3, 1, 2])

# close loaded file
fp_attack_adv.close()

# output data
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

utils.save_obj(data_attack_adv, name=args.output_file, directory=args.output_dir)
