import json
import numpy as np
import os
import gzip
import argparse
from enum import Enum
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta


class Channel(Enum):
    channels_first = 'channels_first'
    channels_last = 'channels_last'

    def __str__(self):
        return self.value


parser = argparse.ArgumentParser(description='Convert channels_first or channels_last between float images')

parser.add_argument('--is_cifar', help='Boolean, whether the data is cifar or not', default=False, type=bool)

requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-d', '--dir', help='attack dir', type=str, required=True)
requiredNamed.add_argument('-n', '--name', help='attack name', type=str, required=True)
requiredNamed.add_argument('-s', '--set_name', help='attack set name', type=str, required=True)
requiredNamed.add_argument('--input_format', type=Channel, choices=list(Channel), required=True)


args = parser.parse_args()

model_meta = model_cifar10_meta if args.is_cifar else model_mnist_meta

real_dir = os.path.join(args.dir, args.name)
config_fp = open(os.path.join(real_dir, "config.json"), "rb")
json_str = config_fp.read()
config_fp.close()
config = json.loads(json_str.decode())

count = int(config[args.set_name + '-count'] / 3)

img_size = model_meta.width * model_meta.height * model_meta.channel
img_width = model_meta.width
img_height = model_meta.height
img_channel = model_meta.channel

attack_adv = os.path.join(real_dir, config[args.set_name + "-adv-img"])
fp_attack_adv = gzip.open(attack_adv, "rb")

# load data
buf = fp_attack_adv.read(count * img_size * 4)
data_attack_adv = np.frombuffer(buf, dtype=np.float32)

if args.input_format == Channel.channels_last:
    # convert data format from channels_last to channels_first
    data_attack_adv = data_attack_adv.reshape(count, img_width, img_height, img_channel)
    data_attack_adv = data_attack_adv.transpose([0, 3, 1, 2])
else:
    # convert data format from channels_first to channels_last
    data_attack_adv = data_attack_adv.reshape(count, img_channel, img_width, img_height)
    data_attack_adv = data_attack_adv.transpose([0, 2, 3, 1])

fp_attack_adv.close()

# save data
fp_attack_adv = gzip.open(attack_adv, "wb")
fp_attack_adv.write(data_attack_adv.astype(np.float32).tobytes())
fp_attack_adv.close()
