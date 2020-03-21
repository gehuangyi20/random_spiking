import os
import gzip
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from RsNet.setup_mnist import MNIST
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta
import tensorflow as tf
from RsNet.tf_config import setup_visibile_gpus, gpu_config


def norm_l0(x1, x2):
    return np.sum((x1-x2) != 0)


def norm_l1(x1, x2):
    return np.sum(np.abs(x1-x2))


def norm_l2(x1, x2):
    return np.sum(np.square(x1-x2)) ** 0.5


def norm_l_inf(x1, x2):
    return np.max(np.abs(x1 - x2))


def l1_arctan(x1, x2):
    return np.sum(np.abs(np.arctanh(x1*1.999999) - np.arctanh(x2*1.999999)))


parser = argparse.ArgumentParser(description='compute adv diff using l_p')
parser.add_argument('--dir', help='save data directory, required', type=str, default=None)
parser.add_argument('--name', help='save data name, required', type=str, default=None)
parser.add_argument('--attack_name', help='attack name, required', type=str, default=None)
parser.add_argument('--set_name', help='set name [mnist, fashion, cifar10], required', type=str, default=None)
parser.add_argument('--is_normalize', help='whether to normalize the data, converting [0-255] to [0-1]',
                    type=str, default='yes')
parser.add_argument('--is_test', help='reference data is test data or training data.',
                    type=str, default='yes')
parser.add_argument('--batch_size', help='batch_size', type=int, default=500)
parser.add_argument('--boxmin', help='model input image value min', type=float, default=0)
parser.add_argument('--boxmax', help='model input image value max', type=float, default=1.0)
parser.add_argument('--image_size', help='image dimension: default 224', type=int, default=224)
parser.add_argument('--preprocess_name', help='preprocess function name', type=str, default=None)
parser.add_argument('--intra_op_parallelism_threads',
                    help="""Nodes that can use multiple threads to 
                    parallelize their execution will schedule the 
                    individual pieces into this pool.
                    Default value 1 avoid pool of Eiden threads""",
                    type=int, default=1)
parser.add_argument('--inter_op_parallelism_threads', help="""All ready nodes are scheduled in this pool.""",
                    type=int, default=5)
parser.add_argument('--num_parallel_calls', help="The level of parallelism for data "
                                                 "preprocessing across multiple CPU cores",
                    type=int, default=5)

parser.add_argument('--out_dir', help='save directory', type=str, default=None)

args = parser.parse_args()

model_dir = args.dir
model_name = args.name
attack_name = args.attack_name
set_name = args.set_name
normalize = args.is_normalize == 'yes'
is_test = args.is_test == 'yes'
out_dir = args.out_dir
batch_size = args.batch_size
boxmin = args.boxmin
boxmax = args.boxmax
image_size = args.image_size

if set_name == 'mnist':
    model_meta = model_mnist_meta
elif set_name == 'fashion':
    model_meta = model_mnist_meta
elif set_name == "cifar10":
    model_meta = model_cifar10_meta
elif set_name == "cifar10magnet":
    model_meta = model_cifar10_meta
else:
    model_meta = None
    print("invalid data set name %s" % set_name)
    exit(0)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

img_size = model_meta.width * model_meta.height * model_meta.channel
img_width = model_meta.width
img_height = model_meta.height
img_channel = model_meta.channel
img_labels = model_meta.labels
img_labels_val = np.arange(model_meta.labels)

if 0 <= img_labels <= 255:
    label_data_type = np.uint8
else:
    label_data_type = np.uint16
label_data_size = np.dtype(label_data_type).itemsize

real_dir = os.path.join(model_dir, model_name)
config_fp = open(os.path.join(real_dir, "config.json"), "rb")
json_str = config_fp.read()
config_fp.close()
config = json.loads(json_str.decode())

count = int(config[attack_name + '-count'] / 3)

attack_adv = os.path.join(real_dir, config[attack_name + "-adv-img"])
attack_adv_img = os.path.join(real_dir, config[attack_name + "-img"])
attack_idx = os.path.join(real_dir, attack_name + '-idx.pkl')
fp_attack_adv = gzip.open(attack_adv, "rb")
fp_attack_adv_img = gzip.open(attack_adv_img, "rb")
fp_attack_idx = open(attack_idx, 'rb')

fp_attack_adv_img.seek(16)

data_idx = pickle.load(fp_attack_idx)

setup_visibile_gpus(None)

gpu_thread_count = 2
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

with tf.Session(config=gpu_config) as sess:
    with tf.device('/cpu:0'):
        # load the original training and testing data
        data = MNIST(model_dir, model_name, model_meta=model_meta, normalize=False, batch_size=batch_size)

        data_type = 'test' if is_test else 'train'

    l0_dist = []
    l1_dist = []
    l2_dist = []
    l_inf_dist = []

    for i in range(0, count):
        if i % batch_size == 0:
            data_ref, _, _ = data.get_data_by_idx(data_idx[i: i+batch_size], data_type=data_type)
            data_ref = (data_ref - boxmin) / (boxmax - boxmin)
        original_img = data_ref[i % batch_size]
        # load float data
        buf = fp_attack_adv.read(img_size * 4)
        float_img = np.frombuffer(buf, dtype=np.float32).reshape(img_width, img_height, img_channel)
        # read adv example in image format
        buf = fp_attack_adv_img.read(img_size * 3)
        batch_adv_img = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

        batch_adv_img = batch_adv_img.reshape(3, img_width, img_height, img_channel)
        floor_img = batch_adv_img[0]
        ceil_img = batch_adv_img[1]
        round_img = batch_adv_img[2]

        # attack fail for this image
        if np.sum(float_img) == 0:
            continue

        diff_data = original_img - round_img
        if normalize:
            diff_data = diff_data / 255
        diff_data = diff_data.reshape([1, -1])
        l0_dist.extend(np.linalg.norm(diff_data, 0, axis=1))
        l1_dist.extend(np.linalg.norm(diff_data, 1, axis=1))
        l2_dist.extend(np.linalg.norm(diff_data, 2, axis=1))
        l_inf_dist.extend(np.linalg.norm(diff_data, np.inf, axis=1))


    # close loaded file
    fp_attack_adv.close()
    fp_attack_adv_img.close()
    fp_attack_idx.close()


# output summary
def gen_summary(name, _data, _valid, _count):
    return "%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\n" % \
           (name, np.mean(_data), np.std(_data), np.median(_data), np.min(_data), np.max(_data), _valid, _count)


valid_count = len(l0_dist)
fp_summary = open(os.path.join(out_dir, "summary.csv"), 'wb')
fp_summary.write("category\tmean\tstd\tmedian\tmin\tmax\tvalid\tall\n".encode())

fp_summary.write(gen_summary("l0", l0_dist, valid_count, count).encode())
fp_summary.write(gen_summary("l1", l1_dist, valid_count, count).encode())
fp_summary.write(gen_summary("l2", l2_dist, valid_count, count).encode())
fp_summary.write(gen_summary("l_inf", l_inf_dist, valid_count, count).encode())
fp_summary.close()

# output raw diff
fp_raw = open(os.path.join(out_dir, "raw.csv"), 'wb')
fp_raw.write("l0\tl1\tl2\tl_inf\n".encode())
for i in range(valid_count):
    fp_raw.write(("%.4f\t%.4f\t%.4f\t%.4f\n" % (l0_dist[i], l1_dist[i], l2_dist[i], l_inf_dist[i])).encode())
fp_raw.close()

# plot histogram
fig = plt.figure(figsize=(20, 5))

ax0 = fig.add_subplot(1, 4, 1)
ax1 = fig.add_subplot(1, 4, 2)
ax2 = fig.add_subplot(1, 4, 3)
ax_inf = fig.add_subplot(1, 4, 4)

ax0.set_xlabel("distortion")
ax0.set_ylabel("count")
ax0.set_title('l0')
ax0.hist(l0_dist)

ax1.set_xlabel("distortion")
ax1.set_ylabel("count")
ax1.set_title('l1')
ax1.hist(l1_dist)

ax2.set_xlabel("distortion")
ax2.set_ylabel("count")
ax2.set_title('l2')
ax2.hist(l2_dist)

ax_inf.set_xlabel("distortion")
ax_inf.set_ylabel("count")
ax_inf.set_title('l_inf')
ax_inf.hist(l_inf_dist)

fig.suptitle(out_dir)

fig.savefig(os.path.join(out_dir, "diff.pdf"))
