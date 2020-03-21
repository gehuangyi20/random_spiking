from RsNet.setup_mnist import MNIST
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta, model_cifar20_meta, model_cifar100_meta
import sys
import os
import json
import gzip
import utils
import numpy as np

argc = len(sys.argv)
if argc < 6:
    print('usage: partition_dataset [data_dir] [data_name] [partition_name] [number of partition] '
          '[disjoint, 1 or 0] [set name, default mnist]')
    sys.exit()
data_dir = sys.argv[1]
data_name = sys.argv[2]
partition_name = sys.argv[3]
partition_count = int(sys.argv[4])
disjoint = True if int(sys.argv[5]) == 1 else False
set_name = "mnist" if argc < 7 else sys.argv[6]
is_cifar10 = False if set_name == 'mnist' or set_name == 'fashion' else True

if set_name == 'mnist' or set_name == 'fashion':
    model_meta = model_mnist_meta
elif set_name == "cifar10":
    model_meta = model_cifar10_meta
elif set_name == "cifar20":
    model_meta = model_cifar20_meta
elif set_name == "cifar100":
    model_meta = model_cifar100_meta
else:
    model_meta = None
    print("invalid data set name %s" % set_name)
    exit(0)

data_set = MNIST(data_dir, data_name, validation_size=0, model_meta=model_meta, normalize=False)
sample_size = data_set.train_count // partition_count

for i in range(partition_count):
    cur_partition_name = partition_name + str(i)
    cur_partition_dir = os.path.join(data_dir, cur_partition_name)
    if not os.path.exists(cur_partition_dir):
        os.makedirs(cur_partition_dir)

    # save json configuration
    config = {
        "name": cur_partition_name,
        "train-img": "train-img.gz",
        "train-label": "train-label.gz",
        "train-count": sample_size,
        "test-img": "test-img.gz",
        "test-label": "test-label.gz",
        "test-count": data_set.test_count,
    }
    config_fp = open(os.path.join(cur_partition_dir, "config.json"), "wb")
    config_str = json.dumps(config)
    config_fp.write(config_str.encode())
    config_fp.close()

    # save train image and label
    cur_train_img_fp = gzip.open(os.path.join(cur_partition_dir, config["train-img"]), "wb")
    cur_train_lbl_fp = gzip.open(os.path.join(cur_partition_dir, config["train-label"]), "wb")
    # fill 16 and 8 zero bytes at the beginning of the file
    cur_train_img_fp.write(np.zeros([16], np.uint8).tobytes())
    cur_train_lbl_fp.write(np.zeros([8], np.uint8).tobytes())

    if disjoint:
        cur_train_img = data_set.train_data[i*sample_size: (i+1)*sample_size]
        cur_train_lbl = data_set.train_labels[i*sample_size: (i+1)*sample_size]
        order = np.arange(i*sample_size, (i+1)*sample_size)
    else:
        order = np.arange(data_set.train_count)
        np.random.shuffle(order)
        order = order[0:sample_size]
        cur_train_img = data_set.train_data[order]
        cur_train_lbl = data_set.train_labels[order]

    utils.save_obj(order, name="train-idx", directory=cur_partition_dir)
    cur_train_img_fp.write(cur_train_img.astype(np.uint8).tobytes())
    cur_train_lbl_fp.write(cur_train_lbl.astype(np.uint8).tobytes())
    cur_train_img_fp.close()
    cur_train_lbl_fp.close()

    # save test image and label
    cur_test_img_fp = gzip.open(os.path.join(cur_partition_dir, config["test-img"]), "wb")
    cur_test_lbl_fp = gzip.open(os.path.join(cur_partition_dir, config["test-label"]), "wb")
    # fill 16 and 8 zero bytes at the beginning of the file
    cur_test_img_fp.write(np.zeros([16], np.uint8).tobytes())
    cur_test_lbl_fp.write(np.zeros([8], np.uint8).tobytes())
    # save data
    cur_test_img_fp.write(data_set.test_data.astype(np.uint8).tobytes())
    cur_test_lbl_fp.write(data_set.test_labels.astype(np.uint8).tobytes())
    cur_test_img_fp.close()
    cur_test_lbl_fp.close()

    print("create partition: ", cur_partition_name)
