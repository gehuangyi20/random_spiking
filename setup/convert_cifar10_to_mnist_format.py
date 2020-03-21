import os
import sys
import json
import gzip
import numpy as np

argc = len(sys.argv)
if argc < 3:
    print('usage: convert_cifar10_to_mnist_format [cifar 10 dir] [output dir]')
    sys.exit()

json_str = ""

cifar_dir = sys.argv[1]
out_dir = sys.argv[2]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if os.path.isfile(out_dir + "/config.json"):
    config_fp = open(out_dir + "/config.json", "rb")
    json_str = config_fp.read()
    config_fp.close()


if json_str == "":
    config = {}
else:
    config = json.loads(json_str.decode())

config['train-img'] = "train-img.gz"
config['train-label'] = "train-label.gz"
config['train-count'] = 50000
config['test-img'] = "test-img.gz"
config['test-label'] = "test-label.gz"
config['test-count'] = 10000


def load_batch(fpath):
    fp = open(fpath, "rb")
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        buf = fp.read(size)
        arr = np.frombuffer(buf, dtype=np.uint8)
        lab = arr[0]
        #img = arr[1:].reshape((3, 32, 32)).transpose((1, 2, 0))
        img = arr[1:].reshape((3, 1024)).transpose()

        labels.append(lab)
        images.append(img)
    return np.array(images), np.array(labels)


train_data = []
train_labels = []
for i in range(5):
    r, s = load_batch(cifar_dir + "/data_batch_" + str(i + 1) + '.bin')
    train_data.extend(r)
    train_labels.extend(s)

train_data = np.asarray(train_data)
train_labels = np.asarray(train_labels)

test_data, test_labels = load_batch(cifar_dir + "/test_batch.bin")

# save training data
fp_stream = gzip.open(out_dir + "/train-img.gz", "wb")
fp_label = gzip.open(out_dir + "/train-label.gz", "wb")

fp_stream.write(np.zeros([16], np.uint8).tobytes())
fp_label.write(np.zeros([8], np.uint8).tobytes())
fp_stream.write(train_data.tobytes())
fp_label.write(train_labels.tobytes())

fp_stream.close()
fp_label.close()

# save testing data
fp_stream = gzip.open(out_dir + "/test-img.gz", "wb")
fp_label = gzip.open(out_dir + "/test-label.gz", "wb")

fp_stream.write(np.zeros([16], np.uint8).tobytes())
fp_label.write(np.zeros([8], np.uint8).tobytes())
fp_stream.write(test_data.tobytes())
fp_label.write(test_labels.tobytes())

fp_stream.close()
fp_label.close()

# save config file
config_fp = open(out_dir + "/config.json", "wb")
config_str = json.dumps(config)
config_fp.write(config_str.encode())
config_fp.close()
