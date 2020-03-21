## utils.py -- utility functions
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import hashlib
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict


def prepare_data(dataset, idx):
    """
    Extract data from index.

    dataset: Full, working dataset. Such as MNIST().
    idx: Index of test examples that we care about.
    return: X, targets, Y
    """
    return dataset.test_data[idx], dataset.test_labels[idx], np.argmax(dataset.test_labels[idx], axis=1)


def save_obj(obj, name, directory='./attack_data/'):
    with open(os.path.join(directory, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, directory='./attack_data/'):
    if name.endswith(".pkl"): name = name[:-4]
    with open(os.path.join(directory, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def save_cache(info, data, directory='./attack_data/', hash_alg='sha256'):
    info_str = json.dumps(info, sort_keys=True, separators=(',', ':'))
    h = hashlib.new(hash_alg, info_str.encode())
    hash_val = h.hexdigest()
    cache = {
        "hash_id": hash_val,
        "hash_alg": hash_alg,
        "info": info,
        "data": data
    }

    if not os.path.exists(directory):
        os.makedirs(directory)
    fp = open(os.path.join(directory, hash_val), "wb")
    pickle.dump(cache, fp, pickle.HIGHEST_PROTOCOL)
    fp.close()
    return hash_val


def load_cache(info, directory='./attack_data/', hash_alg='sha256'):
    info_str = json.dumps(info, sort_keys=True, separators=(',', ':'))
    h = hashlib.new(hash_alg, info_str.encode())
    hash_val = h.hexdigest()
    filename = os.path.join(directory, hash_val)
    if not os.path.isfile(filename):
        return None

    fp = open(filename, "rb")
    cache = pickle.load(fp)
    fp.close()

    return cache


def load_model_idx(path):
    if path.endswith(".idx"):
        pass
    else:
        path += ".idx"
    if not os.path.isfile(path):
        return None
    fp = open(path, 'rb')
    idx = pickle.load(fp)
    fp.close()
    return idx


def save_model_idx(path, data):
    if path.endswith(".idx"):
        pass
    else:
        path += ".idx"

    fp = open(path, 'wb')
    idx = data.get_idx()
    pickle.dump(idx, fp, pickle.HIGHEST_PROTOCOL)
    fp.close()
    return idx


def softmax_cross_entropy_with_logits(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)


def load_json(path):
    if not os.path.isfile(path):
        return None
    json_fp = open(path, "rb")
    json_str = json_fp.read()
    json_fp.close()
    # do not use edict since array type json is not a dictionary
    config = json.loads(json_str.decode())
    return config


def save_json(path, data, indent=None):
    config_fp = open(path, "wb")
    config_str = json.dumps(data, indent=indent)
    config_fp.write(config_str.encode())
    config_fp.close()


def get_num_records(filenames):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count

    print(filenames)
    nfile = len(filenames)
    return (count_records(filenames[0]) * (nfile - 1) +
            count_records(filenames[-1]))


def load_config(filename):
    config_fp = open(filename, "rb")
    json_str = config_fp.read()
    config_fp.close()
    config = edict(json.loads(json_str.decode()))
    return config
