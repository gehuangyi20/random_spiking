## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified for MagNet's use.

import numpy as np
import gzip
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from RsNet.random_spiking.nn_ops import random_spike_sample_scaling, random_spike_sample_scaling_per_sample
from RsNet.utils import load_config
from RsNet.dataset_nn import model_mnist_meta
import RsNet.wide_residual_network as wrn
from RsNet.tf_config import CHANNELS_FIRST, CHANNELS_LAST


def extract_data(filename, num_images, model_meta, normalize=True,
                 input_data_format=CHANNELS_LAST, output_data_format=CHANNELS_LAST, boxmin=0, boxmax=1):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images * model_meta.width * model_meta.height * model_meta.channel)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if normalize:
            data = (data / 255) * (boxmax-boxmin) + boxmin
        if input_data_format == CHANNELS_LAST:
            data = data.reshape(num_images, model_meta.width, model_meta.height, model_meta.channel)
        else:
            data = data.reshape(num_images, model_meta.channel, model_meta.width, model_meta.height)
        if input_data_format != output_data_format:
            if output_data_format == CHANNELS_LAST:
                # Output requires channels_last.
                data = data.transpose([0, 2, 3, 1])
            else:
                # Output requires channels_first.
                data = data.transpose([0, 3, 1, 2])
        return data


def extract_labels(filename, num_images, model_meta, normalize=True):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    if normalize:
        return (np.arange(model_meta.labels) == labels[:, None]).astype(np.float32)
    else:
        return labels


class MNIST:
    def __init__(self, _dir, name, validation_size=5000, model_meta=model_mnist_meta, normalize=True,
                 input_data_format=CHANNELS_LAST, output_data_format=CHANNELS_LAST, train_size=0, train_sel_rand=False,
                 batch_size=100, boxmin=0, boxmax=1):
        assert input_data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        assert output_data_format in (CHANNELS_FIRST, CHANNELS_LAST)

        real_dir = _dir + "/" + name + "/"
        self.dir = _dir
        self.name = name
        self.real_dir = real_dir
        config = load_config(real_dir + "config.json")
        self.model_meta = model_meta
        self.batch_size = batch_size

        self.data_format = output_data_format
        self.normalize = normalize

        self.test_count = config['test-count']
        self.validation_count = validation_size
        if train_size == 0:
            self.train_count = config['train-count'] - validation_size
        else:
            self.train_count = min(train_size, config['train-count'] - validation_size)

        train_data = extract_data(real_dir+config['train-img'], config['train-count'], self.model_meta, normalize,
                                  input_data_format=input_data_format, output_data_format=output_data_format,
                                  boxmin=boxmin, boxmax=boxmax)
        train_labels = extract_labels(real_dir+config['train-label'], config['train-count'], self.model_meta, normalize)

        self.test_idx = np.arange(0, self.test_count)
        self.test_data = extract_data(real_dir+config['test-img'], config['test-count'], self.model_meta, normalize,
                                      input_data_format=input_data_format, output_data_format=output_data_format,
                                      boxmin=boxmin, boxmax=boxmax)
        self.test_labels = extract_labels(real_dir+config['test-label'], config['test-count'], self.model_meta, normalize)

        train_idx_orig = np.arange(0, config['train-count'])
        self.train_idx_orig = train_idx_orig
        self.train_data_orig = train_data
        self.train_labels_orig = train_labels
        self.test_idx_orig = self.test_idx
        self.test_data_orig = self.test_data
        self.test_labels_orig = self.test_labels

        if train_sel_rand:
            np.random.shuffle(train_idx_orig)

        self.validation_idx = train_idx_orig[:validation_size]
        self.validation_data = train_data[self.validation_idx]
        self.validation_labels = train_labels[self.validation_idx]
        self.train_idx = train_idx_orig[validation_size:validation_size + self.train_count]
        self.train_data = train_data[self.train_idx]
        self.train_labels = train_labels[self.train_idx]

        self.train_idx_ptr = 0
        self.validation_idx_ptr = 0
        self.test_idx_ptr = 0

    def data_len(self, data_type='test'):
        if data_type == 'train':
            return len(self.train_data)
        elif data_type == 'validation':
            return len(self.validation_data)
        else:
            return len(self.test_data)

    def next_batch(self, data_type='test'):
        if data_type == 'train':
            cur_data = self.train_data
            cur_label = self.train_labels
            cur_idx_ptr = self.train_idx_ptr
        elif data_type == 'validation':
            cur_data = self.validation_data
            cur_label = self.validation_labels
            cur_idx_ptr = self.validation_idx_ptr
        else:
            cur_data = self.test_data
            cur_label = self.test_labels
            cur_idx_ptr = self.test_idx_ptr

        new_idx_ptr = cur_idx_ptr + self.batch_size
        if new_idx_ptr >= self.train_count:
            new_idx_ptr = 0

        if data_type == 'train':
            self.train_idx_ptr = new_idx_ptr
        elif data_type == 'validation':
            self.validation_idx_ptr = new_idx_ptr
        else:
            self.test_idx_ptr = new_idx_ptr

        batch_data = cur_data[cur_idx_ptr:cur_idx_ptr+self.batch_size]
        batch_label = cur_label[cur_idx_ptr:cur_idx_ptr+self.batch_size]

        return batch_data, batch_label

    def get_data_by_idx(self, idx, data_type='test'):
        if data_type == 'train':
            cur_data = self.train_data
            cur_label = self.train_labels
        elif data_type == 'validation':
            cur_data = self.validation_data
            cur_label = self.validation_labels
        else:
            cur_data = self.test_data
            cur_label = self.test_labels

        data = cur_data[idx]
        label = cur_label[idx]

        return data, label, idx

    def apply_pre_idx(self, pre_idx):
        self.train_idx = pre_idx['train']
        self.train_data = self.train_data_orig[self.train_idx]
        self.train_labels = self.train_labels_orig[self.train_idx]
        self.train_count = len(self.train_idx)

        self.validation_idx = pre_idx['validate']
        self.validation_data = self.train_data_orig[self.validation_idx]
        self.validation_labels = self.train_labels_orig[self.validation_idx]
        self.validation_count = len(self.validation_idx)

        self.test_idx = pre_idx['test']
        self.test_data = self.test_data_orig[self.test_idx]
        self.test_labels = self.test_labels_orig[self.test_idx]
        self.test_count = len(self.test_idx)

    def get_idx(self):
        return {
            'train': self.train_idx,
            'validate': self.validation_idx,
            'test': self.test_idx
        }

    def append_train_data(self, _dir, data_name, count,
                          input_data_format=CHANNELS_LAST, output_data_format=CHANNELS_LAST, sel_rand=False):
        config = load_config(os.path.join(_dir, "config.json"))
        extra_data = extract_data(os.path.join(_dir, config[data_name+'-img']), config[data_name+'-count'],
                                  self.model_meta, self.normalize,
                                  input_data_format=input_data_format, output_data_format=output_data_format)
        extra_labels = extract_labels(os.path.join(_dir, config[data_name+'-label']), config[data_name+'-count'],
                                      self.model_meta, self.normalize)

        if sel_rand:
            idx_rand = np.arange(0, config[data_name+'-count'])
            np.random.shuffle(idx_rand)
            extra_data = extra_data[idx_rand]
            extra_labels = extra_labels[idx_rand]

        self.train_data = np.concatenate([self.train_data, extra_data[:count]])
        self.train_labels = np.concatenate([self.train_labels, extra_labels[:count]])

    @staticmethod
    def print():
        return "MNIST"


class MNISTModel:
    def __init__(self, restore, session=None, output_logits=True,
                 input_data_format=CHANNELS_LAST,
                 data_format=CHANNELS_FIRST, dropout=0.0, rand_params=None, is_batch=True, **kwargs):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        assert input_data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        assert data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        self.input_data_format = input_data_format
        self.data_format = data_format

        if rand_params is None:
            rand_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if input_data_format == CHANNELS_LAST:
            input_shape = (self.image_size, self.image_size, self.num_channels)
        else:
            input_shape = (self.num_channels, self.image_size, self.image_size)

        _input = Input(shape=input_shape)
        x = _input

        if self.data_format != input_data_format:
            if input_data_format == CHANNELS_LAST:
                # Computation requires channels_first.
                convert_layer = Lambda(lambda _x: tf.transpose(_x, [0, 3, 1, 2], name="transpose"))
            else:
                # Computation requires channels_last.
                convert_layer = Lambda(lambda _x: tf.transpose(_x, [0, 2, 3, 1], name="transpose"))
            x = convert_layer(x)

        def random_spike(x, sample_rate, scaling, is_batch=True):
            if is_batch:
                return random_spike_sample_scaling(x, sample_rate=sample_rate, scaling=scaling)
            else:
                return random_spike_sample_scaling_per_sample(x, sample_rate=sample_rate, scaling=scaling)

        x = Conv2D(32, (3, 3), padding="same", data_format=self.data_format)(x)
        x = Activation('relu')(x)
        x = Lambda(function=random_spike, arguments={
            "sample_rate": rand_params[0], "scaling": rand_params[1], "is_batch": is_batch})(x)
        x = Conv2D(32, (3, 3), padding="same", data_format=self.data_format)(x)
        x = Activation('relu')(x)
        x = Lambda(function=random_spike, arguments={
            "sample_rate": rand_params[2], "scaling": rand_params[3], "is_batch": is_batch})(x)
        x = MaxPooling2D(pool_size=(2, 2), data_format=self.data_format)(x)
        x = Lambda(function=random_spike, arguments={
            "sample_rate": rand_params[4], "scaling": rand_params[5], "is_batch": is_batch})(x)

        x = Conv2D(64, (3, 3), padding="same", data_format=self.data_format)(x)
        x = Activation('relu')(x)
        x = Lambda(function=random_spike, arguments={
            "sample_rate": rand_params[6], "scaling": rand_params[7], "is_batch": is_batch})(x)
        x = Conv2D(64, (3, 3), padding="same", data_format=self.data_format)(x)
        x = Activation('relu')(x)
        x = Lambda(function=random_spike, arguments={
            "sample_rate": rand_params[8], "scaling": rand_params[9], "is_batch": is_batch})(x)
        x = MaxPooling2D(pool_size=(2, 2), data_format=self.data_format)(x)
        x = Lambda(function=random_spike, arguments={
            "sample_rate": rand_params[10], "scaling": rand_params[11], "is_batch": is_batch})(x)

        x = Flatten()(x)
        x = Dense(200)(x)
        x = Activation('relu')(x)
        x = Lambda(function=random_spike, arguments={
            "sample_rate": rand_params[12], "scaling": rand_params[13], "is_batch": is_batch})(x)
        if dropout > 0:
            x = Dropout(dropout)(x, training=True)
        x = Dense(200)(x)
        x = Activation('relu')(x)
        x = Lambda(function=random_spike, arguments={
            "sample_rate": rand_params[14], "scaling": rand_params[15], "is_batch": is_batch})(x)
        fn_activation = None if output_logits is True else 'softmax'
        x = Dense(10, activation=fn_activation)(x)
        model = Model(_input, x)
        if restore is not None:
            model.load_weights(restore)

        self.model = model
        self.model.summary()

    def predict(self, data):
        return self.model(data)

    def load_weights(self, restore):
        self.model.load_weights(restore)


class FASHIONModel:
    def __init__(self, restore, session=None, output_logits=True,
                 input_data_format=CHANNELS_LAST,
                 data_format=CHANNELS_FIRST, dropout=0.0, rand_params=0, is_batch=True, **kwargs):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        assert input_data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        assert data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        self.input_data_format = input_data_format
        self.data_format = data_format

        if input_data_format == CHANNELS_LAST:
            input_shape = (self.image_size, self.image_size, self.num_channels)
        else:
            input_shape = (self.num_channels, self.image_size, self.image_size)

        model = wrn.create_wide_residual_network(input_shape, nb_classes=10, N=4, k=10, dropout=dropout,
                                                 output_logits=output_logits,
                                                 input_data_format=self.input_data_format,
                                                 data_format=self.data_format, rand_spike=rand_params)
        if restore is not None:
            model.load_weights(restore)

        self.model = model
        self.model.summary()

    def predict(self, data):
        return self.model(data)

    def load_weights(self, restore):
        self.model.load_weights(restore)


class CIFAR10Model:
    def __init__(self, restore, session=None, output_logits=True,
                 input_data_format=CHANNELS_LAST,
                 data_format=CHANNELS_FIRST, dropout=0.0, rand_params=0, is_batch=True, **kwargs):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        assert input_data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        assert data_format in (CHANNELS_FIRST, CHANNELS_LAST)
        self.input_data_format = input_data_format
        self.data_format = data_format

        if input_data_format == CHANNELS_LAST:
            input_shape = (self.image_size, self.image_size, self.num_channels)
        else:
            input_shape = (self.num_channels, self.image_size, self.image_size)

        model = wrn.create_wide_residual_network(input_shape, nb_classes=10, N=4, k=10, dropout=dropout,
                                                 output_logits=output_logits,
                                                 input_data_format=self.input_data_format,
                                                 data_format=self.data_format, rand_spike=rand_params)
        if restore is not None:
            model.load_weights(restore)

        self.model = model
        self.model.summary()

    def predict(self, data):
        return self.model(data)

    def load_weights(self, restore):
        self.model.load_weights(restore)


class SQModels:
    def __init__(self, model_list, ref_model):
        input_shape = np.array(model_list[0].inputs[0].get_shape().as_list())

        _input = Input(shape=input_shape[1:])
        x = _input

        idx = 0
        for model in model_list:
            x = model(x)
            idx += 1

        sq_model = Model(_input, x)
        self.model = sq_model

        self.model.summary()

        self.num_channels = ref_model.num_channels
        self.image_size = ref_model.image_size
        self.num_labels = ref_model.num_labels

        self.input_data_format = ref_model.input_data_format
        self.data_format = ref_model.data_format

    def predict(self, data):
        return self.model(data)
