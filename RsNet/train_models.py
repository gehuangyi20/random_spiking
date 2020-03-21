## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified for the needs of MagNet.

import os
import argparse
import utils
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from RsNet.setup_mnist import MNIST, MNISTModel
from RsNet.tf_config import gpu_config, setup_visibile_gpus, CHANNELS_LAST, CHANNELS_FIRST
from RsNet.dataset_nn import model_mnist_meta
from RsNet.random_spiking.nn_ops import random_spike_sample_scaling, random_spike_sample_scaling_per_sample


def random_spike(x, sample_rate, scaling, is_batch=True):
    if is_batch:
        return random_spike_sample_scaling(x, sample_rate=sample_rate, scaling=scaling)
    else:
        return random_spike_sample_scaling_per_sample(x, sample_rate=sample_rate, scaling=scaling)


def train(data, file_name, params, rand_params, num_epochs=50, batch_size=128, is_batch=True,
          dropout=0.0, data_format=None, init_model=None, train_temp=1, data_gen=None):
    """
    Standard neural network training procedure.
    """
    _input = Input(shape=data.train_data.shape[1:])
    x = _input

    x = Conv2D(params[0], (3, 3), padding="same", data_format=data_format)(x)
    x = Activation('relu')(x)
    x = Lambda(function=random_spike, arguments={
        "sample_rate": rand_params[0], "scaling": rand_params[1], "is_batch": is_batch})(x)
    x = Conv2D(params[1], (3, 3), padding="same", data_format=data_format)(x)
    x = Activation('relu')(x)
    x = Lambda(function=random_spike, arguments={
        "sample_rate": rand_params[2], "scaling": rand_params[3], "is_batch": is_batch})(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    x = Lambda(function=random_spike, arguments={
        "sample_rate": rand_params[4], "scaling": rand_params[5], "is_batch": is_batch})(x)

    x = Conv2D(params[2], (3, 3), padding="same", data_format=data_format)(x)
    x = Activation('relu')(x)
    x = Lambda(function=random_spike, arguments={
        "sample_rate": rand_params[6], "scaling": rand_params[7], "is_batch": is_batch})(x)
    x = Conv2D(params[3], (3, 3), padding="same", data_format=data_format)(x)
    x = Activation('relu')(x)
    x = Lambda(function=random_spike, arguments={
        "sample_rate": rand_params[8], "scaling": rand_params[9], "is_batch": is_batch})(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(x)
    x = Lambda(function=random_spike, arguments={
        "sample_rate": rand_params[10], "scaling": rand_params[11], "is_batch": is_batch})(x)

    x = Flatten()(x)
    x = Dense(params[4])(x)
    x = Activation('relu')(x)
    x = Lambda(function=random_spike, arguments={
        "sample_rate": rand_params[12], "scaling": rand_params[13], "is_batch": is_batch})(x)
    if dropout > 0:
        x = Dropout(dropout)(x, training=True)
    x = Dense(params[5])(x)
    x = Activation('relu')(x)
    x = Lambda(function=random_spike, arguments={
        "sample_rate": rand_params[14], "scaling": rand_params[15], "is_batch": is_batch})(x)
    x = Dense(10)(x)
    model = Model(_input, x)
    model.summary()

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    if init_model is not None:
        model.load_weights(init_model)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    if data_gen is None:
        model.fit(data.train_data, data.train_labels,
                  batch_size=batch_size,
                  validation_data=(data.test_data, data.test_labels),
                  nb_epoch=num_epochs,
                  shuffle=True)
    else:
        data_flow = data_gen.flow(data.train_data, data.train_labels, batch_size=128, shuffle=True)
        model.fit_generator(data_flow,
                            steps_per_epoch=len(data_flow),
                            validation_data=(data.validation_data, data.validation_labels),
                            nb_epoch=num_epochs,
                            shuffle=True)

    if file_name is not None:
        model.save(file_name)

    # save idx
    utils.save_model_idx(file_name, data)
    return model


def parse_rand_spike(_str):
    _str = _str.split(',')
    return [float(x) for x in _str]


parser = argparse.ArgumentParser(description='Train mnist model')

parser.add_argument('--data_dir', help='data dir, required', type=str, default=None)
parser.add_argument('--data_name', help='data name, required', type=str, default=None)
parser.add_argument('--model_dir', help='save model directory, required', type=str, default=None)
parser.add_argument('--model_name', help='save model name, required', type=str, default=None)
parser.add_argument('--validation_size', help='size of validation dataset', type=int, default=5000)
parser.add_argument('--random_spike', help='parameter used for random spiking', type=str, default=None)
parser.add_argument('--random_spike_batch', help='whether to use batch-wised random noise', type=str, default='yes')
parser.add_argument('--dropout', help='dropout rate', type=float, default=0.5)
parser.add_argument('--rotation', help='rotation angle', type=float, default=10)
parser.add_argument('--gpu_idx', help='gpu index', type=int, default=0)
parser.add_argument('--data_format', help='channels_last or channels_first', type=str, default=CHANNELS_FIRST)
parser.add_argument('--is_dis', help='whether to use distillation training', type=str, default='no')
parser.add_argument('--is_trans', help='whether do transfer training using soft label', type=str, default='no')
parser.add_argument('--is_data_gen', help='whether train on data generator, zoom, rotation', type=str, default='no')
parser.add_argument('--trans_model', help='transfer model name', type=str, default='no')
parser.add_argument('--trans_drop', help='dropout trans model name', type=float, default=0.5)
parser.add_argument('--trans_random_spike', help='random spiking parameter used for trans model',
                    type=str, default=None)
parser.add_argument('--train_sel_rand', help='whether to random select the training data', type=str, default='no')
parser.add_argument('--train_size', help='number of training example', type=int, default=0)
parser.add_argument('--pre_idx', help='predefined idx, duplicated training dataset', type=str, default=None)
parser.add_argument('--ex_data_dir', help='extra data dir, required', type=str, default=None)
parser.add_argument('--ex_data_name', help='extra data name, required', type=str, default=None)
parser.add_argument('--ex_data_size', help='number of extra training example', type=int, default=0)
parser.add_argument('--ex_data_sel_rand', help='whether to random select the extra training data',
                    type=str, default='no')

args = parser.parse_args()

data_dir = args.data_dir
data_name = args.data_name
save_model_dir = args.model_dir
save_model_name = args.model_name
validation_size = args.validation_size
train_size = args.train_size
train_sel_rand = args.train_sel_rand == 'yes'
para_random_spike = None if args.random_spike is None else parse_rand_spike(args.random_spike)
_is_batch = args.random_spike_batch == 'yes'
dropout = args.dropout
gpu_idx = args.gpu_idx
rotation = args.rotation
data_format = args.data_format
is_distillation = args.is_dis == 'yes'
is_data_gen = args.is_data_gen == 'yes'
ex_data_dir = args.ex_data_dir
ex_data_name = args.ex_data_name
ex_data_size = args.ex_data_size
ex_data_sel_rand = args.ex_data_sel_rand == 'yes'
pre_idx_path = args.pre_idx

setup_visibile_gpus(str(gpu_idx))

k.tensorflow_backend.set_session(tf.Session(config=gpu_config))

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

data = MNIST(data_dir, data_name, validation_size, model_meta=model_mnist_meta,
             input_data_format=CHANNELS_LAST, output_data_format=data_format,
             train_size=train_size, train_sel_rand=train_sel_rand)

if pre_idx_path is not None:
    pre_idx = utils.load_model_idx(pre_idx_path)
    data.apply_pre_idx(pre_idx)
if ex_data_dir is not None and ex_data_name is not None and ex_data_size > 0:
    data.append_train_data(ex_data_dir, ex_data_name, ex_data_size,
                           input_data_format=CHANNELS_LAST, output_data_format=data_format, sel_rand=ex_data_sel_rand)

# config data if using transfer training here
is_trans = args.is_trans == 'yes'
if is_trans:
    print("Get the soft label of the transfer model")
    trans_random_spike = None if args.trans_random_spike is None else parse_rand_spike(args.trans_random_spike)
    trans_model = MNISTModel(args.trans_model, None, output_logits=False,
                             input_data_format=data_format, data_format=data_format, dropout=0,
                             rand_params=trans_random_spike, is_batch=True)
    predicted = trans_model.model.predict(data.train_data, batch_size=500, verbose=1)
    train_data_acc = np.mean(np.argmax(predicted, 1) == np.argmax(data.train_labels, 1))
    data.train_labels = predicted
    print("trasfer model acc on training data:", train_data_acc)


if is_data_gen:
    data_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=rotation,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='reflect',
        width_shift_range=4,
        height_shift_range=4,
        horizontal_flip=False,
        vertical_flip=False,
        data_format=data_format
    )
else:
    data_gen = None

if is_distillation:
    print("train init model")
    train(data, save_model_dir + "/" + save_model_name + '_init',
          [32, 32, 64, 64, 200, 200], para_random_spike, num_epochs=1, is_batch=_is_batch,
          data_format=data_format, dropout=dropout, data_gen=data_gen)
    print("train teacher model")
    train(data, save_model_dir + "/" + save_model_name + '_teacher',
          [32, 32, 64, 64, 200, 200], para_random_spike, num_epochs=50, is_batch=_is_batch,
          data_format=data_format, dropout=dropout,
          init_model=save_model_dir + "/" + save_model_name + '_init', train_temp=100, data_gen=data_gen)
    # evaluate label with teacher model
    model_teacher = MNISTModel(os.path.join(save_model_dir, save_model_name + '_teacher'), None, output_logits=True,
                               input_data_format=data_format, data_format=data_format, dropout=0,
                               rand_params=para_random_spike, is_batch=True)
    predicted = model_teacher.model.predict(data.train_data, batch_size=500, verbose=1)
    train_data_acc = np.mean(np.argmax(predicted, 1) == np.argmax(data.train_labels, 1))
    print("train teacher acc:", train_data_acc)
    with tf.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted/100))
        print(y)
        data.train_labels = y
    print("train student model")
    train(data, save_model_dir + "/" + save_model_name,
          [32, 32, 64, 64, 200, 200], para_random_spike, num_epochs=50, is_batch=_is_batch,
          data_format=data_format, dropout=dropout,
          init_model=save_model_dir + "/" + save_model_name + '_init', train_temp=100, data_gen=data_gen)
else:
    train(data, save_model_dir + "/" + save_model_name,
          [32, 32, 64, 64, 200, 200], para_random_spike, num_epochs=50, is_batch=_is_batch,
          data_format=data_format, dropout=dropout, data_gen=data_gen)
