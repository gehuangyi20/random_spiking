## train_defense.py
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta
from RsNet.setup_mnist import MNIST
from RsNet.defensive_models import DenoisingAutoEncoder as DAE

from RsNet.tf_config import gpu_config, setup_visibile_gpus, CHANNELS_FIRST, CHANNELS_LAST
from keras import backend as k
import tensorflow as tf
import os
import argparse


parser = argparse.ArgumentParser(description='Train magnet defensive model')
parser.add_argument('--data_dir', help='data dir, required', type=str, default=None)
parser.add_argument('--data_name', help='data name, required', type=str, default=None)
parser.add_argument('--model_dir', help='save model directory, required', type=str, default=None)
parser.add_argument('--model_name', help='save model name, required', type=str, default=None)
parser.add_argument('--set_name', help='set name [mnist, fashion, cifar10], required', type=str, default=None)
parser.add_argument('--defense_name', help='defense name [magnet, fs], required', type=str, default='magnet')
parser.add_argument('--validation_size', help='size of validation dataset', type=int, default=5000)
parser.add_argument('--train_sel_rand', help='whether to random select the training data', type=str, default='no')
parser.add_argument('--train_size', help='number of training example', type=int, default=0)
parser.add_argument('--gpu_idx', help='gpu index', type=int, default=0)
parser.add_argument('--data_format', help='channels_last or channels_first', type=str, default=CHANNELS_FIRST)

args = parser.parse_args()

data_dir = args.data_dir
data_name = args.data_name
save_model_dir = args.model_dir
save_model_name = args.model_name
set_name = args.set_name
defense_name=args.defense_name
validation_size = args.validation_size
train_size = args.train_size
train_sel_rand = args.train_sel_rand == 'yes'
gpu_idx = args.gpu_idx
data_format = args.data_format
setup_visibile_gpus(str(gpu_idx))

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

if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

k.tensorflow_backend.set_session(tf.Session(config=gpu_config))

poolings = ["average", "max"]

shape = [model_meta.height, model_meta.width, model_meta.channel] \
    if data_format == CHANNELS_LAST else [model_meta.channel, model_meta.height, model_meta.width]
combination_I = [3, "average", 3]
combination_II = [3]
activation = "sigmoid"
reg_strength = 1e-9

data = MNIST(data_dir, data_name, validation_size, model_meta=model_meta,
             input_data_format=CHANNELS_LAST, output_data_format=data_format,
             train_size=train_size, train_sel_rand=train_sel_rand)

if set_name == 'mnist':
    epochs = 100
    AE_BIT = DAE(shape, combination_I, v_noise=0.1, activation=activation,
                 reg_strength=reg_strength, model_dir=save_model_dir,
                 data_format=data_format, input_data_format=data_format)
    AE_BIT.train(data, save_model_name + "_R", num_epochs=epochs)

    AE_I = DAE(shape, combination_I, v_noise=0.1, activation=activation,
               reg_strength=reg_strength, model_dir=save_model_dir,
               data_format=data_format, input_data_format=data_format)
    AE_I.train(data, save_model_name + "_I", num_epochs=epochs)

    AE_II = DAE(shape, combination_II, v_noise=0.1, activation=activation,
                reg_strength=reg_strength, model_dir=save_model_dir,
                data_format=data_format, input_data_format=data_format)
    AE_II.train(data, save_model_name + "_II", num_epochs=epochs)
else:
    epochs = 400
    AE_BIT = DAE(shape, combination_II, v_noise=0.025, activation=activation,
                 reg_strength=reg_strength, model_dir=save_model_dir,
                 data_format=data_format, input_data_format=data_format)
    AE_BIT.train(data, save_model_name + "_R", num_epochs=epochs)

    AE_I = DAE(shape, combination_II, v_noise=0.025, activation=activation,
               reg_strength=reg_strength, model_dir=save_model_dir,
               data_format=data_format, input_data_format=data_format)
    AE_I.train(data, save_model_name + "_I", num_epochs=epochs)

    AE_T10 = DAE(shape, combination_II, v_noise=0.025, activation=activation,
                 reg_strength=reg_strength, model_dir=save_model_dir,
                 data_format=data_format, input_data_format=data_format)
    AE_T10.train(data, save_model_name + "_T10", num_epochs=epochs)

    AE_T40 = DAE(shape, combination_II, v_noise=0.025, activation=activation,
                 reg_strength=reg_strength, model_dir=save_model_dir,
                 data_format=data_format, input_data_format=data_format)
    AE_T40.train(data, save_model_name + "_T40", num_epochs=epochs)
