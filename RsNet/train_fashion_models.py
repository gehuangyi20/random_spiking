## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified for the needs of MagNet.

from keras import backend as k
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import multi_gpu_model

from RsNet.setup_mnist import MNIST, FASHIONModel
from RsNet.tf_config import gpu_config, get_available_gpus, setup_visibile_gpus, CHANNELS_LAST, CHANNELS_FIRST
from RsNet.dataset_nn import model_mnist_meta

import RsNet.wide_residual_network as wrn

import os
import tensorflow as tf
import numpy as np
import argparse
import utils


class SGDLearningRateTracker(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        optimizer = self.model.optimizer
        print("lr:", epoch, k.eval(optimizer.lr), k.eval(optimizer.decay), k.eval(optimizer.iterations))


def train_wrn(data_generator, file_name, num_epochs=200, debug=True, gpus=None,
              initial_epoch=0, initial_model=None, data_format=None, rand_spike=0, dropout=0.0, train_temp=1):
    # For WRN-16-8 put N = 2, k = 8
    # For WRN-28-10 put N = 4, k = 10
    # For WRN-40-4 put N = 6, k = 4

    def _create_model():
        return wrn.create_wide_residual_network(data.train_data.shape[1:],
                                                nb_classes=10, N=4, k=10, dropout=dropout, weight_decay=0.0005,
                                                output_logits=True, input_data_format=data_format,
                                                data_format=data_format, rand_spike=rand_spike)

    devices = get_available_gpus()
    devices_len = len(devices)

    if gpus is None:
        gpus = []
    elif isinstance(gpus, list):
        pass
    elif isinstance(gpus, int):
        gpus = list(range(gpus))
    else:
        raise ValueError('number of gpus is either list or int')

    if devices_len >= 2 and len(gpus) >= 2:
        with tf.device('/cpu:0'):
            _model = _create_model()
        _model_multi = multi_gpu_model(_model, gpus)
    else:

        _model = _create_model()
        _model_multi = _model

    # learning rate schedule
    lr_schedule = [60, 120, 160, 180]  # epoch_step

    def schedule(epoch_idx):
        if (epoch_idx + 1) < lr_schedule[0]:
            return 0.1
        elif (epoch_idx + 1) < lr_schedule[1]:
            return 0.02  # lr_decay_ratio = 0.2
        elif (epoch_idx + 1) < lr_schedule[2]:
            return 0.004
        elif (epoch_idx + 1) < lr_schedule[3]:
            return 0.0008
        return 0.0008

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
    if initial_epoch > 0:
        _model_multi.load_weights(initial_model)
    _model_multi.compile(loss=fn,
                         optimizer=sgd,
                         metrics=['accuracy'])
    _model.summary()

    log_path = file_name + "_log"
    checkpoint_path = os.path.join(log_path, "checkpoint")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    _model_multi.fit_generator(data_generator,
                               steps_per_epoch=len(data_generator),
                               validation_data=(data.validation_data, data.validation_labels),
                               epochs=num_epochs,
                               initial_epoch=initial_epoch,
                               callbacks=[
                                   LearningRateScheduler(schedule=schedule),
                                   SGDLearningRateTracker(),
                                   ModelCheckpoint(os.path.join(
                                       checkpoint_path, 'weights.{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5'),
                                                   monitor='val_acc',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode='auto')
                               ],
                               shuffle=True)

    if debug:
        plot_model(_model, os.path.join(log_path, "WRN-{0}-{1}.png".format(28, 10)), show_shapes=True,
                   show_layer_names=True)
        with open(os.path.join(log_path, 'WRN-{0}-{1}.json'.format(28, 10)), 'w') as f:
            f.write(_model.to_json())
            f.close()

    if file_name is not None:
        _model.save(file_name)

    # save idx
    utils.save_model_idx(file_name, data)

    return _model


parser = argparse.ArgumentParser(description='Train fashion model')

parser.add_argument('--data_dir', help='data dir, required', type=str, default=None)
parser.add_argument('--data_name', help='data name, required', type=str, default=None)
parser.add_argument('--model_dir', help='save model directory, required', type=str, default=None)
parser.add_argument('--model_name', help='save model name, required', type=str, default=None)
parser.add_argument('--validation_size', help='size of validation dataset', type=int, default=5000)
parser.add_argument('--resume_epoch', help='resume epoch', type=int, default=0)
parser.add_argument('--resume_model', help='resume model path', type=str, default=None)
parser.add_argument('--random_spike', help='parameter used for random spiking', type=int, default=None)
parser.add_argument('--dropout', help='dropout rate', type=float, default=0.5)
parser.add_argument('--rotation', help='rotation angle', type=float, default=10)
parser.add_argument('--gpu_idx', help='gpu indexs', type=str, default=0)
parser.add_argument('--data_format', help='channels_last or channels_first', type=str, default=CHANNELS_FIRST)
parser.add_argument('--is_dis', help='whether to use distillation training', type=str, default='no')
parser.add_argument('--is_enc', help='whether to train with encrypted data', type=str, default='no')
parser.add_argument('--enc_chg_key_iter', help='change the key after number of encryption', type=int, default=0)
parser.add_argument('--enc_orig_rate', help='change the key after number of encryption', type=float, default=0)
parser.add_argument('--bit_depth', help='train image with reduced bit depth, [1-8], default 8', type=int, default=8)
parser.add_argument('--is_trans', help='whether do transfer training using soft label', type=str, default='no')
parser.add_argument('--is_1bit', help='whether train on one bit dithered image', type=str, default='no')
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
rotation = args.rotation
data_format = args.data_format
rspike = args.random_spike
dropout = args.dropout
gpus_str = args.gpu_idx
is_distillation = args.is_dis == 'yes'
resume_epoch = args.resume_epoch
resume_model = args.resume_model
encrypt = args.is_enc == 'yes'
enc_chg_key_iter = args.enc_chg_key_iter
enc_orig_rate = args.enc_orig_rate
bit_depth = args.bit_depth
is_one_bit = args.is_1bit == 'yes'
ex_data_dir = args.ex_data_dir
ex_data_name = args.ex_data_name
ex_data_size = args.ex_data_size
ex_data_sel_rand = args.ex_data_sel_rand == 'yes'
pre_idx_path = args.pre_idx


def parse_gpus_str(in_str):
    _tmp = in_str.split(',')
    if len(_tmp) == 1:
        return list(range(int(_tmp[0]))) if _tmp[0] != '' else []
    else:
        return [int(x) for x in _tmp if x != '']


selected_gpus = parse_gpus_str(gpus_str)
setup_visibile_gpus(",".join(map(str, selected_gpus)))
selected_gpus = list(range(len(selected_gpus)))
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
    trans_random_spike = None if args.trans_random_spike is None else args.trans_random_spike
    trans_model = FASHIONModel(args.trans_model, None, output_logits=False,
                               input_data_format=data_format, data_format=data_format, dropout=0,
                               rand_params=trans_random_spike, is_batch=True)
    predicted = trans_model.model.predict(data.train_data, batch_size=500, verbose=1)
    train_data_acc = np.mean(np.argmax(predicted, 1) == np.argmax(data.train_labels, 1))
    data.train_labels = predicted
    print("trasfer model acc on training data:", train_data_acc)


datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=rotation,
                shear_range=0.2,
                zoom_range=(0.8, 1.2),
                fill_mode='reflect',
                width_shift_range=4,
                height_shift_range=4,
                horizontal_flip=True,
                vertical_flip=False,
                data_format=data_format
            )
datagen.fit(data.train_data, augment=True)
data_flow = datagen.flow(data.train_data, data.train_labels, batch_size=128, shuffle=True)


if is_distillation:
    print("train init model")
    if not os.path.exists(os.path.join(save_model_dir, save_model_name + '_init')):
        train_wrn(data_flow, os.path.join(save_model_dir, save_model_name + '_init'), num_epochs=1,
                  debug=False, gpus=selected_gpus, data_format=data_format, rand_spike=rspike,
                  dropout=dropout)
    print("train teacher model")
    if resume_model is None:
        resume_model = os.path.join(save_model_dir, save_model_name + '_init')
    if not os.path.exists(os.path.join(save_model_dir, save_model_name + '_teacher')):
        train_wrn(data_flow, os.path.join(save_model_dir, save_model_name + '_teacher'),
                  num_epochs=200, debug=False, gpus=selected_gpus, initial_epoch=resume_epoch,
                  initial_model=resume_model, data_format=data_format, rand_spike=rspike,
                  dropout=dropout, train_temp=100)

    model_teacher = FASHIONModel(os.path.join(save_model_dir, save_model_name + '_teacher'), None, output_logits=True,
                                 input_data_format=data_format, data_format=data_format, dropout=0,
                                 rand_params=rspike, is_batch=True)
    # evaluate label with teacher model
    predicted = model_teacher.model.predict(data.train_data, batch_size=500, verbose=1)
    train_data_acc = np.mean(np.argmax(predicted, 1) == np.argmax(data.train_labels, 1))
    print("train teacher acc:", train_data_acc)
    with tf.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted / 100))
        data.train_labels = y
        data_flow = datagen.flow(data.train_data, data.train_labels, batch_size=128, shuffle=True)

    print("train student model")
    train_wrn(data_flow, os.path.join(save_model_dir, save_model_name),
              num_epochs=200, debug=False, gpus=selected_gpus, initial_epoch=resume_epoch,
              initial_model=resume_model, data_format=data_format, rand_spike=rspike,
              dropout=dropout, train_temp=100)
else:
    model = train_wrn(data_flow, os.path.join(save_model_dir, save_model_name), num_epochs=200, debug=False,
                      gpus=selected_gpus, initial_epoch=resume_epoch, initial_model=resume_model, data_format=data_format,
                      rand_spike=rspike, dropout=dropout)

k.tensorflow_backend.clear_session()
