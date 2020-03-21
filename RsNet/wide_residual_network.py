from tensorflow.keras.layers import Input, Add, Activation, Dropout, Flatten, Dense, Lambda
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from RsNet.tf_config import CHANNELS_FIRST, CHANNELS_LAST
from RsNet.random_spiking.nn_ops import random_spike_sample_scaling
import tensorflow as tf


def initial_conv(_input, _weight_decay=0.0005, data_format=K.image_data_format(),
                 rand_spike=False, sample_rate=0.2, scaling=1.0):
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(_weight_decay),
                      use_bias=False, data_format=data_format)(_input)

    channel_axis = 1 if data_format == CHANNELS_FIRST else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def expand_conv(_init, base, k, dropout=0.0, strides=(1, 1), _weight_decay=0.0005, data_format=K.image_data_format(),
                rand_spike=False, sample_rate=0.2, scaling=1.0):
    if rand_spike:
        _init = Lambda(lambda _x: random_spike_sample_scaling(_x, sample_rate=sample_rate, scaling=scaling), name='')(_init)
    x = Conv2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                      kernel_regularizer=l2(_weight_decay), use_bias=False, data_format=data_format)(_init)

    channel_axis = 1 if data_format == CHANNELS_FIRST else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x, training=True)
    x = Conv2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(_weight_decay), use_bias=False, data_format=data_format)(x)

    skip = Conv2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                         kernel_regularizer=l2(_weight_decay), use_bias=False, data_format=data_format)(_init)

    m = Add()([x, skip])

    return m


def conv1_block(_input, k=1, dropout=0.0, _weight_decay=0.0005, data_format=K.image_data_format(),
                rand_spike=False, sample_rate=0.2, scaling=1.0):
    _init = _input

    channel_axis = 1 if data_format == CHANNELS_FIRST else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(_input)
    x = Activation('relu')(x)
    if rand_spike:
        x = Lambda(lambda _x: random_spike_sample_scaling(_x, sample_rate=sample_rate, scaling=scaling), name='')(x)
    x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(_weight_decay), use_bias=False, data_format=data_format)(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x, training=True)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(_weight_decay), use_bias=False, data_format=data_format)(x)

    m = Add()([_init, x])
    return m


def conv2_block(_input, k=1, dropout=0.0, _weight_decay=0.0005, data_format=K.image_data_format(),
                rand_spike=False, sample_rate=0.2, scaling=1.0):
    _init = _input

    channel_axis = 1 if data_format == CHANNELS_FIRST else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(_input)
    x = Activation('relu')(x)
    if rand_spike:
        x = Lambda(lambda _x: random_spike_sample_scaling(_x, sample_rate=sample_rate, scaling=scaling), name='')(x)
    x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(_weight_decay), use_bias=False, data_format=data_format)(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x, training=True)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(_weight_decay), use_bias=False, data_format=data_format)(x)

    m = Add()([_init, x])
    return m


def conv3_block(_input, k=1, dropout=0.0, _weight_decay=0.0005, data_format=K.image_data_format()):
    _init = _input

    channel_axis = 1 if data_format == CHANNELS_FIRST else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(_input)
    x = Activation('relu')(x)
    x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(_weight_decay), use_bias=False, data_format=data_format)(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x, training=True)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=l2(_weight_decay), use_bias=False, data_format=data_format)(x)

    m = Add()([_init, x])
    return m


def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, weight_decay=0.0005,
                                 output_logits=True, verbose=1, input_data_format=CHANNELS_LAST,
                                 data_format=K.image_data_format(), rand_spike=0):
    """
    Creates a Wide Residual Network with specified parameters

    :param input_dim: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param weight_decay: page 10: "Used in all experiments"
    :param output_logits: whether output logits (tf) or softmax for keras
    :param verbose: Debug info to describe created WRN
    :param input_data_format: input data format, channels_first or channels_last
    :param data_format: internal model data format, channels_first or channels_last
    :param rand_spike: whether to using rspike in the training model
    :param encrypt whether to encrypt the dataset before feed it into the network
    :param chg_key_iter changes keys after encrypt chg_key_iter number of batches
    :param origin_rate the probability of data does not be encrypted.
    :param enc_layer encryption layer,
    :param enc_grp_chn whether to group the channel then do the encryption, only applicable to color image
    :param force_gray whether to convert the image in gray scale intentionally before training
    :param bit_depth pre-process the image to reduced bit depth
    :param palette_shade the dimension for palette box, which is to reduce the color
    :param one_bit whether to train the image with one bit
    complexity of image with dithering
    :return:
    """
    assert input_data_format in (CHANNELS_FIRST, CHANNELS_LAST)
    assert data_format in (CHANNELS_FIRST, CHANNELS_LAST)
    channel_axis = 1 if data_format == CHANNELS_FIRST else -1

    ip = Input(shape=input_dim)

    x = ip
    if data_format != input_data_format:
        if input_data_format == CHANNELS_LAST:
            # Computation requires channels_first.
            x = Lambda(lambda _x: tf.transpose(_x, [0, 3, 1, 2], name="transpose"))(ip)
        else:
            # Computation requires channels_last.
            x = Lambda(lambda _x: tf.transpose(_x, [0, 2, 3, 1], name="transpose"))(ip)

    layer_rspike = True if rand_spike == 1 else False
    print(0, layer_rspike)
    x = initial_conv(x, _weight_decay=weight_decay, data_format=data_format)
    nb_conv = 4

    x = expand_conv(x, 16, k, dropout, _weight_decay=weight_decay, data_format=data_format,
                    rand_spike=layer_rspike, sample_rate=0.2, scaling=1)
    nb_conv += 2

    for i in range(N - 1):
        if i + 2 == rand_spike:
            layer_rspike = True
        else:
            layer_rspike = False
        print(i+1, layer_rspike)
        x = conv1_block(x, k, dropout, _weight_decay=weight_decay, data_format=data_format,
                        rand_spike=layer_rspike, sample_rate=0.2, scaling=1)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    layer_rspike = True if rand_spike == N+1 else False
    print(N, layer_rspike)
    x = expand_conv(x, 32, k, dropout, strides=(2, 2), _weight_decay=weight_decay, data_format=data_format,
                    rand_spike=layer_rspike, sample_rate=0.2, scaling=1)
    nb_conv += 2

    for i in range(N - 1):
        if i + N + 2 == rand_spike:
            layer_rspike = True
        else:
            layer_rspike = False
            print(i + 1 + N, layer_rspike)
        x = conv2_block(x, k, dropout, _weight_decay=weight_decay, data_format=data_format,
                        rand_spike=layer_rspike, sample_rate=0.2, scaling=1)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, k, dropout, strides=(2, 2), _weight_decay=weight_decay, data_format=data_format)
    nb_conv += 2

    for i in range(N - 1):
        x = conv3_block(x, k, dropout, _weight_decay=weight_decay, data_format=data_format)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    if input_data_format == CHANNELS_LAST:
        pool_size = (input_dim[0]//4, input_dim[1]//4)
    else:
        pool_size = (input_dim[1] // 4, input_dim[2] // 4)
    x = AveragePooling2D(pool_size=pool_size, data_format=data_format)(x)
    x = Flatten()(x)

    fn_activation = None if output_logits is True else 'softmax'
    x = Dense(nb_classes, activation=fn_activation)(x)

    model = Model(ip, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


if __name__ == "__main__":
    from keras.utils import plot_model

    init = (32, 32, 3)

    wrn_28_10 = create_wide_residual_network(init, nb_classes=10, N=2, k=2, dropout=0.0)

    wrn_28_10.summary()

    plot_model(wrn_28_10, "WRN-16-2.png", show_shapes=True, show_layer_names=True)
