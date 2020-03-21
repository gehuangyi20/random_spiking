from PIL import Image
import numpy as np
from io import BytesIO
from scipy import misc
from RsNet.tf_config import CHANNELS_LAST, CHANNELS_FIRST


def _compress(data, quality=75):
    data_shape = np.shape(data)
    is_l = data_shape[-1] == 1
    if is_l:
        data = np.squeeze(data, axis=3)

    buffer_fp = BytesIO()
    tmp_data = []
    for cur_data in data:
        buffer_fp.truncate()
        buffer_fp.seek(0)
        cur_img = Image.fromarray(cur_data)
        cur_img.save(buffer_fp, format='jpeg', quality=quality)
        # buffer = buffer_fp.getbuffer()
        compressed_img = Image.open(buffer_fp)
        #compressed_img.show()
        tmp_data.append(misc.fromimage(compressed_img))

    data = np.asarray(tmp_data)

    if is_l:
        data = np.expand_dims(data, axis=3)
        #data = data[:, :, :, np.newaxis]

    return data


def compress_py(x, quality=75, data_format=CHANNELS_LAST):
    # x_shape = np.shape(x)
    if data_format == CHANNELS_FIRST:
        x = np.transpose(x, [0, 2, 3, 1])

    x = _compress(x, quality=quality)

    if data_format == CHANNELS_FIRST:
        x = np.transpose(x, [0, 3, 1, 2])

    return x


def compress_float_py(x, quality=75, data_format=CHANNELS_LAST, high=255, low=0, cmin=0, cmax=1):
    scale = (high - low) / (cmax - cmin)
    x = (x - cmin) * scale + low
    x = np.rint(np.clip(x, 0, 255))
    x = x.astype(np.uint8)
    x = compress_py(x, quality=quality, data_format=data_format)
    x = x / scale + cmin
    return x


if __name__ == '__main__':
    import argparse
    from RsNet.setup_mnist import MNIST
    from dataset_nn import model_cifar10_meta, model_mnist_meta

    parser = argparse.ArgumentParser(description='compress the image to jpeg')
    parser.add_argument('--data_dir', help='data dir, required', type=str, default=None)
    parser.add_argument('--data_name', help='data name, required', type=str, default=None)
    parser.add_argument('--data_format', help='data format, required', type=str, default=CHANNELS_FIRST)
    parser.add_argument('--normalize', help='whether normalize the data', type=str, default='no')
    parser.add_argument('--set_name', help='dataset name: mnist, fashion, cifar10, required', type=str, default='mnist')
    parser.add_argument('--output', help='output file name', type=str, default=None)
    parser.add_argument('--quality', help='jpeg quality', type=int, default=75)

    args = parser.parse_args()
    data_dir = args.data_dir
    data_name = args.data_name
    data_format = args.data_format
    normalize = args.normalize == 'yes'
    set_name = args.set_name
    output = args.output
    quality = args.quality

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

    data = MNIST(data_dir, data_name, 0, model_meta=model_meta,
                 input_data_format=CHANNELS_LAST, output_data_format=data_format, normalize=normalize)

    images = data.test_data[0:500]
    x_shape = images.shape

    if normalize:
        output_img = compress_float_py(images, data_format=data_format, quality=quality)
        output_img = np.clip(output_img * 255, 0, 255)
        output_img = output_img.astype(np.uint8)
    else:
        images = images.astype(np.uint8)
        output_img = compress_py(images, data_format=data_format, quality=quality)

    print(output_img.shape)
    if data_format == CHANNELS_FIRST:
        output_img = output_img.transpose([0, 2, 3, 1])
        output_img = output_img.reshape([20, 25, x_shape[2], x_shape[3], x_shape[1]])
        output_img = output_img.transpose([0, 2, 1, 3, 4])
        output_img = output_img.reshape([20 * x_shape[2], 25 * x_shape[3], x_shape[1]])
    else:
        output_img = output_img.reshape([20, 25, x_shape[1], x_shape[2], x_shape[3]])
        output_img = output_img.transpose([0, 2, 1, 3, 4])
        output_img = output_img.reshape([20 * x_shape[1], 25 * x_shape[2], x_shape[3]])

    if output_img.shape[-1] == 1:
        output_img = np.squeeze(output_img, axis=2)
    output_img = misc.toimage(output_img)
    misc.imsave(output, output_img)