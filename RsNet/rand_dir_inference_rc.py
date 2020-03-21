import numpy as np
import tensorflow as tf
import os
import argparse
import time
import sys
from tensorflow.keras.optimizers import SGD
from RsNet.tf_config import gpu_config, setup_visibile_gpus, CHANNELS_LAST
from RsNet.setup_mnist import MNIST, MNISTModel, FASHIONModel, CIFAR10Model
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta
from utils import softmax_cross_entropy_with_logits
from RsNet.ciphernet import image_jpeg_compression

parser = argparse.ArgumentParser(description='inference the example with random direction noise')

parser.add_argument('--batch_size', help='batch_size', type=int, default=50)
parser.add_argument('--gpu_idx', help='gpu idx used in the evaluation', type=int, default=0)
parser.add_argument('--iter1', help='iterate 1, number of sample points', type=int, default=None)
parser.add_argument('--iter2', help='iterate 2 bagging or majority', type=int, default=None)
parser.add_argument('--dropout', help='dropout rate', type=float, default=None)
parser.add_argument('--noise_distance', help='l2_distance', type=float, default=0)
parser.add_argument('--noise_iter', help='add l2 times', type=int, default=1)
parser.add_argument('--noise_mthd', help='add noise method', type=str, default='l2')
parser.add_argument('--r_dis', help='noise distance', type=float, default=None)
parser.add_argument('--bagging', help='yes or no', type=str, default=None)
parser.add_argument('--test_data_start', help='start_test_data_idx', type=int, default=0)
parser.add_argument('--test_data_len', help='test_data_length', type=int, default=100)
parser.add_argument('--output_file', help='save filename', type=str, default=None)
parser.add_argument('--verbose', help='verbose 0 or 1', type=int, default=0)

requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-d', '--data_dir', help='attack dir', type=str, required=True)
requiredNamed.add_argument('-n', '--data_name', help='attack name', type=str, required=True)
requiredNamed.add_argument('--model_dir', help='model save dir', type=str, required=True)
requiredNamed.add_argument('--model_name', help='model save name', type=str, required=True)
requiredNamed.add_argument('-s', '--set_name', help='attack set name', type=str, required=True)
requiredNamed.add_argument('--data_format', help='data_format', type=str, required=True)

args = parser.parse_args()

setup_visibile_gpus(str(args.gpu_idx))

set_name = args.set_name
if set_name == 'mnist':
    def parse_rand_spike(_str):
        _str = _str.split(',')
        return [float(x) for x in _str]

    model_meta = model_mnist_meta
    MODEL = MNISTModel
    para_random_spike = None
elif set_name == 'fashion':
    model_meta = model_mnist_meta
    MODEL = FASHIONModel
    para_random_spike = 0
elif set_name == "cifar10":
    model_meta = model_cifar10_meta
    MODEL = CIFAR10Model
    para_random_spike = 0
else:
    model_meta = None
    MODEL = None
    para_random_spike = None
    print("invalid data set name %s" % set_name)
    exit(0)

batch_size = args.batch_size
bagging = args.bagging == 'yes'
iter1 = args.iter1
iter2 = args.iter2
noise_mthd = args.noise_mthd
noise_iter = args.noise_iter
noise_distance = args.noise_distance
r_dis = args.r_dis
data_start = args.test_data_start
data_len = args.test_data_len
data_format = args.data_format
verbose = args.verbose
output_file = args.output_file

model_list = args.model_name.split(",")

data = MNIST(args.data_dir, args.data_name, 0, model_meta=model_meta,
             input_data_format=CHANNELS_LAST, output_data_format=data_format)

if (output_file is not None) and ( not os.path.exists(os.path.dirname(output_file)) ):
    os.makedirs(os.path.dirname(output_file))

fp = open(output_file, 'w') if output_file is not None else sys.stdout
fp.write("name\tl2\ttest_acc\ttest_unchg\ttest_loss\n")

with tf.Session(config=gpu_config) as sess:
    timestart = time.time()

    for model_name in model_list:
        print("=============================")
        print("valid rand_inference for model %s" % model_name)
        print("=============================")
        fp.write(model_name)
        fp.write("\t%.4f" % noise_distance)

        # get current model
        cur_model = MODEL(os.path.join(args.model_dir, model_name), sess, output_logits=True,
                          input_data_format=data_format, data_format=data_format, dropout=args.dropout,
                          rand_params=para_random_spike, is_batch=True)

        k_model = cur_model.model
        sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)

        k_model.compile(loss=softmax_cross_entropy_with_logits,
                        optimizer=sgd,
                        metrics=['accuracy'])

        y_correct = tf.placeholder(tf.float32, [None, model_meta.labels])
        y_predict = tf.placeholder(tf.float32, [None, model_meta.labels])
        label_val = np.arange(model_meta.labels)

        loss = softmax_cross_entropy_with_logits(y_correct, y_predict)

        para_r_dis = tf.placeholder(tf.float32, [])
        para_shape = list(data.test_data.shape)
        para_shape[0] = None
        para_x = tf.placeholder(tf.float32, para_shape)
        x_tmp = tf.random_uniform(tf.shape(para_x),
                                  minval=para_r_dis, maxval=-para_r_dis, dtype=tf.float32) + para_x
        rand_x = tf.clip_by_value(x_tmp, 0, 1)

        def add_noise_data_by_mthd(data, mthd, para):
            if mthd == 'l2':
                return add_noise_data(data, para)
            elif mthd == 'jpeg':
                return image_jpeg_compression.compress_float_py(data, quality=int(para), data_format=data_format)
            elif mthd == 'l_inf':
                return add_noise_data_l_inf(data, para)
            else:
                return data

        def add_noise_data(data, l2):
            init_noise = np.random.normal(size=np.shape(data))
            init_l2 = np.sqrt(np.sum(np.square(init_noise), axis=(1, 2, 3)))

            scale = l2 / init_l2.reshape((-1, 1, 1, 1))
            noise = init_noise * scale

            noise_data = data + noise
            noise_data = np.clip(noise_data, 0, 1)

            return noise_data

        def add_noise_data_l_inf(data, EPSILON):
            init_noise = np.random.random(size=np.shape(data))
            noise = (init_noise - 0.5) * (2 * EPSILON)
            noise_data = data + noise

            lower = np.clip(data - EPSILON, 0, 1)
            upper = np.clip(data + EPSILON, 0, 1)
            noise_data = np.clip(noise_data, lower, upper)

            return noise_data

        def predict_data(data, correct_label):
            predict_label = k_model.predict(x=data, batch_size=batch_size, verbose=verbose)
            computed_loss = loss.eval(session=sess, feed_dict={y_correct: correct_label, y_predict: predict_label})
            cur_loss = np.sum(computed_loss)
            return predict_label, cur_loss

        def evaluate(x, y, iter1, iter2, bagging, r_val):
            data_len = len(y)
            ref_pred, ref_loss = predict_data(x, y)
            total_pred = []
            total_unchg_pred = []
            total_loss = 0
            avg_loss = 0
            avg_acc = 0
            avg_unchg = 0
            correct_label = np.argmax(y, 1)
            ref_label = np.argmax(ref_pred, 1)

            # predict on noise data
            for noise_i in range(noise_iter):
                noise_data = add_noise_data_by_mthd(x, noise_mthd, noise_distance)
                noise_pred = None
                noise_loss = 0

                print("noise round %d/%d" % (noise_i + 1, noise_iter))

                # use range based classifier to predict on noise data
                for i in range(iter1):
                    cur_data = sess.run(rand_x, feed_dict={para_x: noise_data, para_r_dis: r_val})
                    cur_pred = None
                    cur_loss = 0

                    print("rc predict %d/%d" % (i + 1, iter1), end="\r")
                    for j in range(iter2):
                        cur_iter_pred, cur_iter_loss = predict_data(cur_data, y)
                        cur_loss += cur_iter_loss

                        if bagging:
                            batch_pred = cur_iter_pred
                        else:
                            batch_pred = (np.argmax(cur_iter_pred, 1)[:, None] == label_val).astype(np.float32)
                        if cur_pred is None:
                            cur_pred = batch_pred
                        else:
                            cur_pred += batch_pred

                    cur_pred = (np.argmax(cur_pred, 1)[:, None] == label_val).astype(np.float32)

                    noise_loss += cur_loss

                    if noise_pred is None:
                        noise_pred = cur_pred
                    else:
                        noise_pred += cur_pred

                    if i % 100 == 99:
                        print("")
                        tmp_cur_pred_label = np.argmax(cur_pred, 1)
                        tmp_cur_acc = np.mean(np.equal(tmp_cur_pred_label, correct_label))
                        tmp_cur_unchg = np.mean(np.equal(tmp_cur_pred_label, ref_label))
                        tmp_cur_loss = cur_loss / iter2 / data_len

                        tmp_noise_pred_label = np.argmax(noise_pred, 1)
                        tmp_noise_acc = np.mean(np.equal(tmp_noise_pred_label, correct_label))
                        tmp_noise_unchg = np.mean(np.equal(tmp_noise_pred_label, ref_label))
                        tmp_noise_loss = noise_loss / iter2 / data_len / (i+1)

                        print("iter: %d/%d cur_loss: %.4f - cur_acc: %.4f - cur_unchg: %.4f"
                              " - noise_avg_loss: %.4f - noise_avg_acc: %.4f - noise_avg_unchg: %.4f" %
                              ((i+1), iter1,
                               tmp_cur_loss, tmp_cur_acc, tmp_cur_unchg,
                               tmp_noise_loss, tmp_noise_acc, tmp_noise_unchg))

                noise_pred_label = np.argmax(noise_pred, 1)
                noise_correct_pred = np.equal(noise_pred_label, correct_label)
                noise_unchg_pred = np.equal(noise_pred_label, ref_label)
                total_loss += noise_loss
                total_pred.extend(noise_correct_pred)
                total_unchg_pred.extend(noise_unchg_pred)

                noise_acc = np.mean(noise_correct_pred)
                noise_unchg = np.mean(noise_unchg_pred)
                noise_loss = noise_loss / iter1 / iter2 / data_len

                avg_acc = np.mean(total_pred)
                avg_unchg = np.mean(total_unchg_pred)
                avg_loss = total_loss / iter1 / iter2 / (noise_i + 1) / data_len

                print("finish noise %d/%d" % ((noise_i+1), noise_iter))
                print("noise_loss: %.4f - noise_acc: %.4f - noise_unchg: %.4f"
                      " - avg_loss: %.4f - avg_acc: %.4f - avg_unchg: %.4f" %
                      (noise_loss, noise_acc, noise_unchg,
                       avg_loss, avg_acc, avg_unchg))

            return avg_loss, avg_acc, avg_unchg


        print("evaluating model:", model_name)
        print("* test data *")
        re_loss, re_acc, re_unchg = evaluate(x=data.test_data[data_start: data_start + data_len],
                                             y=data.test_labels[data_start: data_start + data_len],
                                             iter1=iter1, iter2=iter2, bagging=bagging, r_val=r_dis)
        print("avg_loss: %.4f - avg_acc: %.4f - avg_acc: %.4f" % (re_loss, re_acc, re_unchg))
        fp.write('\t%.4f\t%.4f\t%.4f\n' % (re_acc, re_unchg, re_loss))
    timeend = time.time()
    print("Took", timeend - timestart, "seconds to run")

    fp.close()
