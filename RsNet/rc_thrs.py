import argparse
import os
import time
import tensorflow as tf
import numpy as np
from keras.optimizers import SGD
from RsNet.tf_config import gpu_config
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta
from RsNet.tf_config import setup_visibile_gpus, CHANNELS_LAST
from RsNet.setup_mnist import MNIST,  MNISTModel, FASHIONModel, CIFAR10Model
from utils import softmax_cross_entropy_with_logits

parser = argparse.ArgumentParser(description='Summarize the result of random direction inference.')

parser.add_argument('--data_dir', help='data dir, required', type=str, default=None)
parser.add_argument('--data_name', help='data name, required', type=str, default=None)
parser.add_argument('--model_dir', help='model directory, required', type=str, default=None)
parser.add_argument('--model_name', help='model name, required', type=str, default=None)
parser.add_argument('-s', '--set_name', help='set name, mnist, fashion, cifar10', type=str, default=None)
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=None)
parser.add_argument('--data_format', help='channels_last or channels_first', type=str, default=None)
parser.add_argument('--gpu_idx', help='gpu index', type=int, default=None)
parser.add_argument('--iter1', help='iterate 1, number of sample points', type=int, default=None)
parser.add_argument('--iter2', help='iterate 2 bagging or majority', type=int, default=None)
parser.add_argument('--dropout', help='dropout rate', type=float, default=None)
parser.add_argument('--r_ep', help='distance epsilon', type=float, default=None)
parser.add_argument('--r_min', help='distance min val', type=float, default=None)
parser.add_argument('--r_max', help='distance max val', type=float, default=None)
parser.add_argument('--bagging', help='yes or no', type=str, default=None)
parser.add_argument('--output_file', help='save filename', type=str, default=None)
parser.add_argument('--verbose', help='verbose 0 or 1', type=int, default=0)

args = parser.parse_args()


setup_visibile_gpus(str(args.gpu_idx))

set_name = args.set_name
verbose = args.verbose

if set_name == 'mnist':
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

bagging = args.bagging == 'yes'
data_format = args.data_format
data = MNIST(args.data_dir, args.data_name, 0, model_meta=model_meta,
             input_data_format=CHANNELS_LAST, output_data_format=data_format)

with tf.Session(config=gpu_config) as sess:
    model = MODEL(os.path.join(args.model_dir, args.model_name), sess, output_logits=True,
                  input_data_format=data_format, data_format=data_format, dropout=args.dropout,
                  rand_params=para_random_spike, is_batch=True)

    k_model = model.model
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

    r_min = args.r_min
    r_max = args.r_max
    r_ep = args.r_ep
    batch_size = args.batch_size

    def evaluate(x, y, iter1, iter2, bagging, r_val):
        data_len = len(y)
        total_pred = None
        total_loss = 0
        avg_loss = 0
        avg_acc = 0

        for i in range(iter1):
            cur_data = sess.run(rand_x, feed_dict={para_x: x, para_r_dis: r_val})
            cur_pred = None
            cur_loss = 0

            for j in range(iter2):
                predict_label = k_model.predict(x=cur_data, batch_size=batch_size, verbose=verbose)
                computed_loss = loss.eval(session=sess, feed_dict={y_correct: y, y_predict: predict_label})
                cur_loss += np.sum(computed_loss)

                if bagging:
                    batch_pred = predict_label
                else:
                    batch_pred = (np.argmax(predict_label, 1)[:, None] == label_val).astype(np.float32)
                if cur_pred is None:
                    cur_pred = batch_pred
                else:
                    cur_pred += batch_pred

            cur_pred = (np.argmax(cur_pred, 1)[:, None] == label_val).astype(np.float32)

            total_loss += cur_loss

            if total_pred is None:
                total_pred = cur_pred
            else:
                total_pred += cur_pred

            avg_acc = np.mean(np.equal(np.argmax(total_pred, 1), np.argmax(y, 1)))
            avg_loss = total_loss / iter2 / data_len / (i+1)
            cur_acc = np.mean(np.equal(np.argmax(cur_pred, 1), np.argmax(y, 1)))
            cur_loss = cur_loss / iter2 / data_len

            print("iter %d/%d cur_loss: %.4f - cur_acc: %.4f - avg_loss: %.4f - avg_acc: %.4f" %
                  (i+1, iter1, cur_loss, cur_acc, avg_loss, avg_acc), end="\n")

        return avg_loss, avg_acc


    output_file = args.output_file
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    fp = open(output_file, 'wb')
    fp.write("name\tr_dist\ttest_acc\ttest_loss\n".encode())

    timestart = time.time()

    print("* test data reference*")
    re_loss, re_acc = evaluate(x=data.test_data, y=data.test_labels,
                               iter1=1, iter2=args.iter2, bagging=bagging, r_val=0)
    print("\navg_loss: %.4f - avg_acc: %.4f" % (re_loss, re_acc))
    fp.write(args.model_name.encode())
    fp.write(('\t0.00\t%.4f\t%.4f\n' % (re_acc, re_loss)).encode())

    for r_val in np.arange(r_min, r_max, r_ep):
        print("* test data distance: %.3f *" % r_val)
        re_loss, re_acc = evaluate(x=data.test_data, y=data.test_labels,
                                   iter1=args.iter1, iter2=args.iter2, bagging=bagging, r_val=r_val)
        fp.write(args.model_name.encode())
        fp.write(('\t%.3f\t%.4f\t%.4f\n' % (r_val, re_acc, re_loss)).encode())

    timeend = time.time()
    print("Took", timeend - timestart, "seconds to run")
