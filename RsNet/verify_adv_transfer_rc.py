import argparse
import os
import gzip
import json
import utils
import tensorflow as tf
import numpy as np
from RsNet.tf_config import gpu_config
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta
from RsNet.setup_mnist import MNIST, MNISTModel, FASHIONModel, CIFAR10Model
from RsNet.tf_config import setup_visibile_gpus, CHANNELS_LAST, CHANNELS_FIRST

parser = argparse.ArgumentParser(description='Summarize the result of random direction inference.')

parser.add_argument('--data_dir', help='data dir, required', type=str, default=None)
parser.add_argument('--data_name', help='data name, required', type=str, default=None)
parser.add_argument('--attack_name', help='attack name, required', type=str, default=None)
parser.add_argument('--model_dir', help='model directory, required', type=str, default=None)
parser.add_argument('--model_name', help='model name, required', type=str, default=None)
parser.add_argument('-s', '--set_name', help='set name, mnist, fashion, cifar10', type=str, default=None)
parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=None)
parser.add_argument('--data_format', help='channels_last or channels_first', type=str, default=None)
parser.add_argument('--gpu_idx', help='gpu index', type=int, default=None)
parser.add_argument('--iter1', help='iterate 1, number of sample points', type=int, default=None)
parser.add_argument('--iter2', help='iterate 2 bagging or majority', type=int, default=None)
parser.add_argument('--dropout', help='dropout rate', type=float, default=None)
parser.add_argument('--r_dis', help='noise distance', type=float, default=None)
parser.add_argument('--bagging', help='yes or no', type=str, default=None)
parser.add_argument('--output_file', help='save filename', type=str, default=None)
parser.add_argument('--verbose', help='verbose 0 or 1', type=int, default=0)
parser.add_argument('--is_test_data', help='if we test on the testing data', type=str, default='test')

args = parser.parse_args()


setup_visibile_gpus(str(args.gpu_idx))

set_name = args.set_name
batch_size = args.batch_size
verbose = args.verbose
is_test_data = args.is_test_data == 'test'

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


img_size = model_meta.width * model_meta.height * model_meta.channel
img_width = model_meta.width
img_height = model_meta.height
img_channel = model_meta.channel
img_labels = model_meta.labels
img_labels_val = np.arange(model_meta.labels)

real_dir = args.data_dir + "/" + args.data_name + "/"
config_fp = open(real_dir+"config.json", "rb")
json_str = config_fp.read()
config_fp.close()
config = json.loads(json_str.decode())

attack_name = args.attack_name
count = int(config[attack_name + '-count'] / 3)

attack_adv = real_dir + (config[attack_name + "-adv-img"])
attack_adv_img = real_dir + (config[attack_name + "-img"])
attack_adv_label = real_dir + (config[attack_name + "-adv-label"])
attack_adv_raw_label = real_dir + (config[attack_name + "-adv-raw-label"])
fp_attack_adv = gzip.open(attack_adv, "rb")
fp_attack_adv_img = gzip.open(attack_adv_img, "rb")
fp_attack_adv_label = gzip.open(attack_adv_label, "rb")
fp_attack_adv_raw_label = gzip.open(attack_adv_raw_label, "rb")

fp_attack_adv_img.read(16)

# load data
buf = fp_attack_adv.read(count * img_size * 4)
data_attack_adv = np.frombuffer(buf, dtype=np.float32)
data_attack_adv = data_attack_adv.reshape(count, img_width, img_height, img_channel)
buf = fp_attack_adv_label.read(count)
data_attack_adv_label = np.frombuffer(buf, dtype=np.uint8)
buf = fp_attack_adv_raw_label.read(count)
data_attack_adv_raw_label = np.frombuffer(buf, dtype=np.uint8)

# read adv example in image format
buf = fp_attack_adv_img.read(count * img_size * 3)
data_attack_adv_img = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data_attack_adv_img = (data_attack_adv_img / 255)
data_attack_adv_img = data_attack_adv_img.reshape(count * 3, img_width, img_height, img_channel)

data_format = args.data_format
if data_format == CHANNELS_FIRST:
    data_attack_adv = data_attack_adv.transpose([0, 3, 1, 2])
    data_attack_adv_img = data_attack_adv_img.transpose([0, 3, 1, 2])

# close loaded file
fp_attack_adv.close()
fp_attack_adv_img.close()
fp_attack_adv_label.close()
fp_attack_adv_raw_label.close()

# load image idx
adv_img_idx = utils.load_obj(attack_name + "-idx", directory=real_dir)

output_file = args.output_file
if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

if output_file.endswith(".csv"):
    output_raw = output_file[:-4] + "-raw.csv"
else:
    output_raw = output_file + "-raw.csv"
    output_filename = output_file + ".csv"

fp = open(output_file, 'wb')
fp.write("name\tfloat_adv_acc\tfloat_pred_acc\tfloat_det_acc\tfloor_adv_acc\tfloor_pred_acc\tfloor_det_acc\t"
         "ceil_adv_acc\tceil_pred_acc\tceil_det_acc\tround_adv_acc\tround_pred_acc\tround_det_acc\t"
         "attack_success_count\tattack_success_rate\n".encode())

fp_raw = open(output_raw, 'wb')
fp_raw.write("model_id\tidx\tl0\tl1\tl2\tl_inf\tfloat_pred\tfloor_pred\tceil_pred\tround_pred\tfloat_tran\t"
             "floor_tran\tceil_trans\tround_trans\tfloat_det\tfloor_det\tceil_det\tround_det\n".encode())

with tf.Session(config=gpu_config) as sess:
    data = MNIST(args.data_dir, args.data_name, model_meta=model_meta,
                 input_data_format=CHANNELS_LAST, output_data_format=data_format)
    if is_test_data:
        ref_data = data.test_data[adv_img_idx]
    else:
        ref_data = data.train_data[adv_img_idx]

    round_img_idx = np.arange(count) * 3 + 2
    diff_data = ref_data - data_attack_adv_img[round_img_idx]
    diff_data = diff_data.reshape([count, -1])
    dist_l0 = np.linalg.norm(diff_data, 0, axis=1)
    dist_l1 = np.linalg.norm(diff_data, 1, axis=1)
    dist_l2 = np.linalg.norm(diff_data, 2, axis=1)
    dist_l_inf = np.linalg.norm(diff_data, np.inf, axis=1)

    para_shape = list(data.test_data.shape)
    para_shape[0] = None
    para_x = tf.placeholder(tf.float32, para_shape)
    x_tmp = tf.random_uniform(tf.shape(para_x),
                              minval=args.r_dis, maxval=-args.r_dis, dtype=tf.float32) + para_x
    rand_x = tf.clip_by_value(x_tmp, 0, 1)

    model_name_list = args.model_name.split(',')

    model_id = 0
    eva_model = MODEL(None, sess, output_logits=True,
                      input_data_format=data_format, data_format=data_format, dropout=args.dropout,
                      rand_params=para_random_spike, is_batch=True)
    for cur_model_name in model_name_list:
        fp.write(cur_model_name.encode())
        print("=============================")
        print("valid transferability for model %s" % cur_model_name)
        print("=============================")
        # get current model
        cur_path = os.path.join(args.model_dir, cur_model_name)
        eva_model.load_weights(cur_path)
        cur_model = eva_model

        adv_work = np.array([True]*count)
        img_cate = ["floor", "ceil", "round"]

        total_pred = None
        total_pred_img = None

        total_acc = None
        total_acc_img = None
        total_adv_work = None
        total_adv_work_img = None

        raw_acc = None
        raw_acc_img = None
        raw_adv_work = None
        raw_adv_work_img = None

        for img_idx in range(0, count):
            float_img = data_attack_adv[img_idx]
            if np.sum(float_img) == 0:
                adv_work[img_idx] = False

        for i in range(args.iter1):
            cur_data_attack_adv = sess.run(rand_x, feed_dict={para_x: data_attack_adv})
            cur_data_attack_adv_img = sess.run(rand_x, feed_dict={para_x: data_attack_adv_img})
            cur_pred = None
            cur_pred_img = None

            print("%d/%d" % (i+1, args.iter1))
            for j in range(args.iter2):
                y_res = cur_model.model.predict(cur_data_attack_adv, batch_size=batch_size, verbose=verbose)
                y_res_img = cur_model.model.predict(cur_data_attack_adv_img, batch_size=batch_size, verbose=verbose)

                if args.bagging == 'yes':
                    batch_pred = y_res
                    batch_pred_img = y_res_img
                else:
                    batch_pred = (np.argmax(y_res, 1)[:, None] == img_labels_val).astype(np.float32)
                    batch_pred_img = (np.argmax(y_res_img, 1)[:, None] == img_labels_val).astype(np.float32)

                if cur_pred is None:
                    cur_pred = batch_pred
                    cur_pred_img = batch_pred_img
                else:
                    cur_pred += batch_pred
                    cur_pred_img += batch_pred_img

            cur_pred = (np.argmax(cur_pred, 1)[:, None] == img_labels_val).astype(np.float32)
            cur_pred_img = (np.argmax(cur_pred_img, 1)[:, None] == img_labels_val).astype(np.float32)

            if total_pred is None:
                total_pred = cur_pred
                total_pred_img = cur_pred_img
            else:
                total_pred += cur_pred
                total_pred_img += cur_pred_img

            tmp_total_pred = np.argmax(total_pred, 1)
            tmp_total_pred_img = np.reshape(np.argmax(total_pred_img, 1), [count, 3]).transpose()

            raw_acc = (tmp_total_pred == data_attack_adv_raw_label) & adv_work
            raw_acc_img = np.array([
                (tmp_total_pred_img[0] == data_attack_adv_raw_label) & adv_work,
                (tmp_total_pred_img[1] == data_attack_adv_raw_label) & adv_work,
                (tmp_total_pred_img[2] == data_attack_adv_raw_label) & adv_work
            ])
            raw_adv_work = (tmp_total_pred == data_attack_adv_label) & adv_work
            raw_adv_work_img = np.array([
                (tmp_total_pred_img[0] == data_attack_adv_label) & adv_work,
                (tmp_total_pred_img[1] == data_attack_adv_label) & adv_work,
                (tmp_total_pred_img[2] == data_attack_adv_label) & adv_work
            ])

            total_acc = np.mean(raw_acc)
            total_acc_img = np.mean(raw_acc_img, axis=1)
            total_adv_work = np.mean(raw_adv_work)
            total_adv_work_img = np.mean(raw_adv_work_img, axis=1)

            print("acc adv work: %.4f acc correct predicted: %.4f" % (total_adv_work, total_acc))
            for idx in range(3):
                print("acc %s adv work: %.4f acc %s correct predicted: %.4f" %
                      (img_cate[idx], total_adv_work_img[idx], img_cate[idx], total_acc_img[idx]))

        fp.write(('\t%.4f\t%.4f\t%.4f' % (total_adv_work, total_acc, 0)).encode())
        for idx in range(3):
            fp.write(('\t%.4f\t%.4f\t%.4f' % (total_adv_work_img[idx], total_acc_img[idx], 0)).encode())
        adv_work_count = np.sum(adv_work)
        fp.write(('\t%d/%d\t%.4f\n' % (adv_work_count, count, adv_work_count / count)).encode())

        # save raw data for each adv example: L2 etc, whether transfer or not, whether it can be classified correctly
        raw_acc = raw_acc.astype(int)
        raw_acc_img = raw_acc_img.astype(int)
        raw_adv_work = raw_adv_work.astype(int)
        raw_adv_work_img = raw_adv_work_img.astype(int)

        for img_idx in range(0, count):
            if not adv_work[img_idx]:
                continue
            fp_raw.write(("%d\t%d\t%d\t%.4f\t%.4f\t%.4f" %
                          (model_id, adv_img_idx[img_idx], dist_l0[img_idx],
                           dist_l1[img_idx], dist_l2[img_idx], dist_l_inf[img_idx])).encode())
            fp_raw.write(("\t%d\t%d\t%d\t%d" %
                          (raw_acc[img_idx], raw_acc_img[0][img_idx],
                           raw_acc_img[1][img_idx], raw_acc_img[2][img_idx])).encode())
            fp_raw.write(("\t%d\t%d\t%d\t%d" %
                          (raw_adv_work[img_idx], raw_adv_work_img[0][img_idx],
                           raw_adv_work_img[1][img_idx], raw_adv_work_img[2][img_idx])).encode())
            fp_raw.write(("\t%d\t%d\t%d\t%d\n" % (0, 0, 0, 0)).encode())

        model_id += 1

fp.close()
fp_raw.close()
