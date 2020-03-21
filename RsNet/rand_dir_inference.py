import numpy as np
import tensorflow as tf
import os
import argparse
import time
import sys
import utils
from tensorflow.keras.optimizers import SGD

import worker
from RsNet.tf_config import gpu_config, setup_visibile_gpus, CHANNELS_LAST
from RsNet.setup_mnist import MNIST, MNISTModel, FASHIONModel, CIFAR10Model, SQModels
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta
from RsNet.worker import SimpleReformer
from utils import softmax_cross_entropy_with_logits
from RsNet.ciphernet import image_jpeg_compression

parser = argparse.ArgumentParser(description='inference the example with random direction noise')

parser.add_argument('--dropout', help='dropout value', type=float, default=0)
parser.add_argument('--bagging', help='whether using bagging or not', type=str, default='True')
parser.add_argument('--rspike_para', help='rand spike parameter, for mnist only', type=str, default=None)
parser.add_argument('--rspike_layer', help='rand spike enabled layer, for cifar and fashion', type=int, default=0)
parser.add_argument('--rspike_isbatch', help='whether using batch in rspike', type=str, default='yes')
parser.add_argument('--gpu_idx', help='gpu idx used in the evaluation', type=int, default=0)
parser.add_argument('--batch_size', help='batch_size', type=int, default=50)
parser.add_argument('--iteration', help='number of iteration for predicting one instance', type=int, default=1)
parser.add_argument('--noise_distance', help='l2_distance', type=float, default=0)
parser.add_argument('--noise_iter', help='add noise times', type=int, default=1)
parser.add_argument('--noise_mthd', help='add noise method', type=str, default='l2')
parser.add_argument('--test_data_start', help='start_test_data_idx', type=int, default=0)
parser.add_argument('--test_data_len', help='test_data_length', type=int, default=100)
parser.add_argument('--output_file', help='output file, default stdout', type=str, default=None)
parser.add_argument('--is_enc', help='whether to encrypted data', type=str, default='no')
parser.add_argument('--enc_grp_chn', help='whether to encrypt the pixel by grouping channel', type=str, default='no')
parser.add_argument('--force_gray', help='whether to train with gray scale data', type=str, default='no')
parser.add_argument('--bit_depth', help='train image with reduced bit depth, [1-8], default 8', type=int, default=8)
parser.add_argument('--palette_shade', help='train image with color palette, cubic of [2-6] colors, default: -1, '
                                            'disable this feature', type=int, default=-1)
parser.add_argument('--is_1bit', help='whether train on one bit dithered image', type=str, default='no')
parser.add_argument('--det_model_dir', help='detect model dir', type=str, default='')
parser.add_argument('--det_model_names', help='detect model names', type=str, default='')
parser.add_argument('--reformer_name', help='detect model reformers', type=str, default='')
parser.add_argument('--is_det_joint', help='whether use one threshold for all detectors', type=str, default='no')
parser.add_argument('--is_logits', help='whether tp apply bagging with logits or softmax', type=str, default='yes')

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
    para_random_spike = parse_rand_spike(args.rspike_para) if args.rspike_para is not None else []
elif set_name == 'fashion':
    model_meta = model_mnist_meta
    MODEL = FASHIONModel
    para_random_spike = args.rspike_layer
elif set_name == "cifar10":
    model_meta = model_cifar10_meta
    MODEL = CIFAR10Model
    para_random_spike = args.rspike_layer
else:
    model_meta = None
    MODEL = None
    para_random_spike = None
    print("invalid data set name %s" % set_name)
    exit(0)

save_model_dir = args.model_dir
batch_size = args.batch_size
bagging = args.bagging == 'True'
iteration = args.iteration
noise_mthd = args.noise_mthd
noise_iter = args.noise_iter
noise_distance = args.noise_distance
rspike_isbatch = args.rspike_isbatch == 'yes'
data_start = args.test_data_start
data_len = args.test_data_len
data_format = args.data_format
encrypt = args.is_enc == 'yes'
enc_grp_chn = args.enc_grp_chn == 'yes'
force_gray = args.force_gray == 'yes'
bit_depth = args.bit_depth
palette_shade = args.palette_shade
is_one_bit = args.is_1bit == 'yes'
is_det_joint = args.is_det_joint == 'yes'
is_logits = args.is_logits == 'yes'

if palette_shade == -1:
    palette_shade = None
elif palette_shade > 6 or palette_shade < 2:
    print("Error: invalid palette shade value", palette_shade, ". Possible value [-1, 2-6]")
    exit(0)

model_list = args.model_name.split(",")
det_model_dir = args.det_model_dir
det_model_names = args.det_model_names.split(",")
det_model_reformers = args.reformer_name.split(',')

data = MNIST(args.data_dir, args.data_name, model_meta=model_meta, validation_size=5000,
             input_data_format=CHANNELS_LAST, output_data_format=data_format)

if (args.output_file is not None) and ( not os.path.exists(os.path.dirname(args.output_file)) ):
    os.makedirs(os.path.dirname(args.output_file))

fp = open(args.output_file, 'w') if args.output_file is not None else sys.stdout
fp.write("name\tl2\ttest_acc\ttest_unchg\ttest_loss\n")


with tf.Session(config=gpu_config) as sess:
    # cache model names
    models_by_name = {_model_name: _model_name for _model_name in model_list}

    models_by_id = {}
    idx = 0
    for _model_name in model_list:
        models_by_id[idx] = _model_name
        idx += 1

    # cache detector names
    det_dict = {}
    for val in det_model_names:
        if val == '':
            continue
        cur_name, cur_p, cur_det_type, cur_dropout_rate, cur_model_id = val.split('/')

        cur_model_name = models_by_id[int(cur_model_id)]
        if cur_model_name not in det_dict:
            det_dict[cur_model_name] = []

        det_dict[cur_model_name].append(val)
    # # cache detector names
    # det_dict = {}
    # for val in det_model_names:
    #     if val == '':
    #         continue
    #     cur_name, cur_p, cur_det_type, cur_dropout_rate, cur_model_id = val.split('/')
    #     cur_det = {
    #         "p": cur_p,
    #         "type": cur_det_type,
    #         "dropout_rate": cur_dropout_rate
    #     }
    #
    #     cur_model_name = models_by_id[int(cur_model_id)]
    #     if cur_model_name not in det_dict:
    #         det_dict[cur_model_name] = {}
    #
    #     det_dict[cur_model_name][cur_name] = cur_det

    # cache reformers
    reformers = {}
    reformer_id = 0
    for cur_reformer_name in det_model_reformers:
        if cur_reformer_name == '':
            reformer_id += 1
            continue
        cur_model_name = models_by_id[reformer_id]
        reformer_id += 1
        reformers[cur_model_name] = cur_reformer_name

    timestart = time.time()
    enc_cfg_pth = os.path.join(save_model_dir, model_list[0] + '.json')
    eva_model = MODEL(None, sess, output_logits=True,
                      input_data_format=data_format, data_format=data_format, dropout=args.dropout,
                      rand_params=para_random_spike, is_batch=rspike_isbatch, encrypt=encrypt, enc_cfg_path=enc_cfg_pth,
                      enc_grp_chn=enc_grp_chn,
                      force_gray=force_gray, bit_depth=bit_depth, palette_shade=palette_shade, one_bit=is_one_bit)
    det_model = MODEL(None, sess, output_logits=True,
                      input_data_format=data_format, data_format=data_format, is_batch=rspike_isbatch,
                      encrypt=encrypt, enc_cfg_path=enc_cfg_pth, enc_grp_chn=enc_grp_chn,
                      force_gray=force_gray, bit_depth=bit_depth, palette_shade=palette_shade, one_bit=is_one_bit)

    for model_name in model_list:
        print("=============================")
        print("valid rand_inference for model %s" % model_name)
        print("=============================")
        fp.write(model_name)
        fp.write("\t%.4f" % noise_distance)

        # get current model
        cur_path = os.path.join(save_model_dir, model_name)
        eva_model.load_weights(cur_path)
        det_model.load_weights(cur_path)
        cur_model = eva_model
        cur_model_idx = utils.load_model_idx(cur_path)
        if cur_model_idx is None:
            cur_model_idx = utils.save_model_idx(cur_path, data)

        # restore the key for current key
        if encrypt:
            eva_model.restore_key(cur_path)
            det_model.restore_key(cur_path)

        cur_det_dict = det_dict[model_name] if model_name in det_dict else {}
        cur_det_set, cur_thrs_set, cur_det_gpu_idx = \
            worker.build_detector(det_model_dir, cur_det_dict,
                                  model_name, save_model_dir, cur_path,
                                  MODEL, det_model, data, data_format, is_det_joint, cur_model_idx)
        # # construct detector and dropout_rate
        # cur_det_dict = det_dict[model_name] if model_name in det_dict else {}
        # cur_det_set = {}
        # cur_dropout_rate_set = {}

        # for cur_det_name, cur_det_conf in cur_det_dict.items():
        #
        #     cur_p = cur_det_conf['p']
        #     cur_det_type = cur_det_conf['type']
        #     cur_dropout_rate = cur_det_conf['dropout_rate']
        #
        #     # build detector
        #     print("# build detector: ", cur_det_name)
        #     print("type:", cur_det_type)
        #     print("p:", cur_p)
        #     print("drop_rate:", cur_dropout_rate)
        #     if cur_det_type == 'AED':
        #         cur_detector = AEDetector(os.path.join(det_model_dir, cur_det_name), p=int(cur_p))
        #     elif cur_det_type == "DBD":
        #         id_reformer = IdReformer()
        #         print("# build reformer", cur_det_name)
        #         cur_reformer_t = SimpleReformer(os.path.join(det_model_dir, cur_det_name))
        #         classifier = Classifier(os.path.join(save_model_dir, model_name), MODEL,
        #                                 data_format=data_format, model=det_model)
        #         cur_detector = DBDetector(reconstructor=id_reformer, prober=cur_reformer_t,
        #                                   classifier=classifier, T=int(cur_p))
        #     elif cur_det_type.startswith("bit_depth_"):
        #         bits = int(cur_det_type[len("bit_depth_"):])
        #         cur_detector = BitDepthDetector(cur_model.model, p=int(cur_p), bits=bits, verbose=1)
        #     elif cur_det_type.startswith("median_filter_"):
        #         mat = cur_det_type[len("median_filter_"):].split("_")
        #         cur_detector = MedianDetector(cur_model.model, height=int(mat[0]), width=int(mat[1]),
        #                                       p=int(cur_p), verbose=1)
        #
        #     cur_dropout_rate_set[cur_det_name] = float(cur_dropout_rate)
        #     cur_det_set[cur_det_name] = cur_detector
        #
        # # compute thrs
        # cur_thrs_set = {}
        # validation_data = data.train_data[:5000]
        # for cur_det_name, cur_det in cur_det_set.items():
        #     num = int(len(validation_data) * cur_dropout_rate_set[cur_det_name])
        #     marks = cur_det.mark(validation_data, data_format=data_format)
        #     marks = np.sort(marks)
        #     cur_thrs_set[cur_det_name] = marks[-num]
        #
        #     print("compute thrs for model #", cur_det_name, "#:", marks[-num])

        # concat reformer in front of models
        if model_name in reformers:
            cur_reformer_name = reformers[model_name]
            cur_reformer = SimpleReformer(os.path.join(det_model_dir, cur_reformer_name))
            print("concat reformer", cur_reformer_name, "before model", model_name)
            cur_model = SQModels(model_list=[cur_reformer.model, cur_model.model], ref_model=cur_model)

        k_model = cur_model.model
        sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)

        k_model.compile(loss=softmax_cross_entropy_with_logits,
                        optimizer=sgd,
                        metrics=['accuracy'])

        y_correct = tf.placeholder(tf.float32, [None, model_meta.labels])
        y_predict = tf.placeholder(tf.float32, [None, model_meta.labels])
        label_val = np.arange(model_meta.labels)

        loss = softmax_cross_entropy_with_logits(y_correct, y_predict)
        out_softmax = tf.nn.softmax(y_predict)

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
            #init_noise = np.random.random_sample(np.shape(data))-0.5
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
            cur_pred = None
            cur_loss = 0

            # detector checking:
            cur_det_adv = np.array([False] * len(data))

            for det_name, det in cur_det_set.items():
                cur_thrs = cur_thrs_set[det_name]

                cur_det_batch_adv = det.mark(data, data_format=data_format) >= cur_thrs

                cur_det_adv |= cur_det_batch_adv

            for j in range(iteration):
                predict_label = k_model.predict_on_batch(x=data)
                computed_loss = loss.eval(session=sess, feed_dict={y_correct: correct_label, y_predict: predict_label})
                cur_loss += np.sum(computed_loss)
                if bagging:
                    if not is_logits:
                        predict_label = out_softmax.eval(session=sess, feed_dict={y_predict: predict_label})
                    batch_pred = predict_label
                else:
                    batch_pred = (np.argmax(predict_label, 1)[:, None] == label_val).astype(np.float32)

                if cur_pred is None:
                    cur_pred = batch_pred
                else:
                    cur_pred += batch_pred

            return cur_pred, cur_loss, cur_det_adv

        def evaluate(x, y):

            data_len = len(y)
            total_pred = []
            total_unchg_pred = []
            total_loss = 0
            avg_loss = 0
            avg_acc = 0
            avg_unchg = 0

            for i in range(0, data_len, batch_size):
                end_idx = i + batch_size
                if end_idx > data_len:
                    end_idx = data_len

                cur_batch_size = end_idx - i
                cur_data = x[i: end_idx]
                cur_label = y[i: end_idx]
                correct_label = np.argmax(cur_label, 1)

                # make a ground truth of current model
                ref_pred, ref_loss, ref_det_adv = predict_data(cur_data, cur_label)
                ref_label = np.argmax(ref_pred, 1)

                # predict on noise data
                for noise_i in range(noise_iter):
                    noise_data = add_noise_data_by_mthd(cur_data, noise_mthd, noise_distance)
                    cur_pred, cur_loss, cur_det_adv = predict_data(noise_data, cur_label)

                    cur_pred_label = np.argmax(cur_pred, 1)
                    cur_pred = np.equal(cur_pred_label, correct_label)
                    cur_unchg_pred = np.equal(cur_pred_label, ref_label)
                    for adv_i in range(len(ref_det_adv)):
                        # detect the image is an adv image which is not in this case since all images are test images
                        #
                        # if both detector result are same:
                        # then
                        #     if cur detector detect the example as adv
                        #     then
                        #         cur_pred is false
                        #         cur_unchg is true since both model think the example is an adv
                        #     else
                        #         cur_pred result depends on correct_label
                        #         cur_unchg result depends on ref_label
                        # else
                        #     cur_unchg is false since both the detector result of two models are different
                        #
                        #     if cur detector detect the example as adv
                        #     then
                        #         cur_pred is false
                        #     else
                        #         cur_pred result depends on correct_label
                        if cur_det_adv[adv_i] == ref_det_adv[adv_i]:
                            if cur_det_adv[adv_i]:
                                cur_pred[adv_i] = False
                                cur_unchg_pred[adv_i] = True
                        else:
                            cur_unchg_pred[adv_i] = False
                            if cur_det_adv[adv_i]:
                                cur_pred[adv_i] = False

                    total_loss += cur_loss
                    total_pred.extend(cur_pred)
                    total_unchg_pred.extend(cur_unchg_pred)

                    avg_acc = np.mean(total_pred)
                    avg_unchg = np.mean(total_unchg_pred)
                    avg_loss = total_loss / end_idx / iteration / noise_iter
                    cur_acc = np.mean(cur_pred)
                    cur_unchg = np.mean(cur_unchg_pred)
                    cur_loss = cur_loss / cur_batch_size / iteration / noise_iter

                    print("%d/%d iter: %d/%d cur_loss: %.4f - cur_acc: %.4f - cur_unchg: %.4f"
                          " - avg_loss: %.4f - avg_acc: %.4f - avg_unchg: %.4f" %
                          (end_idx, data_len, noise_i, noise_iter,
                           cur_loss, cur_acc, cur_unchg, avg_loss, avg_acc, avg_unchg), np.shape(total_pred), end="\r")

            return avg_loss, avg_acc, avg_unchg


        print("evaluating model:", model_name)
        print("* test data *")
        re_loss, re_acc, re_unchg = evaluate(x=data.test_data[data_start: data_start+data_len],
                                             y=data.test_labels[data_start: data_start+data_len])
        print("avg_loss: %.4f - avg_acc: %.4f - avg_acc: %.4f" % (re_loss, re_acc, re_unchg))
        fp.write('\t%.4f\t%.4f\t%.4f\n' % (re_acc, re_unchg, re_loss))

    timeend = time.time()
    print("Took", timeend - timestart, "seconds to run")

    fp.close()
