import os
import gzip
import json
import utils
import argparse
import tensorflow as tf
import numpy as np

import worker
from RsNet.tf_config import gpu_config
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta
from RsNet.setup_mnist import MNIST, MNISTModel, FASHIONModel, CIFAR10Model, SQModels
from RsNet.tf_config import setup_visibile_gpus, CHANNELS_LAST, CHANNELS_FIRST
from RsNet.worker import SimpleReformer


def parse_rand_spike(_str):
    _str = _str.split(',')
    return [float(x) for x in _str]


parser = argparse.ArgumentParser(description='Verify adv example transferability')

parser.add_argument('--data_dir', help='data dir, required', type=str, default=None)
parser.add_argument('--data_name', help='data name, required', type=str, default=None)
parser.add_argument('--attack_name', help='attack name, required', type=str, default=None)
parser.add_argument('--model_dir', help='save model directory, required', type=str, default=None)
parser.add_argument('--model_name', help='save model name, required', type=str, default=None)
parser.add_argument('--det_model_dir', help='detector model directory, required', type=str, default='')
parser.add_argument('--det_model_names', help='detector model names, required', type=str, default='')
parser.add_argument('--reformer_name', help='reformer name, required', type=str, default='')
parser.add_argument('--set_name', help='set name [mnist, fashion, cifar10], required', type=str, default=None)
parser.add_argument('--batch_size', help='batch_size', type=int, default=500)
parser.add_argument('--data_format', help='channels_last or channels_first', type=str, default=CHANNELS_LAST)
parser.add_argument('--gpu_idx', help='gpu indexs', type=int, default=0)
parser.add_argument('--output_file', help='save filename', type=str, default=None)
parser.add_argument('--iteration', help='iteration, number of prediction per instance', type=int, default=1)
parser.add_argument('--dropout', help='dropout rate', type=float, default=0)
parser.add_argument('--bagging', help='yes or no', type=str, default='no')
parser.add_argument('--random_spike', help='parameter used for random spiking', type=str, default=None)
parser.add_argument('--is_targeted', help='whether the attack is targeted attack, yes or no', type=str, default='yes')
parser.add_argument('--ref_data', help='reference data set, [test, train], default: test', type=str, default='test')
parser.add_argument('--is_det_joint', help='whether use one threshold for all detectors', type=str, default='no')
parser.add_argument('--is_logits', help='whether tp apply bagging with logits or softmax', type=str, default='yes')
parser.add_argument('--boxmin', help='model input image value min', type=float, default=0)
parser.add_argument('--boxmax', help='model input image value max', type=float, default=1.0)
parser.add_argument('--top_k', help='evaluate extra top_k task', type=int, default=1)


args = parser.parse_args()

data_dir = args.data_dir
data_name = args.data_name
attack_name = args.attack_name
save_model_dir = args.model_dir
save_model_name_list = args.model_name.split(",")
detector_model_dir = args.det_model_dir
detector_model_names = args.det_model_names.split(",")
reformer_names = args.reformer_name.split(',')
set_name = args.set_name
batch_size = args.batch_size
data_format = args.data_format
gpu_idx = args.gpu_idx
output_filename = args.output_file
iteration = args.iteration
dropout = args.dropout
bagging = args.bagging == 'yes'
para_random_spike = args.random_spike
targeted = args.is_targeted == 'yes'
is_test_data = args.ref_data == 'test'
is_det_joint = args.is_det_joint == 'yes'
is_logits = args.is_logits == 'yes'
boxmin = args.boxmin
boxmax = args.boxmax
top_k = args.top_k

setup_visibile_gpus(str(gpu_idx))

if set_name == 'mnist':
    model_meta = model_mnist_meta
    MODEL = MNISTModel
    para_random_spike = None if para_random_spike is None else parse_rand_spike(para_random_spike)
elif set_name == 'fashion':
    model_meta = model_mnist_meta
    MODEL = FASHIONModel
    para_random_spike = 0 if para_random_spike is None else int(para_random_spike)
elif set_name == "cifar10":
    model_meta = model_cifar10_meta
    MODEL = CIFAR10Model
    para_random_spike = 0 if para_random_spike is None else int(para_random_spike)
else:
    model_meta = None
    MODEL = None
    print("invalid data set name %s" % set_name)
    exit(0)

img_size = model_meta.width * model_meta.height * model_meta.channel
img_width = model_meta.width
img_height = model_meta.height
img_channel = model_meta.channel
img_labels = model_meta.labels
img_labels_val = np.arange(model_meta.labels)

if 0 <= img_labels <= 255:
    label_data_type = np.uint8
else:
    label_data_type = np.uint16
label_data_size = np.dtype(label_data_type).itemsize


real_dir = data_dir + "/" + data_name + "/"
config_fp = open(real_dir+"config.json", "rb")
json_str = config_fp.read()
config_fp.close()
config = json.loads(json_str.decode())

count = int(config[attack_name + '-count'] / 3)

attack_adv = real_dir + (config[attack_name + "-adv-img"])
attack_adv_img = real_dir + (config[attack_name + "-img"])
attack_adv_label = real_dir + (config[attack_name + "-adv-label"])
attack_adv_raw_label = real_dir + (config[attack_name + "-adv-raw-label"])
fp_attack_adv = gzip.open(attack_adv, "rb")
fp_attack_adv_img = gzip.open(attack_adv_img, "rb")
fp_attack_adv_label = gzip.open(attack_adv_label, "rb")
fp_attack_adv_raw_label = gzip.open(attack_adv_raw_label, "rb")

# load image idx
adv_img_idx = utils.load_obj(attack_name + "-idx", directory=real_dir)

if not os.path.exists(os.path.dirname(output_filename)):
    os.makedirs(os.path.dirname(output_filename))

if output_filename.endswith(".csv"):
    output_raw = output_filename[:-4] + "-raw.csv"
else:
    output_raw = output_filename + "-raw.csv"
    output_filename = output_filename + ".csv"

fp = open(output_filename, 'wb')
fp.write("name\tfloat_adv_acc\tfloat_pred_acc\tfloat_pred_acc_topk\tfloat_det_acc\t"
         "floor_adv_acc\tfloor_pred_acc\tfloor_pred_acc_topk\tfloor_det_acc\t"
         "ceil_adv_acc\tceil_pred_acc\tceil_pred_acc_topk\tceil_det_acc\t"
         "round_adv_acc\tround_pred_acc\tround_pred_acc_topk\tround_det_acc\t"
         "attack_success_count\tattack_success_rate\n".encode())

fp_raw = open(output_raw, 'wb')
fp_raw.write("model_id\tidx\tl0\tl1\tl2\tl_inf\tfloat_pred\tfloor_pred\tceil_pred\tround_pred\tfloat_tran\t"
             "floor_tran\tceil_trans\tround_trans\tfloat_det\tfloor_det\tceil_det\tround_det"
             "\tfloat_pred_topk\tfloor_pred_topk\tceil_pred_topk\tround_pred_topk\n".encode())

gpu_thread_count = 2
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

with tf.Session(config=gpu_config) as sess:
    data = MNIST(data_dir, data_name, model_meta=model_meta, validation_size=5000,
                 input_data_format=CHANNELS_LAST, output_data_format=data_format, batch_size=batch_size)

    data_type = 'test' if is_test_data else 'train'

    # cache model names
    models_by_name = {_model_name: _model_name for _model_name in save_model_name_list}

    models_by_id = {}
    idx = 0
    for _model_name in save_model_name_list:
        models_by_id[idx] = _model_name
        idx += 1

    # cache detector names
    detector_dict = {}
    for val in detector_model_names:
        if val == '':
            continue
        cur_name, cur_p, cur_det_type, cur_dropout_rate, cur_model_id = val.split('/')

        cur_model_name = models_by_id[int(cur_model_id)]
        if cur_model_name not in detector_dict:
            detector_dict[cur_model_name] = []

        detector_dict[cur_model_name].append(val)

    # cache reformers
    reformers = {}
    reformer_id = 0
    for cur_reformer_name in reformer_names:
        if cur_reformer_name == '':
            reformer_id += 1
            continue
        cur_model_name = models_by_id[reformer_id]
        reformer_id += 1
        reformers[cur_model_name] = cur_reformer_name

    model_id = 0
    eva_model = MODEL(None, sess, output_logits=is_logits,
                      input_data_format=data_format, data_format=data_format, dropout=dropout,
                      rand_params=para_random_spike, is_batch=True,
                      cmin=boxmin, cmax=boxmax)
    det_model = MODEL(None, sess, output_logits=True,
                      input_data_format=data_format, data_format=data_format, is_batch=True,
                      cmin=boxmin, cmax=boxmax)
    y_targets = tf.placeholder(tf.int32, [None])
    y_predict = tf.placeholder(tf.float32, [None, model_meta.labels])
    out_top_k = tf.nn.in_top_k(y_predict, y_targets, top_k)

    for cur_save_model_name in save_model_name_list:
        fp.write(cur_save_model_name.encode())
        print("=============================")
        print("valid transferability for model %s" % cur_save_model_name)
        print("=============================")
        # get current model
        cur_path = os.path.join(save_model_dir, cur_save_model_name)
        eva_model.load_weights(cur_path)
        det_model.load_weights(cur_path)
        cur_model = eva_model
        cur_model_idx = utils.load_model_idx(cur_path)
        if cur_model_idx is None:
            cur_model_idx = utils.save_model_idx(cur_path, data)

        # restore the key for current key
        # prepare encrypted data if we need to evaluate multi times
        # if encrypt and iteration > 1:
        #     enc_obj.restore_key(cur_path)
        #     data_attack_adv = enc_obj.enc_tf(sess, bak_data_attack_adv, normalize=True, batch_size=batch_size)
        #     data_attack_adv_img = enc_obj.enc_tf(sess, bak_data_attack_adv_img, normalize=True, batch_size=batch_size)

        cur_det_dict = detector_dict[cur_save_model_name] if cur_save_model_name in detector_dict else {}
        cur_det_set, cur_thrs_set, cur_det_gpu_idx = \
            worker.build_detector(detector_model_dir, cur_det_dict,
                                  cur_save_model_name, save_model_dir, cur_path,
                                  MODEL, det_model, data, data_format, is_det_joint, cur_model_idx)

        # concat reformer in front of models
        if cur_save_model_name in reformers:
            cur_reformer_name = reformers[cur_save_model_name]
            cur_reformer = SimpleReformer(os.path.join(detector_model_dir, cur_reformer_name))
            print("concat reformer", cur_reformer_name, "before model", cur_save_model_name)
            cur_model = SQModels(model_list=[cur_reformer.model, cur_model.model], ref_model=cur_model)

        print("start verify transferability")

        adv_work_count = 0
        total_acc = 0
        total_acc_topk = 0
        adv_work_acc = 0
        adv_det_acc = 0
        img_pred_acc = [0, 0, 0]
        img_pred_acc_topk = [0, 0, 0]
        img_adv_work_acc = [0, 0, 0]
        img_adv_det_acc = [0, 0, 0]
        img_cate = ["floor", "ceil", "round"]

        fp_attack_adv.seek(0)
        fp_attack_adv_img.seek(16)
        fp_attack_adv_label.seek(0)
        fp_attack_adv_raw_label.seek(0)

        for i in range(0, count, batch_size):
            num_images = batch_size if i+batch_size <= count else count-i

            ref_data, _, _ = data.get_data_by_idx(adv_img_idx[i: i + num_images], data_type=data_type)
            ref_data = (ref_data - boxmin) / (boxmax - boxmin)
            # load data
            buf = fp_attack_adv.read(num_images * img_size * 4)
            batch_adv = np.frombuffer(buf, dtype=np.float32).reshape(num_images, img_width, img_height, img_channel)
            buf = fp_attack_adv_label.read(num_images * label_data_size)
            batch_adv_label = np.frombuffer(buf, dtype=label_data_type)
            buf = fp_attack_adv_raw_label.read(num_images * label_data_size)
            batch_adv_raw_label = np.frombuffer(buf, dtype=label_data_type) #data_attack_adv_raw_label[i:i+num_images]
            # read adv example in image format
            buf = fp_attack_adv_img.read(num_images * img_size * 3)
            batch_adv_img = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            batch_adv_img = (batch_adv_img / 255) * (boxmax - boxmin) + boxmin
            batch_adv_img = batch_adv_img.reshape(num_images * 3, img_width, img_height, img_channel)

            batch_adv_img_orig = batch_adv_img.copy()
            # imagenet apply dithering

            if data_format == CHANNELS_FIRST:
                batch_adv = batch_adv.transpose([0, 3, 1, 2])
                batch_adv_img = batch_adv_img.transpose([0, 3, 1, 2])
                batch_adv_img_orig = batch_adv_img_orig.transpose([0, 3, 1, 2])

            res_label = np.zeros([num_images, img_labels])
            res_img_label = np.zeros([num_images * 3, img_labels])

            # detector checking:
            det_batch_adv = np.array([False] * num_images)
            det_batch_adv_img = np.array([False] * (num_images * 3))

            for det_name, det in cur_det_set.items():
                cur_thrs = cur_thrs_set[det_name]

                cur_det_batch_adv = det.mark(batch_adv, data_format=data_format) >= cur_thrs
                cur_det_batch_adv_img = det.mark(batch_adv_img, data_format=data_format) >= cur_thrs

                det_batch_adv |= cur_det_batch_adv
                det_batch_adv_img |= cur_det_batch_adv_img

            det_batch_adv_img = np.reshape(det_batch_adv_img, (num_images, 3))

            for j in range(0, iteration):
                y_res = cur_model.model.predict(batch_adv)
                y_res_img = cur_model.model.predict(batch_adv_img)
                if bagging:
                    res_label += y_res
                    res_img_label += y_res_img
                else:
                    tmp_res_label = np.argmax(y_res, 1)
                    res_label += (img_labels_val == tmp_res_label[:, None]).astype(np.float32)
                    tmp_res_img_label = np.argmax(y_res_img, 1)
                    res_img_label += (img_labels_val == tmp_res_img_label[:, None]).astype(np.float32)

            res_img_topk = out_top_k.eval(session=sess, feed_dict={
                y_targets: np.repeat(batch_adv_raw_label, 3), y_predict: res_img_label})
            res_img_topk = res_img_topk.reshape(num_images, 3)
            res_img_label = np.argmax(res_img_label, 1)
            res_img_label = res_img_label.reshape(num_images, 3)

            res_label_topk = out_top_k.eval(session=sess,
                                            feed_dict={y_targets: batch_adv_raw_label, y_predict: res_label})
            res_label = np.argmax(res_label, 1)

            round_img_idx = np.arange(num_images) * 3 + 2
            diff_data = ref_data - batch_adv_img_orig[round_img_idx]
            diff_data = diff_data.reshape([num_images, -1])
            dist_l0 = np.linalg.norm(diff_data, 0, axis=1)
            dist_l1 = np.linalg.norm(diff_data, 1, axis=1)
            dist_l2 = np.linalg.norm(diff_data, 2, axis=1)
            dist_l_inf = np.linalg.norm(diff_data, np.inf, axis=1)

            for img_idx in range(0, num_images):
                float_img = batch_adv[img_idx]
                if np.sum(float_img) == 0:
                    continue
                adv_work_count += 1
                tmp_img_pred = [0, 0, 0]
                tmp_img_pred_topk = [0, 0, 0]
                tmp_img_tran = [0, 0, 0]
                tmp_img_det = [0, 0, 0]

                for idx in range(3):
                    if det_batch_adv_img[img_idx][idx]:
                        img_pred_acc[idx] += 1
                        img_pred_acc_topk[idx] += 1
                        img_adv_det_acc[idx] += 1
                        tmp_img_pred[idx] = 1
                        tmp_img_pred_topk[idx] = 1
                        tmp_img_det[idx] = 1
                    else:
                        tmp_img_pred[idx] = 1 if batch_adv_raw_label[img_idx] == res_img_label[img_idx][idx] else 0
                        tmp_img_pred_topk[idx] = 1 if res_img_topk[img_idx][idx] else 0
                        img_pred_acc[idx] += tmp_img_pred[idx]
                        img_pred_acc_topk[idx] += tmp_img_pred_topk[idx]
                        tmp_img_tran[idx] = int(not ((batch_adv_label[img_idx] == res_img_label[img_idx][idx]) ^ targeted))
                        img_adv_work_acc[idx] += tmp_img_tran[idx]

                if det_batch_adv[img_idx]:
                    total_acc += 1
                    total_acc_topk += 1
                    adv_det_acc += 1
                    tmp_float_pred = 1
                    tmp_float_pred_topk = 1
                    tmp_float_tran = 0
                    tmp_float_det = 1
                else:
                    tmp_float_pred = 1 if batch_adv_raw_label[img_idx] == res_label[img_idx] else 0
                    tmp_float_pred_topk = 1 if res_label_topk[img_idx] else 0
                    total_acc += tmp_float_pred
                    total_acc_topk += tmp_float_pred_topk
                    tmp_float_tran = int(not ((batch_adv_label[img_idx] == res_label[img_idx]) ^ targeted))
                    adv_work_acc += tmp_float_tran
                    tmp_float_det = 0

                cur_img_idx = img_idx
                fp_raw.write(("%d\t%d\t%d\t%.4f\t%.4f\t%.4f" %
                              (model_id, adv_img_idx[i + cur_img_idx], dist_l0[cur_img_idx],
                               dist_l1[cur_img_idx], dist_l2[cur_img_idx], dist_l_inf[cur_img_idx])).encode())
                fp_raw.write(("\t%d\t%d\t%d\t%d" %
                              (tmp_float_pred, tmp_img_pred[0], tmp_img_pred[1], tmp_img_pred[2])).encode())
                fp_raw.write(("\t%d\t%d\t%d\t%d" %
                              (tmp_float_tran, tmp_img_tran[0], tmp_img_tran[1], tmp_img_tran[2])).encode())
                fp_raw.write(("\t%d\t%d\t%d\t%d" %
                              (tmp_float_det, tmp_img_det[0], tmp_img_det[1], tmp_img_det[2])).encode())
                fp_raw.write(("\t%d\t%d\t%d\t%d\n" %
                              (tmp_float_pred_topk, tmp_img_pred_topk[0],
                               tmp_img_pred_topk[1], tmp_img_pred_topk[2])).encode())

        adv_acc = adv_work_acc / adv_work_count
        pred_acc = total_acc / adv_work_count
        pred_acc_topk = total_acc_topk / adv_work_count
        det_acc = adv_det_acc / adv_work_count
        print("acc adv tran: %.4f acc pred: %.4f acc pred top %d: %.4f adv det: %.4f" %
              (adv_acc, pred_acc, top_k, pred_acc_topk, det_acc))
        fp.write(('\t%.4f\t%.4f\t%.4f\t%.4f' % (adv_acc, pred_acc, pred_acc_topk, det_acc)).encode())
        for idx in range(3):
            adv_acc = img_adv_work_acc[idx] / adv_work_count
            pred_acc = img_pred_acc[idx] / adv_work_count
            pred_acc_topk = img_pred_acc_topk[idx] / adv_work_count
            det_acc = img_adv_det_acc[idx] / adv_work_count
            print("acc %s adv tran: %.4f acc %s pred: %.4f acc %s pred top %d: %.4f adv %s det: %.4f" %
                  (img_cate[idx], adv_acc, img_cate[idx], pred_acc,
                   img_cate[idx], top_k, pred_acc_topk, img_cate[idx], det_acc))
            fp.write(('\t%.4f\t%.4f\t%.4f\t%.4f' % (adv_acc, pred_acc, pred_acc_topk, det_acc)).encode())
        fp.write(('\t%d/%d\t%.4f\n' % (adv_work_count, count, adv_work_count/count)).encode())
        model_id += 1

# close loaded file
fp_attack_adv.close()
fp_attack_adv_img.close()
fp_attack_adv_label.close()
fp_attack_adv_raw_label.close()

fp.close()
fp_raw.close()
