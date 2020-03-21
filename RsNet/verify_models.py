import worker
from RsNet.setup_mnist import MNIST, MNISTModel, FASHIONModel, CIFAR10Model, SQModels
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta
import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from RsNet.tf_config import gpu_config, setup_visibile_gpus, CHANNELS_LAST
from RsNet.worker import SimpleReformer
from utils import softmax_cross_entropy_with_logits
import numpy as np
import argparse
import utils


def parse_rand_spike(_str):
    _str = _str.split(',')
    return [float(x) for x in _str]


parser = argparse.ArgumentParser(description='Verify model accuracy')

parser.add_argument('--data_dir', help='data dir, required', type=str, default=None)
parser.add_argument('--data_name', help='data name, required', type=str, default=None)
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
parser.add_argument('--is_detail', help='whether to output detail data', type=str, default='no')
parser.add_argument('--is_det_joint', help='whether use one threshold for all detectors', type=str, default='no')
parser.add_argument('--is_logits', help='whether tp apply bagging with logits or softmax', type=str, default='yes')
parser.add_argument('--boxmin', help='model input image value min', type=float, default=0)
parser.add_argument('--boxmax', help='model input image value max', type=float, default=1.0)
parser.add_argument('--top_k', help='evaluate extra top_k task', type=int, default=1)
parser.add_argument('--model_arch_name', help='model name used for imagenet', type=str, default='inception_v3')
parser.add_argument('--image_size', help='image dimension: default 224', type=int, default=224)
parser.add_argument('--preprocess_name', help='preprocess function name', type=str, default=None)
parser.add_argument('--num_parallel_calls', help="The level of parallelism for data "
                                                 "preprocessing across multiple CPU cores",
                    type=int, default=5)
parser.add_argument('--fp16', help="""Train using float16 (half) precision instead of float32.""",
                    type=str, default='yes')
parser.add_argument('--defend_para', help='the parameter of selected defensive method', type=str, default=None)
parser.add_argument('--defend_mthd', help='defensive method name', type=str, default=None)

args = parser.parse_args()

data_dir = args.data_dir
data_name = args.data_name
save_model_dir = args.model_dir
save_model_name = args.model_name
detector_model_dir = args.det_model_dir
detector_model_names = args.det_model_names.split(",")
reformer_name = args.reformer_name
set_name = args.set_name
batch_size = args.batch_size
data_format = args.data_format
gpu_idx = args.gpu_idx
output_file = args.output_file
iteration = args.iteration
dropout = args.dropout
bagging = args.bagging == 'yes'
para_random_spike = args.random_spike
detail = args.is_detail == 'yes'
is_det_joint = args.is_det_joint == 'yes'
is_logits = args.is_logits == 'yes'
boxmin = args.boxmin
boxmax = args.boxmax
top_k = args.top_k
model_arch_name = args.model_arch_name
image_size = args.image_size
dtype = tf.float16 if args.fp16 == 'yes' else tf.float32

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

if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

fp = open(output_file, 'wb')
fp.write("name\ttest_acc\ttest_acc_top_k\ttest_loss\ttrain_acc\ttrain_acc_top_k\ttrain_loss\n".encode())
fp.write(save_model_name.encode())


with tf.Session(config=gpu_config) as sess:
    cur_path = os.path.join(save_model_dir, save_model_name)
    data = MNIST(data_dir, data_name, 0, model_meta=model_meta,
                 input_data_format=CHANNELS_LAST, output_data_format=data_format, batch_size=batch_size)

    sess.run(tf.local_variables_initializer())
    # if encrypt and iteration > 1:
    #     data.encrypt_tf(sess, os.path.join(save_model_dir, save_model_name), batch_size=batch_size)
    #     encrypt = False
    model = MODEL(cur_path, sess, output_logits=True,
                  input_data_format=data_format, data_format=data_format, dropout=dropout,
                  rand_params=para_random_spike, is_batch=True,
                  model_name=model_arch_name, cmin=boxmin, cmax=boxmax, image_size=image_size, dtype=dtype)
    det_model = MODEL(cur_path, sess, output_logits=True,
                      input_data_format=data_format, data_format=data_format, is_batch=True,
                      model_name=model_arch_name, cmin=boxmin, cmax=boxmax, image_size=image_size, dtype=dtype)

    cur_model_idx = utils.load_model_idx(cur_path)
    if cur_model_idx is None:
        cur_model_idx = utils.save_model_idx(cur_path, data)

    cur_det_set, cur_thrs_set, cur_det_gpu_idx = \
        worker.build_detector(detector_model_dir, detector_model_names, save_model_name, save_model_dir, cur_path,
                              MODEL, det_model, data, data_format, is_det_joint, cur_model_idx)

    if reformer_name != '':
        cur_reformer = SimpleReformer(os.path.join(detector_model_dir, reformer_name))
        print("concat reformer", reformer_name, "before model", save_model_name)
        cur_model = SQModels(model_list=[cur_reformer.model, model.model], ref_model=model)

    k_model = model.model
    sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)

    y_correct = tf.placeholder(tf.float32, [None, model_meta.labels])
    y_targets = tf.placeholder(tf.int32, [None])
    y_predict = tf.placeholder(tf.float32, [None, model_meta.labels])
    label_val = np.arange(model_meta.labels)

    loss = softmax_cross_entropy_with_logits(y_correct, y_predict)
    out_softmax = tf.nn.softmax(y_predict)
    out_top_k = tf.nn.in_top_k(y_predict, y_targets, top_k)

    def evaluate(data, data_type='test', fp_detail=None, dname=''):
        data_len = data.data_len(data_type=data_type)
        total_pred = []
        total_loss = 0
        total_output = []
        total_output_top_k = []
        avg_loss = 0
        avg_acc = 0

        for i in range(0, data_len, batch_size):
            end_idx = i + batch_size
            if end_idx > data_len:
                end_idx = data_len

            cur_batch_size = end_idx - i
            cur_data, cur_label = data.next_batch(data_type=data_type)
            cur_loss = 0
            cur_pred = None

            # detector checking:
            det_batch_adv = np.array([False] * cur_batch_size)

            for det_name, det in cur_det_set.items():
                cur_thrs = cur_thrs_set[det_name]
                cur_det_batch_adv = det.mark(cur_data, data_format=data_format) >= cur_thrs
                det_batch_adv |= cur_det_batch_adv

            for j in range(iteration):
                predict_label = k_model.predict_on_batch(x=cur_data) + 1
                #print(np.argmax(cur_label[0]), np.argmax(predict_label[0]), '\n')
                computed_loss = loss.eval(session=sess, feed_dict={y_correct: cur_label, y_predict: predict_label})
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

            bak_pred = cur_pred
            cur_targets = np.argmax(cur_label, 1)
            cur_pred = np.equal(np.argmax(cur_pred, 1), cur_targets)
            cur_top_k = out_top_k.eval(session=sess, feed_dict={y_targets: cur_targets, y_predict: bak_pred})
            total_output.extend(bak_pred)

            for img_idx in range(0, cur_batch_size):
                if det_batch_adv[img_idx]:
                    cur_pred[img_idx] = False
                    cur_top_k[img_idx] = False

            if fp_detail is not None:
                orig_lb = np.argmax(cur_label, 1)
                pred_lb = np.argmax(bak_pred, 1)
                for img_idx in range(0, cur_batch_size):
                    if not cur_pred[img_idx]:
                        fp_detail.write(("%d\t%s\t%d\t%d\t%d\t%d\n" % (
                            i+img_idx, dname, orig_lb[img_idx], pred_lb[img_idx],
                            det_batch_adv[img_idx], cur_top_k[img_idx])).encode())

            total_loss += cur_loss
            total_pred.extend(cur_pred)
            total_output_top_k.extend(cur_top_k)

            avg_acc = np.mean(total_pred)
            avg_loss = total_loss / end_idx / iteration
            avg_acc_top_k = np.mean(total_output_top_k)
            cur_acc = np.mean(cur_pred)
            cur_loss = cur_loss / cur_batch_size / iteration
            cur_acc_top_k = np.mean(cur_top_k)

            print("%d/%d cur_loss: %.4f - cur_acc: %.4f cur_acc_top_k %.4f - "
                  "avg_loss: %.4f - avg_acc: %.4f - avg_acc_top_k: %.4f" %
                  (end_idx, data_len, cur_loss, cur_acc, cur_acc_top_k, avg_loss, avg_acc, avg_acc_top_k), end="\r")

        return avg_loss, avg_acc, avg_acc_top_k, total_output

    timestart = time.time()

    if detail:
        if output_file.endswith(".csv"):
            output_file = output_file[:-4]
        fp_detail_test = open(output_file + '-test.csv', 'wb')
        fp_detail_test.write("idx\tdataset\torig\tpred\tdet\ttop_k\n".encode())
        fp_detail_train = open(output_file + '-train.csv', 'wb')
        fp_detail_train.write("idx\tdataset\torig\tpred\tdet\ttop_k\n".encode())
    else:
        fp_detail_test = None
        fp_detail_train = None

    data.apply_pre_idx(cur_model_idx)
    print("* test data *")
    re_loss, re_acc, re_acc_top_k, test_out = \
        evaluate(data=data, data_type='test', fp_detail=fp_detail_test, dname='test')
    print("avg_loss: %.4f - avg_acc: %.4f - avg_acc_top_k: %.4f" % (re_loss, re_acc, re_acc_top_k))
    fp.write(('\t%.4f\t%.4f\t%.4f' % (re_acc, re_acc_top_k, re_loss)).encode())

    if set_name != 'imagenet':
        print("* training data *")
        re_loss, re_acc, re_acc_top_k, train_out = \
            evaluate(data=data, data_type='train', fp_detail=fp_detail_train, dname='train')
        print("avg_loss: %.4f - avg_acc: %.4f - avg_acc_top_k: %.4f" % (re_loss, re_acc, re_acc_top_k))
        fp.write(('\t%.4f\t%.4f\t%.4f\n' % (re_acc, re_acc_top_k, re_loss)).encode())
    else:
        train_out = test_out
        fp.write(('\t%.4f\t%.4f\t%.4f\n' % (re_acc, re_acc_top_k, re_loss)).encode())

    # save prediction raw output
    raw_out = {
        "train": np.array(train_out),
        "test": np.array(test_out)
    }

    if output_file.endswith('.csv'):
        output_file = output_file[:-4] + '_raw'
    else:
        output_file += '_raw'

    utils.save_obj(raw_out, output_file, directory='')

    if detail:
        fp_detail_test.close()
        fp_detail_train.close()

    timeend = time.time()
    print("Took", timeend - timestart, "seconds to run")

    fp.close()
