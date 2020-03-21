import gzip
import json
import os
import time
import utils
import numpy as np
import tensorflow as tf
import argparse

import worker
from RsNet.setup_mnist import MNIST, MNISTModel, FASHIONModel, CIFAR10Model, SQModels
from RsNet.l0_attack_multi import CarliniL0
from RsNet.l2_attack_multi import CarliniL2
from RsNet.li_attack_multi import CarliniLi
from RsNet.fgm_attack import Fgm
from RsNet.tf_config import gpu_config, setup_visibile_gpus, CHANNELS_LAST
from RsNet.worker import SimpleReformer
from RsNet.dataset_nn import model_mnist_meta, model_cifar10_meta


def parse_rand_spike(_str):
    _str = _str.split(',')
    return [float(x) for x in _str]


def to_img_ceil(img):
    """
    Convert pixel value from float64 to 8 bit with ceiling
    :param img :np.ndarray
    :return: np.ndarray
    """
    return np.clip(np.ceil(img), 0, 255).astype(np.uint8)


def to_img_floor(img):
    """
    Convert pixel value from float64 to 8 bit with floor
    :param img :np.ndarray
    :return: np.ndarray
    """
    return np.clip(np.floor(img), 0, 255).astype(np.uint8)


def to_img_round(img):
    """
    Convert pixel value from float64 to 8 bit with round
    :param img :np.ndarray
    :return: np.ndarray
    """
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def generate_data(data, models_pred, models_idx, samples, targeted=True, start=0, imagenet=False, is_test_data=True, is_rand=False,
                  target_random=False, random_count=1, data_type='test', ref_idx=[], target_exclude_topk=1):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    _inputs = []
    _targets = []
    _raw_targets = []
    _idx = []
    ref_pred = []

    ref_data, ref_label, ref_idx = data.get_data_by_idx(ref_idx, data_type=data_type)
    for model_i in range(len(models_pred)):
        ref_pred.append(models_pred[model_i][data_type])

    tmp_pred = np.argmax(ref_pred, axis=2)
    print(np.shape(ref_pred), np.shape(tmp_pred))
    print(tmp_pred)
    for __pred in tmp_pred:
        cur_acc = __pred[ref_idx] == np.argmax(ref_label, axis=1)
        print("axx_count", np.sum(cur_acc))

    label_count = ref_label.shape[1]
    ref_pred = np.argpartition(ref_pred, label_count - target_exclude_topk, axis=2)
    ref_pred = ref_pred[:, :, -target_exclude_topk:]
    ref_pred = np.transpose(ref_pred, [1, 0, 2])
    ref_pred = np.reshape(ref_pred, [ref_pred.shape[0], -1])

    for i in range(len(ref_idx)):
        if targeted:
            j_ref = np.argmax(ref_label[i])
            j_pred = ref_pred[ref_idx[i]]
            rand_start_idx = 1 if imagenet else 0
            seq_init = np.arange(rand_start_idx, label_count)

            if target_random:
                seq = np.random.choice(seq_init, random_count, replace=False)
                # avoid target label equal to original label or predicted label
                while j_ref in seq or np.any(np.isin(seq, j_pred)):
                    seq = np.random.choice(seq_init, random_count, replace=False)
            else:
                seq_argwhere = np.argwhere(seq_init != j_ref).flatten()
                seq = seq_init[seq_argwhere]

            for j in seq:
                _inputs.append(ref_data[i])
                _targets.append(np.eye(label_count, dtype=np.float32)[j])
                _raw_targets.append(ref_label[i])
                _idx.append(ref_idx[i])
        else:
            _inputs.append(ref_data[i])
            _targets.append(ref_label[i])
            _raw_targets.append(ref_label[i])
            _idx.append(ref_idx[i])

    _inputs = np.array(_inputs)
    _targets = np.array(_targets)
    _raw_targets = np.array(_raw_targets)
    _idx = np.array(_idx)

    return _inputs, _targets, _raw_targets, _idx


def parse_gpus_str(in_str):
    _tmp = in_str.split(',')
    if len(_tmp) == 1:
        return list(range(int(_tmp[0]))) if _tmp[0] != '' else []
    else:
        return [int(x) for x in _tmp if x != '']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate adv example ion multi models')

    parser.add_argument('--data_dir', help='data dir, required', type=str, default=None)
    parser.add_argument('--data_name', help='data name, required', type=str, default=None)
    parser.add_argument('--attack_name', help='attack name, required', type=str, default=None)
    parser.add_argument('--start', help='start index of the test dataset', type=int, default=0)
    parser.add_argument('--num', help='number of test example wanted to attack', type=int, default=0)
    parser.add_argument('--model_dir', help='save model directory, required', type=str, default=None)
    parser.add_argument('--model_name', help='save model name, required', type=str, default=None)
    parser.add_argument('--det_model_dir', help='detector model directory, required', type=str, default='')
    parser.add_argument('--det_model_names', help='detector model names, required', type=str, default='')
    parser.add_argument('--reformer_name', help='reformer name, required', type=str, default='')
    parser.add_argument('--cw_confidence', help='confidence value for C&W attack', type=int, default=0)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
    parser.add_argument('--set_name', help='set name [mnist, fashion, cifar10], required', type=str, default=None)
    parser.add_argument('--dropout', help='dropout rate', type=float, default=0)
    parser.add_argument('--random_spike', help='parameter used for random spiking', type=str, default=None)
    parser.add_argument('--is_targeted', help='whether the attack is targeted attack, yes or no', type=str,
                        default='yes')
    parser.add_argument('--gpu_mem', help='gpu memory fraction', type=float, default=1)
    parser.add_argument('--att_mthd', help='attack method: l2, li, l0, licold, fgm', type=str, default='l2')
    parser.add_argument('--cw_iter', help='iteration, number of c&w attack iteration', type=int, default=1000)
    parser.add_argument('--data_format', help='channels_last or channels_first', type=str, default=CHANNELS_LAST)
    parser.add_argument('--gpu_idx', help='gpu indexs', type=str, default=None)
    parser.add_argument('--dis_temp', help='distillation temperature', type=int, default=1)
    parser.add_argument('--is_test', help='attack on test or training dataset', type=str, default='yes')
    parser.add_argument('--is_data_rand', help='whether to choose random benign from the dataset', type=str,
                        default='no')
    parser.add_argument('--is_target_rand', help='whether to choose the targeted label randomly', type=str,
                        default='no')
    parser.add_argument('--cw_const_factor', help='cw rate at which we increase constant, smaller better', type=float,
                        default=10.0)
    parser.add_argument('--cw_lr', help='cw rate at which we increase constant, smaller better', type=float,
                        default=1e-2)
    parser.add_argument('--cw_init_const', help='cw initial const', type=float, default=1e-3)
    parser.add_argument('--cw_l2_eot_count', help='number of prediction, expectation of transformation, '
                                                  'cw l2 adaptive attack of randomized algorithm', type=int, default=1)
    parser.add_argument('--cw_l2_eot_det_count', help='', type=int, default=1)
    parser.add_argument('--fgm_eps', help='fgm epsilonepsilon', type=float, default=0.3)
    parser.add_argument('--is_det_joint', help='whether use one threshold for all detectors', type=str, default='no')
    parser.add_argument('--eval_lab', help='the raw prediction output', type=str, default='no')
    parser.add_argument('--palette_shade', help='train image with color palette, cubic of [2-6] colors, default: -1, '
                                                'disable this feature', type=int, default=-1)
    parser.add_argument('--boxmin', help='model input image value min', type=float, default=0)
    parser.add_argument('--boxmax', help='model input image value max', type=float, default=1.0)
    parser.add_argument('--model_arch_name', help='model name used for imagenet', type=str, default='inception_v3')
    parser.add_argument('--top_k', help='exclude top_k attack target', type=int, default=1)
    parser.add_argument('--att_label_num', help='attack label count', type=int, default=1)
    parser.add_argument('--imagenet_image_size', help='imagenet image size', type=int, default=224)
    parser.add_argument('--preprocess_name', help='preprocess function name', type=str, default=None)
    parser.add_argument('--intra_op_parallelism_threads',
                        help="""Nodes that can use multiple threads to 
                        parallelize their execution will schedule the 
                        individual pieces into this pool.
                        Default value 1 avoid pool of Eiden threads""",
                        type=int, default=1)
    parser.add_argument('--inter_op_parallelism_threads', help="""All ready nodes are scheduled in this pool.""",
                        type=int, default=5)
    parser.add_argument('--num_parallel_calls', help="The level of parallelism for data "
                                                     "preprocessing across multiple CPU cores",
                        type=int, default=5)
    parser.add_argument('--fp16', help="""Train using float16 (half) precision instead of float32.""",
                        type=str, default='yes')

    args = parser.parse_args()

    _dir = args.data_dir
    name = args.data_name
    attack_name = args.attack_name
    start = args.start
    sample_size = args.num
    target_model_dir = args.model_dir
    target_model_names = args.model_name.split(",")

    detector_model_dir = args.det_model_dir
    detector_model_names = args.det_model_names.split(",")
    reformer_names = args.reformer_name.split(',')

    out_dir = _dir + "/" + name + "/"
    confidence = args.cw_confidence
    batch_size = args.batch_size
    set_name = args.set_name
    json_str = ""

    dropout = args.dropout
    para_random_spike = args.random_spike
    targeted = args.is_targeted == 'yes'
    attack_method = args.att_mthd
    iteration = args.cw_iter
    data_format = args.data_format
    gpu_idx = parse_gpus_str(args.gpu_idx)
    temp = args.dis_temp
    is_test_data = args.is_test == 'yes'
    is_rand = args.is_data_rand == 'yes'
    target_rand = args.is_target_rand == 'yes'
    const_factor = args.cw_const_factor
    initial_const = args.cw_init_const
    is_det_joint = args.is_det_joint == 'yes'
    eval_lab = args.eval_lab.split(",")

    palette_shade = args.palette_shade
    boxmin = args.boxmin
    boxmax = args.boxmax
    top_k = args.top_k
    model_arch_name = args.model_arch_name.split(',')
    att_label_num = args.att_label_num
    image_size = args.imagenet_image_size
    dtype = tf.float16 if args.fp16 == 'yes' else tf.float32
    gpu_config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem

    fgm_eps = args.fgm_eps

    setup_visibile_gpus(",".join(map(str, gpu_idx)))
    selected_gpus = list(range(len(gpu_idx)))
    gpu_count = len(gpu_idx)

    if palette_shade == -1:
        palette_shade = None
    elif palette_shade > 6 or palette_shade < 2:
        print("Error: invalid palette shade value", palette_shade, ". Possible value [-1, 2-6]")
        exit(0)

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

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gpu_thread_count = 2
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    with tf.Session(config=gpu_config) as sess:
        models = []
        models_idx = []
        models_pred = []
        gpu_i = 0
        for _model_name in target_model_names:
            with tf.device('/gpu:' + str(gpu_i % gpu_count)):
                _path = os.path.join(target_model_dir, _model_name)
                model_arch = model_arch_name[gpu_i] if len(model_arch_name) > gpu_i else None
                models.append(MODEL(_path, sess,
                                    input_data_format=data_format, data_format=data_format,
                                    dropout=dropout, rand_params=para_random_spike, is_batch=True,
                                    palette_shade=palette_shade, model_name=model_arch, image_size=image_size,
                                    dtype=dtype))
                models_idx.append(utils.load_model_idx(_path))
                gpu_i += 1

        for _eval_lab in eval_lab:
            models_pred.append(utils.load_obj(_eval_lab, directory=''))

        data = MNIST(_dir, name, model_meta=model_meta, validation_size=0,
                     input_data_format=CHANNELS_LAST, output_data_format=data_format, batch_size=batch_size,
                     boxmin=boxmin, boxmax=boxmax)

        num_labels = model_meta.labels
        if 0 <= num_labels <= 255:
            label_data_type = np.uint8
        else:
            label_data_type = np.uint16

        # construct detector and dropout_rate
        detector_dict, thrs, detector_gpu_idx = \
            worker.build_detector(detector_model_dir, detector_model_names, target_model_names, target_model_dir,
                                 "", MODEL, models, data, data_format, is_det_joint, models_idx, gpu_count)

        # concat reformer in front of models
        reformer_id = 0
        for cur_reformer_name in reformer_names:
            if cur_reformer_name == '':
                reformer_id += 1
                continue
            with tf.device('/gpu:' + str(reformer_id % gpu_count)):
                cur_reformer = SimpleReformer(os.path.join(detector_model_dir, cur_reformer_name))
                cur_model = models[reformer_id]

                models[reformer_id] = SQModels(model_list=[cur_reformer.model, cur_model.model], ref_model=cur_model)

            reformer_id += 1

        # prediction result evaluation
        if data_format == CHANNELS_LAST:
            shape = (None, model_meta.width, model_meta.height, model_meta.channel)
        else:
            shape = (None, model_meta.channel, model_meta.width, model_meta.height)

        img_eval = tf.placeholder(tf.float32, shape)
        pred_res = []

        gpu_i = 0
        for _model in models:
            with tf.device('/gpu:' + str(gpu_i % gpu_count)):
                _pred_res = _model.predict(img_eval)
                pred_res.append(_pred_res)

        time.sleep(np.random.rand() * confidence % 60)

        if os.path.isfile(out_dir + "/config.json"):
            config_fp = open(out_dir + "/config.json", "rb")
            json_str = config_fp.read()
            config_fp.close()

        if json_str == "":
            config = {
                "name": name,
                attack_name + "-img": attack_name + "-img.gz",
                attack_name + "-label": attack_name + "-label.gz",
                attack_name + "-adv-img": attack_name + "-adv-img.gz",
                attack_name + "-adv-label": attack_name + "-adv-label.gz",
                attack_name + "-adv-raw-label": attack_name + "-adv-raw-label.gz",
            }
        else:
            config = json.loads(json_str.decode())
            config[attack_name + "-img"] = attack_name + "-img.gz"
            config[attack_name + "-label"] = attack_name + "-label.gz"
            config[attack_name + "-adv-img"] = attack_name + "-adv-img.gz"
            config[attack_name + "-adv-label"] = attack_name + "-adv-label.gz"
            config[attack_name + "-adv-raw-label"] = attack_name + "-adv-raw-label.gz"

        # build attack model
        if attack_method == 'l0':
            attack = CarliniL0(sess=sess, models=models, detector_dict=detector_dict, thrs=thrs,
                               batch_size=batch_size, max_iterations=iteration,
                               confidence=confidence, learning_rate=1e-2, targeted=targeted, boxmin=boxmin, boxmax=boxmax,
                               const_factor=const_factor, data_format=data_format, initial_const=initial_const)
        elif attack_method == 'l2':
            attack = CarliniL2(sess=sess, models=models, detector_dict=detector_dict,
                               detector_gpu_idx=detector_gpu_idx, thrs=thrs,
                               batch_size=batch_size, max_iterations=iteration,
                               confidence=confidence, learning_rate=args.cw_lr, targeted=targeted, boxmin=boxmin, boxmax=boxmax,
                               const_factor=const_factor, data_format=data_format, initial_const=initial_const,
                               temp=temp, gpu_count=gpu_count, eot_count=args.cw_l2_eot_count,
                               eot_det_count=args.cw_l2_eot_det_count)
        elif attack_method == 'li':
            attack = CarliniLi(sess=sess, models=models, detector_dict=detector_dict, thrs=thrs,
                               batch_size=batch_size, max_iterations=iteration,
                               confidence=confidence, learning_rate=5e-3, targeted=targeted, boxmin=boxmin, boxmax=boxmax,
                               const_factor=const_factor, decrease_factor=0.6, warm_start=True,
                               data_format=data_format, initial_const=initial_const)
        elif attack_method == 'licold':
            attack = CarliniLi(sess=sess, models=models, detector_dict=detector_dict, thrs=thrs,
                               batch_size=batch_size, max_iterations=iteration,
                               confidence=confidence, learning_rate=5e-3, targeted=targeted, boxmin=boxmin, boxmax=boxmax,
                               const_factor=const_factor, decrease_factor=0.6, warm_start=False,
                               data_format=data_format, initial_const=initial_const)
        elif attack_method == 'fgm':
            attack = Fgm(model=models[0], has_labels=True, eps=fgm_eps, ord=np.inf, clip_min=boxmin, clip_max=boxmax,
                         targeted=targeted, model_meta=model_meta, data_format=data_format)


        # adversarial file
        attack_stream = gzip.open(out_dir + "/" + config[attack_name + "-img"], "wb")
        attack_label = gzip.open(out_dir + "/" + config[attack_name + "-label"], "wb")
        attack_adv = gzip.open(out_dir + "/" + config[attack_name + "-adv-img"], "wb")
        attack_adv_label = gzip.open(out_dir + "/" + config[attack_name + "-adv-label"], "wb")
        attack_adv_raw_label = gzip.open(out_dir + "/" + config[attack_name + "-adv-raw-label"], "wb")
        # fill 16 and 8 zero bytes at the beginning of the file
        attack_stream.write(np.zeros([16], np.uint8).tobytes())
        attack_label.write(np.zeros([8], np.uint8).tobytes())

        # start attack
        timestart = time.time()

        if is_test_data:
            data_type = 'test'
        else:
            data_type = 'train'
        ref_idx = models_idx[0][data_type]
        if is_rand:
            np.random.shuffle(ref_idx)

        ref_idx = ref_idx[start:start + sample_size]
        ref_idx.sort()
        # generate targeted attack data
        if set_name != 'imagenet':
            inputs, targets, raw_targets, orig_idx = generate_data(
                data, models_pred, models_idx, samples=sample_size,
                targeted=targeted,
                start=start, imagenet=False, is_test_data=is_test_data,
                is_rand=is_rand, target_random=target_rand,
                random_count=att_label_num, ref_idx=ref_idx,
                target_exclude_topk=args.top_k)
            print(inputs.dtype, targets.dtype, raw_targets.dtype)
            # save orig idx
            utils.save_obj(orig_idx, name=attack_name + "-idx", directory=out_dir)
        else:
            orig_idx = []

        # save the attack image and miss-classified label into the file
        input_len = len(inputs) if set_name != 'imagenet' else data.data_len()
        config[attack_name + "-count"] = input_len * 3
        config_fp = open(out_dir + "/config.json", "wb")
        config_str = json.dumps(config)
        config_fp.write(config_str.encode())
        config_fp.close()

        for i in range(0, input_len, batch_size):
            i_end = i + batch_size if i + batch_size <= input_len else input_len
            if set_name == 'imagenet':
                batch_inputs, batch_raw_targets, batch_targets, batch_orig_idx = data.next_batch()
                print(np.argmax(batch_raw_targets, axis=1))
                print(np.argmax(batch_targets, axis=1))
                print(batch_orig_idx)
                if not targeted:
                    batch_targets = batch_raw_targets
                orig_idx.extend(batch_orig_idx)
            else:
                batch_inputs = inputs[i:i_end]
                batch_targets = targets[i:i_end]
                batch_raw_targets = raw_targets[i:i_end]
            batch_adv = attack.attack(batch_inputs, batch_targets,
                                      input_data_format=data_format, output_data_format=data_format)
            cur_batch_size = i_end - i

            if data_format == CHANNELS_LAST:
                output_batch_adv = batch_adv
            else:
                # convert data format from channels_first to channels_last
                output_batch_adv = batch_adv.transpose([0, 2, 3, 1])
            attack_adv.write(output_batch_adv.astype(np.float32).tobytes())
            attack_adv_label.write(np.argmax(batch_targets, 1).astype(label_data_type).tobytes())
            attack_adv_raw_label.write(np.argmax(batch_raw_targets, 1).astype(label_data_type).tobytes())
            tt = np.argmax(batch_targets, 1)

            adv_raw = (batch_adv - boxmin) / (boxmax-boxmin) * 255
            adv_floor = to_img_floor(adv_raw)
            adv_ceil = to_img_ceil(adv_raw)
            adv_round = to_img_round(adv_raw)

            adv_floor_float = (adv_floor / 255) * (boxmax-boxmin) + boxmin
            adv_ceil_float = (adv_ceil / 255) * (boxmax-boxmin) + boxmin
            adv_round_float = (adv_round / 255) * (boxmax-boxmin) + boxmin
            classify_floor = sess.run(pred_res, feed_dict={img_eval: adv_floor_float})
            classify_ceil = sess.run(pred_res, feed_dict={img_eval: adv_ceil_float})
            classify_round = sess.run(pred_res, feed_dict={img_eval: adv_round_float})

            def find_class(res):
                res = [np.argmax(_res, axis=1) for _res in res]
                res = [(np.arange(model_meta.labels) == _res[:, None]).astype(np.float32) for _res in res]
                res = np.sum(res, axis=0)
                return np.argmax(res, axis=1)


            label_floor = find_class(classify_floor)
            label_ceil = find_class(classify_ceil)
            label_round = find_class(classify_round)

            if data_format == CHANNELS_LAST:
                pass
            else:
                # convert data format from channels_first to channels_last
                adv_floor = adv_floor.transpose([0, 2, 3, 1])
                adv_ceil = adv_ceil.transpose([0, 2, 3, 1])
                adv_round = adv_round.transpose([0, 2, 3, 1])

            adv_img = np.transpose([adv_floor, adv_ceil, adv_round], [1, 0, 2, 3, 4])
            pred_labels = np.transpose([label_floor, label_ceil, label_round], [1, 0]).astype(dtype=label_data_type)
            attack_stream.write(adv_img.tobytes())
            attack_label.write(pred_labels.tobytes())

            print("finish %d/%d" % (i_end, input_len))

        # end of attack
        timeend = time.time()

        print("Took", timeend - timestart, "seconds to run", input_len, "samples.")

        if set_name == 'imagenet':
            orig_idx = np.squeeze(orig_idx)
            utils.save_obj(orig_idx, name=attack_name + "-idx", directory=out_dir)

        attack_stream.close()
        attack_label.close()
        attack_adv.close()
        attack_adv_label.close()
        attack_adv_raw_label.close()
