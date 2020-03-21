## worker.py -- evaluation code
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import matplotlib
from scipy.stats import entropy
from numpy.linalg import norm
from matplotlib.ticker import FuncFormatter
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.activations import softmax
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from RsNet.tf_config import CHANNELS_LAST
from utils import load_obj, load_model_idx, load_cache, save_cache

matplotlib.use('Agg')


class AEDetector:
    def __init__(self, path, p=1, verbose=1):
        """
        Error based detector.
        Marks examples for filtering decisions.

        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model = load_model(path)
        if verbose:
            self.model.summary()
        self.path = path
        self.p = p

    def mark(self, X, data_format=CHANNELS_LAST):
        if self.model.inputs[0].shape[1:] != np.shape(X)[1:]:
            if data_format == CHANNELS_LAST:
                X = np.transpose(X, [0, 3, 1, 2])
            else:
                X = np.transpose(X, [0, 2, 3, 1])
        diff = np.abs(X - self.model.predict(X))
        marks = np.mean(np.power(diff, self.p), axis=(1, 2, 3))
        return marks

    def tf_mark(self, X, data_format=CHANNELS_LAST):
        if self.model.inputs[0].shape[1:] != np.shape(X)[1:]:
            if data_format == CHANNELS_LAST:
                X = tf.transpose(X, [0, 3, 1, 2])
            else:
                X = tf.transpose(X, [0, 2, 3, 1])
        diff = tf.abs(X - self.model(X))
        marks = tf.reduce_mean(tf.pow(diff, self.p), axis=(1, 2, 3))
        return marks

    def layer(self, X, name, data_format=CHANNELS_LAST):
        def _layer(_x, model, p):
            if self.model.inputs[0].shape[1:] != np.shape(_x)[1:]:
                if data_format == CHANNELS_LAST:
                    _x = tf.transpose(_x, [0, 3, 1, 2])
                else:
                    _x = tf.transpose(_x, [0, 2, 3, 1])
            diff = tf.abs(_x - model(_x))
            marks = tf.reduce_mean(tf.pow(diff, p), axis=(1, 2, 3))
            return marks
        return Lambda(lambda x: _layer(x, self.model, self.p), name=name)(X)

    def print(self):
        return "AEDetector:" + self.path.split("/")[-1]


class IdReformer:
    def __init__(self, path="IdentityFunction"):
        """
        Identity reformer.
        Reforms an example to itself.
        """
        self.path = path
        self.heal = lambda X: X
        self.heal_tf = lambda X: X

    def print(self):
        return "IdReformer:" + self.path


class SimpleReformer:
    def __init__(self, path, verbose=1):
        """
        Reformer.
        Reforms examples with autoencoder. Action of reforming is called heal.

        path: Path to the autoencoder used.
        """
        self.model = load_model(path)
        if verbose:
            self.model.summary()
        self.path = path

    def heal(self, X):
        X = self.model.predict(X)
        return np.clip(X, 0.0, 1.0)

    def heal_tf(self, X):
        X = self.model(X)
        return tf.clip_by_value(X, 0.0, 1.0)

    def print(self):
        return "SimpleReformer:" + self.path.split("/")[-1]


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def JSD_tf(P, Q):
    _P = P / tf.expand_dims(tf.norm(P, ord=1, axis=1), axis=1)
    _Q = Q / tf.expand_dims(tf.norm(Q, ord=1, axis=1), axis=1)
    _M = 0.5 * (_P + _Q)

    def kl(p, q):
        return tf.reduce_sum(p * tf.log(p / q), axis=1)

    return 0.5 * (kl(_P, _M) + kl(_Q, _M))


class DBDetector:
    def __init__(self, reconstructor, prober, classifier, option="jsd", T=1):
        """
        Divergence-Based Detector.

        reconstructor: One autoencoder.
        prober: Another autoencoder.
        classifier: Classifier object.
        option: Measure of distance, jsd as default.
        T: Temperature to soften the classification decision.
        """
        self.prober = prober
        self.reconstructor = reconstructor
        self.classifier = classifier
        self.option = option
        self.T = T

    def mark(self, X, data_format):
        return self.mark_jsd(X)

    def mark_jsd(self, X):
        Xp = self.prober.heal(X)
        Xr = self.reconstructor.heal(X)
        Pp = self.classifier.classify(Xp, option="prob", T=self.T)
        Pr = self.classifier.classify(Xr, option="prob", T=self.T)

        marks = [(JSD(Pp[i], Pr[i])) for i in range(len(Pr))]
        return np.array(marks)

    def tf_mark(self, X, data_format):
        Xp = self.prober.heal_tf(X)
        Xr = self.reconstructor.heal_tf(X)
        Pp = self.classifier.classify_tf(Xp, option="prob", T=self.T)
        Pr = self.classifier.classify_tf(Xr, option="prob", T=self.T)

        marks = JSD_tf(Pp, Pr)
        return marks

    def print(self):
        return "Divergence-Based Detector"


class Classifier:
    def __init__(self, classifier_path, model_class, data_format, model=None):
        """
        Keras classifier wrapper.
        Note that the wrapped classifier should spit logits as output.

        classifier_path: Path to Keras classifier file.
        """
        self.path = classifier_path
        self.model = model_class(classifier_path, output_logits=True,
                                 input_data_format=data_format, data_format=data_format).model \
            if model is None else model.model
        self.softmax = Sequential()
        self.softmax.add(Lambda(lambda X: softmax(X, axis=1), input_shape=(10,)))

    def classify(self, X, option="logit", T=1):
        if option == "logit":
            return self.model.predict(X)
        if option == "prob":
            logits = self.model.predict(X)/T
            return self.softmax.predict(logits)

    def classify_tf(self, X, option="logit", T=1):
        if option == "logit":
            return self.model(X)
        if option == "prob":
            logits = self.model(X) / T
            return self.softmax(logits)

    def print(self):
        return "Classifier:"+self.path.split("/")[-1]


class Operator:
    def __init__(self, data, classifier, det_dict, reformer, data_format):
        """
        Operator.
        Describes the classification problem and defense.

        data: Standard problem dataset. Including train, test, and validation.
        classifier: Target classifier.
        reformer: Reformer of defense.
        det_dict: Detector(s) of defense.
        """
        self.data = data
        self.classifier = classifier
        self.det_dict = det_dict
        self.reformer = reformer
        self.data_format = data_format
        self.normal = self.operate(AttackData(self.data.test_data,
                                              np.argmax(self.data.test_labels, axis=1), "Normal",
                                              input_data_format=data_format, data_format=data_format))

    def get_thrs(self, drop_rate):
        """
        Get filtering threshold by marking validation set.
        """
        thrs = dict()
        for name, detector in self.det_dict.items():
            num = int(len(self.data.validation_data) * drop_rate[name])
            marks = detector.mark(self.data.validation_data, self.data_format)
            marks = np.sort(marks)
            thrs[name] = marks[-num]
        return thrs

    def operate(self, untrusted_obj):
        """
        For untrusted input(normal or adversarial), classify original input and
        reformed input. Classifier is unaware of the source of input.

        untrusted_obj: Input data.
        """
        X = untrusted_obj.data
        Y_true = untrusted_obj.labels

        X_prime = self.reformer.heal(X)
        Y = np.argmax(self.classifier.classify(X), axis=1)
        Y_judgement = (Y == Y_true[:len(X_prime)])
        Y_prime = np.argmax(self.classifier.classify(X_prime), axis=1)
        Y_prime_judgement = (Y_prime == Y_true[:len(X_prime)])

        return np.array(list(zip(Y_judgement, Y_prime_judgement)))

    def filter(self, X, thrs):
        """
        untrusted_obj: Untrusted input to test against.
        thrs: Thresholds.

        return:
        all_pass: Index of examples that passed all detectors.
        collector: Number of examples that escaped each detector.
        """
        collector = dict()
        all_pass = np.array(range(10000))
        for name, detector in self.det_dict.items():
            marks = detector.mark(X, self.data_format)
            idx_pass = np.argwhere(marks < thrs[name])
            collector[name] = len(idx_pass)
            all_pass = np.intersect1d(all_pass, idx_pass)
        return all_pass, collector

    def print(self):
        components = [self.reformer, self.classifier]
        return " ".join(map(lambda obj: getattr(obj, "print")(), components))


class AttackData:
    def __init__(self, examples, labels, name="", directory='./attack_data/',
                 input_data_format=CHANNELS_LAST, data_format=CHANNELS_LAST):
        """
        Input data wrapper. May be normal or adversarial.

        examples: Path or object of input examples.
        labels: Ground truth labels.
        """
        if isinstance(examples, str):
            self.data = load_obj(examples, directory=directory)
        else:
            self.data = examples

        if input_data_format != data_format:
            if data_format == CHANNELS_LAST:
                self.data = np.transpose(self.data, [0, 2, 3, 1])
            else:
                self.data = np.transpose(self.data, [0, 3, 1, 2])
        self.labels = labels
        self.name = name

    def print(self):
        return "Attack:"+self.name


class Evaluator:
    def __init__(self, operator, untrusted_data, graph_dir="./graph", data_format=CHANNELS_LAST):
        """
        Evaluator.
        For strategy described by operator, conducts tests on untrusted input.
        Mainly stats and plotting code. Most methods omitted for clarity.

        operator: Operator object.
        untrusted_data: Input data to test against.
        graph_dir: Where to spit the graphs.
        """
        self.operator = operator
        self.untrusted_data = untrusted_data
        self.graph_dir = graph_dir
        self.data_format = data_format
        self.data_package = operator.operate(untrusted_data)

    def bind_operator(self, operator):
        self.operator = operator
        self.data_package = operator.operate(self.untrusted_data)

    def load_data(self, data):
        self.untrusted_data = data
        self.data_package = self.operator.operate(self.untrusted_data)

    def get_normal_acc(self, normal_all_pass):
        """
        Break down of who does what in defense. Accuracy of defense on normal
        input.

        both: Both detectors and reformer take effect
        det_only: detector(s) take effect
        ref_only: Only reformer takes effect
        none: Attack effect with no defense
        """
        normal_tups = self.operator.normal
        num_normal = len(normal_tups)
        filtered_normal_tups = normal_tups[normal_all_pass]

        both_acc = sum(1 for _, XpC in filtered_normal_tups if XpC)/num_normal
        det_only_acc = sum(1 for XC, XpC in filtered_normal_tups if XC)/num_normal
        ref_only_acc = sum([1 for _, XpC in normal_tups if XpC])/num_normal
        none_acc = sum([1 for XC, _ in normal_tups if XC])/num_normal

        return both_acc, det_only_acc, ref_only_acc, none_acc

    def get_attack_acc(self, attack_pass):
        attack_tups = self.data_package
        num_untrusted = len(attack_tups)
        filtered_attack_tups = attack_tups[attack_pass]

        both_acc = 1 - sum(1 for _, XpC in filtered_attack_tups if not XpC)/num_untrusted
        det_only_acc = 1 - sum(1 for XC, XpC in filtered_attack_tups if not XC)/num_untrusted
        ref_only_acc = sum([1 for _, XpC in attack_tups if XpC])/num_untrusted
        none_acc = sum([1 for XC, _ in attack_tups if XC])/num_untrusted
        return both_acc, det_only_acc, ref_only_acc, none_acc

    def plot_various_confidences(self, graph_name, drop_rate, data_format,
                                 Y, directory='./attack_data/',
                                 confs=(0.0, 10.0, 20.0, 30.0, 40.0),
                                 get_attack_data_name=lambda c: "example_carlini_"+str(c)):
        """
        Test defense performance against Carlini L2 attack of various confidences.

        graph_name: Name of graph file.
        drop_rate: How many normal examples should each detector drops?
        idx_file: Index of adversarial examples in standard test set.
        confs: A series of confidence to test against.
        get_attack_data_name: Function mapping confidence to corresponding file.
        """
        import matplotlib.pyplot as plt
        import pylab

        pylab.rcParams['figure.figsize'] = 6, 4
        fig = plt.figure(1, (6, 4))
        ax = fig.add_subplot(1, 1, 1)

        det_only = []
        ref_only = []
        both = []
        none = []

        print("Drop Rate:", drop_rate)
        thrs = self.operator.get_thrs(drop_rate)

        all_pass, _detector = self.operator.filter(self.operator.data.test_data, thrs)
        all_on_acc, _, _, _ = self.get_normal_acc(all_pass)

        print(_detector)
        print("Classification accuracy with all defense on:", all_on_acc)

        for confidence in confs:
            f = get_attack_data_name(confidence)
            attack_data = AttackData(f, Y, "Carlini L2 " + str(confidence), directory=directory,
                                     input_data_format=CHANNELS_LAST, data_format=data_format)
            # compute number of all input data and filter out valid data
            total = len(attack_data.data)
            valid_adv_idx = np.argwhere(np.sum(attack_data.data, axis=(1, 2, 3)) > [0] * total).flatten()
            attack_data.data = attack_data.data[valid_adv_idx]
            attack_data.labels = attack_data.labels[valid_adv_idx]
            self.load_data(attack_data)

            print("Confidence:", confidence)
            valid_adv_len = len(valid_adv_idx)
            print("valid attack %d/%d" % (valid_adv_len, total))
            all_pass, detector_breakdown = self.operator.filter(self.untrusted_data.data, thrs)
            both_acc, det_only_acc, ref_only_acc, none_acc = self.get_attack_acc(all_pass)
            print(detector_breakdown)
            both.append(both_acc)
            det_only.append(det_only_acc)
            ref_only.append(ref_only_acc)
            none.append(none_acc)

        size = 2.5
        plt.plot(confs, none, c="green", label="No fefense", marker="x", markersize=size)
        plt.plot(confs, det_only, c="orange", label="With detector", marker="o", markersize=size)
        plt.plot(confs, ref_only, c="blue", label="With reformer", marker="^", markersize=size)
        plt.plot(confs, both, c="red", label="With detector & reformer", marker="s", markersize=size)

        pylab.legend(loc='lower left', bbox_to_anchor=(0.02, 0.1), prop={'size':8})
        plt.grid(linestyle='dotted')
        plt.xlabel(r"Confidence in Carlini $L^2$ attack")
        plt.ylabel("Classification accuracy")
        plt.xlim(min(confs)-1.0, max(confs)+1.0)
        plt.ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))

        save_path = os.path.join(self.graph_dir, graph_name+".pdf")
        plt.savefig(save_path)
        plt.clf()

    def print(self):
        return " ".join([self.operator.print(), self.untrusted_data.print()])


def build_detector(detector_model_dir, detector_model_names, save_model_name, save_model_dir, model_path,
                   MODEL, det_model, data, data_format, is_det_joint, model_idx, gpu_count=1):
    det_dict = {}
    det_set = {}
    det_idx_set = {}
    dropout_rate_set = {}
    det_gpu_idx = {}

    for val in detector_model_names:
        if val == '':
            continue

        cur_det_name, cur_p, cur_det_type, cur_dropout_rate, cur_model_id = val.split('/')
        cur_model_id = int(cur_model_id)
        cur_det_path = os.path.join(detector_model_dir, cur_det_name)
        cur_detector = {
            "p": cur_p,
            "type": cur_det_type,
            "dropout_rate": cur_dropout_rate
        }
        det_dict[cur_det_name] = cur_detector

        if type(det_model) is list:
            cur_det_model = det_model[cur_model_id]
            cur_model_path = os.path.join(save_model_dir, save_model_name[cur_model_id])
            cur_det_idx = model_idx[cur_model_id]
        else:
            cur_det_model = det_model
            cur_model_path = model_path
            cur_det_idx = model_idx
        default_det_idx = cur_det_idx

        with tf.device('/gpu:' + str(cur_model_id % gpu_count)):
            # build detector
            print("# build detector: ", cur_det_name)
            print("type:", cur_det_type)
            print("p:", cur_p)
            print("drop_rate:", cur_dropout_rate)

            if cur_det_type == 'AED':
                cur_detector = AEDetector(cur_det_path, p=int(cur_p))
                cur_det_idx = load_model_idx(cur_det_path)
            elif cur_det_type == "DBD":
                id_reformer = IdReformer()
                print("# build reformer", cur_det_name)
                cur_reformer_t = SimpleReformer(cur_det_path)
                classifier = Classifier(cur_model_path, MODEL,
                                        data_format=data_format, model=cur_det_model)
                cur_detector = DBDetector(reconstructor=id_reformer, prober=cur_reformer_t,
                                          classifier=classifier, T=int(cur_p))
                cur_det_idx = load_model_idx(cur_det_path)

        if cur_det_idx is None:
            cur_det_idx = default_det_idx

        det_idx_set[cur_det_name] = cur_det_idx['validate']

        dropout_rate_set[cur_det_name] = float(cur_dropout_rate)
        det_set[cur_det_name] = cur_detector
        det_gpu_idx[cur_det_name] = cur_model_id % gpu_count

    # compute thrs
    thrs_set = {}
    det_info = {
        "model": save_model_name,
        "model_dir": save_model_dir,
        "det": det_dict,
        "det_dir": detector_model_dir,
        "joint_thrs": is_det_joint
    }

    cache_path = os.path.join(detector_model_dir, "cache")

    if is_det_joint:
        marks_set = []
        num = 0
        cache = load_cache(det_info, cache_path)
        if cache is None:
            cache_data = {}
            for cur_det_name, cur_det in det_set.items():
                validation_data = data.train_data_orig[det_idx_set[cur_det_name]]
                num = int(len(validation_data) * dropout_rate_set[cur_det_name])
                marks = cur_det.mark(validation_data, data_format=data_format)
                marks_set.append(marks)

                marks = np.sort(marks)
                cache_data[cur_det_name] = marks[-num]
                print("compute thrs for model #", cur_det_name, "#:", marks[-num])

            marks_set = np.transpose(marks_set)
            marks_max = np.max(marks_set, axis=1)
            marks_max = np.sort(marks_max)
            max_thrs = marks_max[-num]

            cache_data['thrs'] = max_thrs
            if len(det_set) > 0:
                hash_id = save_cache(det_info, cache_data, cache_path)
                print("save cache:", hash_id)
        else:
            print("hit cache:", cache['hash_id'])
            cache_data = cache['data']
            for cur_det_name, cur_det in det_set.items():
                print("compute thrs for model #", cur_det_name, "#:", cache_data[cur_det_name])
            max_thrs = cache_data['thrs']

        for cur_det_name, cur_det in det_set.items():
            thrs_set[cur_det_name] = max_thrs

        print("use joint thrs:", max_thrs)
    else:
        cache = load_cache(det_info, cache_path)
        if cache is None:
            cache_data = {}
            for cur_det_name, cur_det in det_set.items():
                validation_data = data.train_data_orig[det_idx_set[cur_det_name]]
                num = int(len(validation_data) * dropout_rate_set[cur_det_name])
                marks = cur_det.mark(validation_data, data_format=data_format)
                marks = np.sort(marks)

                thrs_set[cur_det_name] = marks[-num]
                cache_data[cur_det_name] = marks[-num]
                print("compute thrs for model #", cur_det_name, "#:", marks[-num])

            if len(det_set) > 0:
                hash_id = save_cache(det_info, cache_data, cache_path)
                print("save cache:", hash_id)
        else:
            print("hit cache:", cache['hash_id'])
            cache_data = cache['data']
            for cur_det_name, cur_det in det_set.items():
                thrs_set[cur_det_name] = cache_data[cur_det_name]
                print("compute thrs for model #", cur_det_name, "#:", cache_data[cur_det_name])

    return det_set, thrs_set, det_gpu_idx