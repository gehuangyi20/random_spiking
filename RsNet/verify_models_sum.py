#!/usr/bin/python3
import os
import csv
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser(description='summarize model accuracy for different method')
parser.add_argument('--dir', help='accuracy save dir, required', type=str, default=None)
parser.add_argument('--output', help='output filename, required', type=str, default=None)
parser.add_argument('--config', help='config file, required', type=str, default="config.json")
parser.add_argument('--list', help='list file for each method', type=str, default="list.json")

args = parser.parse_args()

_dir = args.dir
output_file = args.output
config_filename = args.config
list_filename = args.list

config = utils.load_json(os.path.join(_dir, config_filename))

out_fp = open(os.path.join(_dir, output_file), 'wb')

out_fp.write('name\ttest_acc_mean\ttest_acc_std\ttest_acc_top_k_mean\ttest_acc_top_k_std'
             '\ttest_loss_mean\ttest_loss_std'
             '\ttrain_acc_mean\ttrain_acc_std\ttrain_acc_top_k_mean\ttrain_acc_top_k_std'
             '\ttrain_loss_mean\ttrain_loss_std\n'.encode())


for cur_mthd in config:
    cur_mthd_name = cur_mthd['name']
    cur_mthd_dir = cur_mthd['dir']

    cur_mthd_config = utils.load_json(os.path.join(_dir, cur_mthd_dir, list_filename))
    if cur_mthd_config is None:
        cur_mthd_config = utils.load_json(os.path.join(_dir, list_filename))

    cur_mthd_fp = open(os.path.join(_dir, cur_mthd_dir, output_file), 'wb')
    cur_mthd_fp.write((cur_mthd_name + '\n\n\n').encode())

    cur_mthd_fp.write('name\ttest_acc\ttest_acc_top_k\ttest_loss\ttrain_acc\ttrain_acc_top_k\ttrain_loss\n'.encode())

    test_acc = []
    test_acc_top_k = []
    test_loss = []
    train_acc = []
    train_acc_top_k = []
    train_loss = []
    for cur_set in cur_mthd_config:

        cur_mthd_model_fp = open(os.path.join(_dir, cur_mthd_dir, cur_set), 'r')
        cur_reader = csv.DictReader(cur_mthd_model_fp, dialect='excel-tab')

        for row in cur_reader:
            test_acc.append(float(row['test_acc']))
            if 'test_acc_top_k' in row:
                test_acc_top_k.append(float(row['test_acc_top_k']))
            else:
                test_acc_top_k.append(float(row['test_acc']))
            test_loss.append(float(row['test_loss']))
            train_acc.append(float(row['train_acc']))
            if 'train_acc_top_k' in row:
                train_acc_top_k.append(float(row['train_acc_top_k']))
            else:
                train_acc_top_k.append(float(row['train_acc']))
            train_loss.append(float(row['train_loss']))

            cur_mthd_fp.write((cur_set + '\t' + str(test_acc[-1]) + "\t" + str(test_acc_top_k[-1]) + "\t" +
                               str(test_loss[-1]) + "\t" + str(train_acc[-1]) + "\t" +
                               str(train_loss[-1]) + "\t" + str(train_acc_top_k[-1]) + '\n').encode())

        cur_mthd_model_fp.close()

    cur_mthd_fp.write('\nsummary\n'.encode())
    cur_mthd_fp.write('stat\ttest_acc\ttest_acc_top_k\ttest_loss\ttrain_acc\ttrain_acc_top_k\ttrain_loss\n'.encode())
    cur_mthd_fp.write(('mean\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' %
                       (np.mean(test_acc), np.mean(test_acc_top_k), np.mean(test_loss),
                        np.mean(train_acc), np.mean(train_acc_top_k), np.mean(train_loss))).encode())
    cur_mthd_fp.write(('std\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' %
                       (np.std(test_acc), np.std(test_acc_top_k), np.std(test_loss),
                        np.std(train_acc), np.std(train_acc_top_k), np.std(train_loss))).encode())

    cur_mthd_fp.close()

    out_fp.write(("%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" %
                  (cur_mthd_name, np.mean(test_acc), np.std(test_acc), np.mean(test_acc_top_k), np.std(test_acc_top_k),
                   np.mean(test_loss), np.std(test_loss), np.mean(train_acc), np.std(train_acc),
                   np.mean(train_acc_top_k), np.std(train_acc_top_k),
                   np.mean(train_loss), np.std(train_loss))).encode())

out_fp.close()
