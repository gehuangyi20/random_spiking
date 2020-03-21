#!/usr/bin/python3
import os
import sys
import json
import csv
import numpy as np


_dir = sys.argv[1]
output_file = sys.argv[2]

config_fp = open(os.path.join(_dir, "list.json"), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

fp = open(os.path.join(_dir, output_file), 'wb')
fp.write((_dir + '\n\n\n').encode())

fp.write('name\ttest_acc\ttest_loss\ttrain_acc\ttrain_loss\n'.encode())

test_acc = []
test_loss = []
train_acc = []
train_loss = []
for cur_set in config:

    cur_fp = open(os.path.join(_dir, cur_set), 'r')
    cur_reader = csv.DictReader(cur_fp, dialect='excel-tab')

    for row in cur_reader:
        test_acc.append(float(row['test_acc']))
        test_loss.append(float(row['test_loss']))
        train_acc.append(float(row['train_acc']))
        train_loss.append(float(row['train_loss']))

        fp.write((cur_set + '\t' + str(test_acc[-1]) + "\t" + str(test_loss[-1]) + "\t" +
                  str(train_acc[-1]) + "\t" + str(train_loss[-1]) + '\n').encode())

    cur_fp.close()

fp.write('\nsummary\n'.encode())
fp.write('stat\ttest_acc\ttest_loss\ttrain_acc\ttrain_loss\n'.encode())
fp.write(('mean\t%.4f\t%.4f\t%.4f\t%.4f\n' %
          (np.mean(test_acc), np.mean(test_loss), np.mean(train_acc), np.mean(train_loss),)).encode())
fp.write(('std\t%.4f\t%.4f\t%.4f\t%.4f\n' %
          (np.std(test_acc), np.std(test_loss), np.std(train_acc), np.std(train_loss),)).encode())

fp.close()
