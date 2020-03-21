import numpy as np
import argparse
import os
import json
import csv

parser = argparse.ArgumentParser(description='stat region based classifier')

parser.add_argument('-d', '--dir', help='directory, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('-o', '--output', help='output name, default summary', type=str, default='summary.csv')

args = parser.parse_args()

_dir = args.dir
output_file = args.output

config_fp = open(os.path.join(_dir, args.config), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

# open output stat summary csv
output_csv = open(os.path.join(_dir, output_file), 'wb')
output_csv.write("name\tr_dist\ttest_acc\ttest_acc_std\ttest_loss\ttest_loss_std\n".encode())

names = []
keys = []
test_acc = {}
test_loss = {}


for filename in config:
    cur_fp = open(os.path.join(_dir, filename), 'r')
    cur_reader = csv.DictReader(cur_fp, dialect='excel-tab')

    for row in cur_reader:
        idx = float(row['r_dist'])
        if idx not in test_acc:
            test_acc[idx] = []
            test_loss[idx] = []
            keys.append(idx)
            names.append(row['name'])

        test_acc[idx].append(float(row['test_acc']))
        test_loss[idx].append(float(row['test_loss']))
    cur_fp.close()

for i in range(len(keys)):
    dist = keys[i]
    output_csv.write(("%s\t%.3f\t%.4f\t%.4f\t%.4f\t%.4f\n" %
                      (names[i], dist, np.mean(test_acc[dist]), np.std(test_acc[dist]),
                       np.mean(test_loss[dist]), np.std(test_loss[dist]))).encode())
output_csv.close()
