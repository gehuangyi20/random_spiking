import os
import csv
import json
import argparse

parser = argparse.ArgumentParser(description='Fix incorrect loss value computed by random direction inference.')

parser.add_argument('-d', '--dir', help='directory, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('--noise_iter', help='add l2 times', type=int, default=20)

args = parser.parse_args()

if not os.path.isdir(args.dir):
    print("Directory", args.dir, "does not exist")
    exit(0)

_dir = args.dir
noise_iter = args.noise_iter

config_fp = open(os.path.join(_dir, args.config), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

model_name = config['model_name']
model_name_st = config['model_name_st']
l2 = config['l2']

for i in range(len(model_name)):
    cur_model_name = model_name[i]

    for j in range(len(l2[i])):
        cur_l2 = l2[i][j]

        filename = cur_model_name + '_' + str(cur_l2) + '.csv'
        cur_csvfile = open(os.path.join(_dir, filename), 'r')
        cur_reader = csv.DictReader(cur_csvfile, dialect='excel-tab')

        tmp_name = []
        tmp_l2 = []
        tmp_acc = []
        tmp_unchg = []
        tmp_loss = []

        # read data
        for row in cur_reader:
            tmp_name.append(row['name'])
            tmp_l2.append(row['l2'])
            tmp_acc.append(row['test_acc'])
            tmp_unchg.append(row['test_unchg'])
            # fix with iteration
            tmp_loss.append(float(row['test_loss']) / noise_iter)

        cur_csvfile.close()

        # write data
        cur_csvfile = open(os.path.join(_dir, filename), 'w')
        cur_csvfile.write("name\tl2\ttest_acc\ttest_unchg\ttest_loss\n")

        for k in range(len(tmp_name)):
            cur_csvfile.write('%s\t%s\t%s\t%s\t%.4f\n' %
                              (tmp_name[k], tmp_l2[k], tmp_acc[k], tmp_unchg[k], tmp_loss[k]))

        cur_csvfile.close()
