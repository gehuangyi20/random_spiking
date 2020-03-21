#!/usr/bin/python3
import os
import json
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import argparse
import numpy as np
from itertools import cycle

import matplotlib
matplotlib.rcParams['text.usetex'] = True

parser = argparse.ArgumentParser(description='Plot adv_diff vs transferability or confidence value.')

parser.add_argument('-d', '--dir', help='directory, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('-o', '--output', help='output name, default summary', type=str, default='summary')
parser.add_argument('-H', '--height', help='height, default 5', type=float, default=5)
parser.add_argument('-W', '--width', help='width, default 5', type=float, default=5)
parser.add_argument('-B', '--bin', help='number of bins', type=int, default=10)
parser.add_argument('-X', '--x_scale', help='x_scale: linear or log', type=str, default='linear')
parser.add_argument('-Y', '--y_scale', help='y_scale: linear or log', type=str, default='linear')
parser.add_argument('--fext', help='filename extension, default -raw', type=str, default='-raw')
parser.add_argument('--dcol_tran', help='column name for trans, default: round_trans', type=str, default='round_trans')
parser.add_argument('--dcol_pred', help='column name for trans, default: round_trans', type=str, default='round_pred')
parser.add_argument('--legend_col', help='number of columns of legend', type=int, default=1)
parser.add_argument('--bin_manual', help='manual_bin_range', type=str, default=None)
parser.add_argument('--shift_space', help='shift space between methods', type=float, default=None)
parser.add_argument('--ignore_upper', help='ignore examples having l2 norm great than ...', type=str, default='yes')
parser.add_argument('--style_index', help='style indexs for different methods', type=str, default='')
parser.add_argument('--y_count', help='y axis of l2 vs adv cdf', type=str, default='Adv Percentage')
parser.add_argument('--y_pass', help='y axis transfer', type=str, default='Validation Passing Rate')
parser.add_argument('--y_rotation', help='y axis rotation degree', type=float, default=0)

args = parser.parse_args()

_dir = args.dir
output_file = args.output
ignore_upper = args.ignore_upper == 'yes'

config_fp = open(os.path.join(_dir, args.config), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

style_index = np.arange(len(config)) if args.style_index == '' \
    else np.array([int(_x) for _x in args.style_index.split(',')])
style_index = style_index % 10

# mkdir
if not os.path.exists(os.path.dirname(os.path.join(_dir, output_file))):
    os.makedirs(os.path.dirname(os.path.join(_dir, output_file)))

# plot data initialization

data_set = []
data_set_manual = []
data_set_manual_count = []
manual_bin = [float(_x) for _x in args.bin_manual.split(',')]
num_manual_bin = len(manual_bin) + 1

l2_stat = []
l2_stat_bydataset = []
model_mapping = []
names = []
for pair in config:
    names.append(pair['name'])
    cur_raw_trans_fp = open(os.path.join(_dir, pair['transfer'][:-4] + args.fext + ".csv"), "r")
    cur_transfer_reader = csv.DictReader(cur_raw_trans_fp, dialect='excel-tab')

    cur_data_set = []
    cur_l2_stat_bydataset = []
    cur_model_mapping = {}
    cur_model_idx = 0
    for transfer_row in cur_transfer_reader:
        cur_model_id = transfer_row['model_id']
        if cur_model_id not in cur_model_mapping:
            cur_model_mapping[cur_model_id] = cur_model_idx
            cur_model_idx += 1

        cur_l2 = float(transfer_row['l2'])
        cur_trans = int(transfer_row[args.dcol_tran])
        cur_pred = int(transfer_row[args.dcol_pred])

        cur_data_set.append([cur_l2, cur_trans, cur_model_mapping[cur_model_id], cur_pred])
        l2_stat.append(cur_l2)
        cur_l2_stat_bydataset.append(cur_l2)

    # find which bin should the each point fall into
    cur_data_set_manual = np.digitize(cur_l2_stat_bydataset, manual_bin)
    data_set_manual.append(cur_data_set_manual)

    cur_data_set_manual_bins, cur_data_set_manual_bins_count = np.unique(cur_data_set_manual, return_counts=True)
    tmp_data_set_manual_bins_count = [0] * num_manual_bin
    for i in range(len(cur_data_set_manual_bins)):
        tmp_data_set_manual_bins_count[cur_data_set_manual_bins[i]] = cur_data_set_manual_bins_count[i]
    data_set_manual_count.append(tmp_data_set_manual_bins_count)

    l2_stat_bydataset.append(cur_l2_stat_bydataset)
    data_set.append(cur_data_set)
    model_mapping.append(cur_model_mapping)


num_bin = args.bin
num_dataset = len(config)

l2_stat_sort = np.sort(l2_stat)
l2_stat_sort_bydataset = [np.sort(_x) for _x in l2_stat_bydataset]

count_g = len(l2_stat)
count_bydataset = [len(_x) for _x in l2_stat_bydataset]

l2_thrs_idx = []
l2_thrs_idx_bydataset = [[] for _x in names]
l2_thrs = []
l2_thrs_bydataset = [[] for _x in names]

for i in range(num_bin):
    if i + 1 < num_bin:
        idx = int(count_g/num_bin*(i+1))
    else:
        idx = count_g - 1
    l2_thrs.append(l2_stat_sort[idx])
    l2_thrs_idx.append(idx)
    for j in range(num_dataset):
        if i + 1 < num_bin:
            idx = int(count_bydataset[j]/num_bin*(i+1))
        else:
            idx = count_bydataset[j] - 1
        l2_thrs_bydataset[j].append(l2_stat_sort_bydataset[j][idx])
        l2_thrs_idx_bydataset[j].append(idx)


def write_fp(fp, string):
    print(string)
    fp.write((string + '\n').encode())


# print thrs
fp_l2_thrs = open(os.path.join(_dir,  output_file + "-l2_thrs.csv"), "wb")
write_fp(fp_l2_thrs, "Global: " + str(count_g))
write_fp(fp_l2_thrs, '\t'.join(['%d' % _x for _x in l2_thrs_idx]))
write_fp(fp_l2_thrs, '\t'.join(['%.4f' % _x for _x in l2_thrs]))

for i in range(num_dataset):
    write_fp(fp_l2_thrs, names[i] + ": " + str(count_bydataset[i]))
    write_fp(fp_l2_thrs, '\t'.join(['%d' % _x for _x in l2_thrs_idx_bydataset[i]]))
    write_fp(fp_l2_thrs, '\t'.join(['%.4f' % _x for _x in l2_thrs_bydataset[i]]))

write_fp(fp_l2_thrs, '\nmanual_bins: ' + '\t'.join(['%.4f' % _x for _x in manual_bin]))
write_fp(fp_l2_thrs, 'number of advs in each bin for each data method')

for i in range(num_dataset):
    write_fp(fp_l2_thrs, names[i] + ": " + str(count_bydataset[i]))
    write_fp(fp_l2_thrs, '\t'.join(['%d' % _x for _x in data_set_manual_count[i]]))
    write_fp(fp_l2_thrs, '\t'.join(['%.4f' % (_x/count_bydataset[i]) for _x in data_set_manual_count[i]]))

fp_l2_thrs.close()

global_l2 = []
global_transfer = []
global_pred = []
dataset_l2 = []
dataset_transfer = []
dataset_pred = []
dataset_mn_l2 = []
dataset_mn_transfer = []
dataset_mn_pred = []

for i in range(num_dataset):
    tmp_l2 = []
    tmp_transfer = []
    tmp_pred = []
    tmp_dt_l2 = []
    tmp_dt_transfer = []
    tmp_dt_pred = []
    tmp_mn_l2 = []
    tmp_mn_transfer = []
    tmp_mn_pred = []

    for j in range(num_bin):
        tmp_l2_bin = []
        tmp_transfer_bin = []
        tmp_pred_bin = []
        tmp_dt_l2_bin = []
        tmp_dt_transfer_bin = []
        tmp_dt_pred_bin = []

        for k in range(len(model_mapping[i])):
            tmp_l2_bin.append([])
            tmp_transfer_bin.append([])
            tmp_pred_bin.append([])
            tmp_dt_l2_bin.append([])
            tmp_dt_transfer_bin.append([])
            tmp_dt_pred_bin.append([])

        tmp_l2.append(tmp_l2_bin)
        tmp_transfer.append(tmp_transfer_bin)
        tmp_pred.append(tmp_pred_bin)
        tmp_dt_l2.append(tmp_dt_l2_bin)
        tmp_dt_transfer.append(tmp_dt_transfer_bin)
        tmp_dt_pred.append(tmp_dt_pred_bin)

    global_l2.append(tmp_l2)
    global_transfer.append(tmp_transfer)
    global_pred.append(tmp_pred)
    dataset_l2.append(tmp_dt_l2)
    dataset_transfer.append(tmp_dt_transfer)
    dataset_pred.append(tmp_dt_pred)

    for j in range(num_manual_bin):
        tmp_mn_l2_bin = []
        tmp_mn_transfer_bin = []
        tmp_mn_pred_bin = []

        for k in range(len(model_mapping[i])):
            tmp_mn_l2_bin.append([])
            tmp_mn_transfer_bin.append([])
            tmp_mn_pred_bin.append([])

        tmp_mn_l2.append(tmp_mn_l2_bin)
        tmp_mn_transfer.append(tmp_mn_transfer_bin)
        tmp_mn_pred.append(tmp_mn_pred_bin)

    dataset_mn_l2.append(tmp_mn_l2)
    dataset_mn_transfer.append(tmp_mn_transfer)
    dataset_mn_pred.append(tmp_mn_pred)


for i in range(num_dataset):
    cur_data_set = data_set[i]
    cur_l2_thrs_dataset = l2_thrs_bydataset[i]
    cur_l2_stat_bydataset = l2_stat_bydataset[i]

    cur_g_bin = np.digitize(cur_l2_stat_bydataset, l2_thrs, right=True)
    cur_dt_bin = np.digitize(cur_l2_stat_bydataset, cur_l2_thrs_dataset, right=True)
    cur_mn_bin = data_set_manual[i]

    for row_i in range(count_bydataset[i]):
        cur_row = cur_data_set[row_i]
        cur_l2 = cur_row[0]
        cur_trans = cur_row[1]
        cur_pred = cur_row[3]
        cur_model_id = cur_row[2]

        cur_bin_i = cur_g_bin[row_i]
        global_l2[i][cur_bin_i][cur_model_id].append(cur_l2)
        global_transfer[i][cur_bin_i][cur_model_id].append(cur_trans)
        global_pred[i][cur_bin_i][cur_model_id].append(cur_pred)

        cur_bin_i = cur_dt_bin[row_i]
        dataset_l2[i][cur_bin_i][cur_model_id].append(cur_l2)
        dataset_transfer[i][cur_bin_i][cur_model_id].append(cur_trans)
        dataset_pred[i][cur_bin_i][cur_model_id].append(cur_pred)

        cur_bin_i = cur_mn_bin[row_i]
        dataset_mn_l2[i][cur_bin_i][cur_model_id].append(cur_l2)
        dataset_mn_transfer[i][cur_bin_i][cur_model_id].append(cur_trans)
        dataset_mn_pred[i][cur_bin_i][cur_model_id].append(cur_pred)


# store stat
fp_l2_stat_model = open(os.path.join(_dir, output_file + "-l2_stat_model.csv"), "wb")
fp_l2_stat_model.write("dataset\tmodel_id\tbin\tl2_mean\tl2_std\ttrans_count\ttotal_count\ttrans_rate\tpred_count\ttp_count\tpred_rate\n".encode())
fp_l2_stat_model_dt = open(os.path.join(_dir, output_file + "-l2_stat_model_dt.csv"), "wb")
fp_l2_stat_model_dt.write("dataset\tmodel_id\tbin\tl2_mean\tl2_std\ttrans_count\ttotal_count\ttrans_rate\tpred_count\ttp_count\tpred_rate\n".encode())
fp_l2_stat_model_mn = open(os.path.join(_dir, output_file + "-l2_stat_model_mn.csv"), "wb")
fp_l2_stat_model_mn.write("dataset\tmodel_id\tbin\tbin_r\tl2_mean\tl2_std\ttrans_count\ttotal_count\ttrans_rate\tpred_count\ttp_count\tpred_rate\n".encode())

fp_l2_stat = open(os.path.join(_dir, output_file + "-l2_stat.csv"), "wb")
fp_l2_stat.write("dataset\tbin\tl2_mean\tl2_std\ttrans_rate_mean\ttrans_rate_std\tpred_rate_mean\tpred_rate_std\n".encode())
fp_l2_stat_dt = open(os.path.join(_dir, output_file + "-l2_stat_dt.csv"), "wb")
fp_l2_stat_dt.write("dataset\tbin\tl2_mean\tl2_std\ttrans_rate_mean\ttrans_rate_std\tpred_rate_mean\tpred_rate_std\n".encode())
fp_l2_stat_mn = open(os.path.join(_dir, output_file + "-l2_stat_mn.csv"), "wb")
fp_l2_stat_mn.write("dataset\tbin\tbin_r\tl2_mean\tl2_std\ttrans_rate_mean\ttrans_rate_std\tpred_rate_mean\tpred_rate_std\n".encode())

global_l2_mean = []
global_l2_std = []
global_tran_mean = []
global_tran_std = []
global_pred_mean = []
global_pred_std = []
dt_l2_mean = []
dt_l2_std = []
dt_tran_mean = []
dt_tran_std = []
dt_pred_mean = []
dt_pred_std = []
mn_l2 = []
mn_l2_mean = []
mn_l2_std = []
mn_tran_mean = []
mn_tran_std = []
mn_pred_mean = []
mn_pred_std = []

for i in range(num_dataset):
    cur_data_set = data_set[i]
    cur_name = names[i]
    cur_model_mapping = model_mapping[i]

    tmp_l2_mean = []
    tmp_l2_std = []
    tmp_trans_mean = []
    tmp_trans_std = []
    tmp_pred_mean = []
    tmp_pred_std = []
    tmp_dt_l2_mean = []
    tmp_dt_l2_std = []
    tmp_dt_trans_mean = []
    tmp_dt_trans_std = []
    tmp_dt_pred_mean = []
    tmp_dt_pred_std = []

    for bin_i in range(num_bin):
        cur_l2_cat = []
        cur_dt_l2_cat = []
        for cur_model_name in cur_model_mapping:
            cur_model_idx = cur_model_mapping[cur_model_name]

            cur_l2_ref = global_l2[i][bin_i][cur_model_idx]
            cur_l2_cat.extend(cur_l2_ref)
            cur_l2_mean = np.mean(cur_l2_ref)
            cur_l2_std = np.std(cur_l2_ref)
            global_l2[i][bin_i][cur_model_idx] = cur_l2_mean

            cur_tran_ref = global_transfer[i][bin_i][cur_model_idx]
            cur_tran_total = len(cur_tran_ref)
            cur_tran_count = np.sum(cur_tran_ref)
            cur_tran = np.mean(cur_tran_ref)
            global_transfer[i][bin_i][cur_model_idx] = cur_tran

            cur_pred_ref = global_pred[i][bin_i][cur_model_idx]
            cur_pred_total = len(cur_pred_ref)
            cur_pred_count = np.sum(cur_pred_ref)
            cur_pred = np.mean(cur_pred_ref)
            global_pred[i][bin_i][cur_model_idx] = cur_pred

            fp_l2_stat_model.write(("%s\t%s\t%d\t%.4f\t%.4f\t%d\t%d\t%.4f\t%d\t%d\t%.4f\n" % (
                cur_name, cur_model_name, bin_i, cur_l2_mean,
                cur_l2_std, cur_tran_count, cur_tran_total, cur_tran,
                cur_pred_count, cur_pred_total, cur_pred)).encode())

            cur_l2_ref = dataset_l2[i][bin_i][cur_model_idx]
            cur_dt_l2_cat.extend(cur_l2_ref)
            cur_l2_mean = np.mean(cur_l2_ref)
            cur_l2_std = np.std(cur_l2_ref)
            dataset_l2[i][bin_i][cur_model_idx] = cur_l2_mean

            cur_tran_ref = dataset_transfer[i][bin_i][cur_model_idx]
            cur_tran_total = len(cur_tran_ref)
            cur_tran_count = np.sum(cur_tran_ref)
            cur_tran = np.mean(cur_tran_ref)
            dataset_transfer[i][bin_i][cur_model_idx] = cur_tran

            cur_pred_ref = dataset_pred[i][bin_i][cur_model_idx]
            cur_pred_total = len(cur_pred_ref)
            cur_pred_count = np.sum(cur_pred_ref)
            cur_pred = np.mean(cur_pred_ref)
            dataset_pred[i][bin_i][cur_model_idx] = cur_pred

            fp_l2_stat_model_dt.write(("%s\t%s\t%d\t%.4f\t%.4f\t%d\t%d\t%.4f\t%d\t%d\t%.4f\n" % (
                cur_name, cur_model_name, bin_i, cur_l2_mean,
                cur_l2_std, cur_tran_count, cur_tran_total, cur_tran,
                cur_pred_count, cur_pred_total, cur_pred)).encode())

        tmp_l2_mean.append(np.mean(cur_l2_cat))
        tmp_l2_std.append(np.std(cur_l2_cat))
        tmp_trans_mean.append(np.mean(global_transfer[i][bin_i]))
        tmp_trans_std.append(np.std(global_transfer[i][bin_i]))
        tmp_pred_mean.append(np.mean(global_pred[i][bin_i]))
        tmp_pred_std.append(np.std(global_pred[i][bin_i]))
        fp_l2_stat.write(("%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (
            cur_name, bin_i, tmp_l2_mean[-1], tmp_l2_std[-1], tmp_trans_mean[-1], tmp_trans_std[-1],
            tmp_pred_mean[-1], tmp_pred_std[-1])).encode())

        tmp_dt_l2_mean.append(np.mean(cur_dt_l2_cat))
        tmp_dt_l2_std.append(np.std(cur_dt_l2_cat))
        tmp_dt_trans_mean.append(np.mean(dataset_transfer[i][bin_i]))
        tmp_dt_trans_std.append(np.std(dataset_transfer[i][bin_i]))
        tmp_dt_pred_mean.append(np.mean(dataset_pred[i][bin_i]))
        tmp_dt_pred_std.append(np.std(dataset_pred[i][bin_i]))
        fp_l2_stat_dt.write(("%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (
            cur_name, bin_i, tmp_dt_l2_mean[-1], tmp_dt_l2_std[-1],
            tmp_dt_trans_mean[-1], tmp_dt_trans_std[-1], tmp_dt_pred_mean[-1], tmp_dt_pred_std[-1])).encode())

    global_l2_mean.append(tmp_l2_mean)
    global_l2_std.append(tmp_l2_std)
    global_tran_mean.append(tmp_trans_mean)
    global_tran_std.append(tmp_trans_std)
    global_pred_mean.append(tmp_pred_mean)
    global_pred_std.append(tmp_pred_std)

    dt_l2_mean.append(tmp_dt_l2_mean)
    dt_l2_std.append(tmp_dt_l2_std)
    dt_tran_mean.append(tmp_dt_trans_mean)
    dt_tran_std.append(tmp_dt_trans_std)
    dt_pred_mean.append(tmp_dt_pred_mean)
    dt_pred_std.append(tmp_dt_pred_std)

    if args.x_scale == 'log':
        init_l2 = manual_bin[0]**2 / manual_bin[1]
    else:
        init_l2 = manual_bin[0]*2 - manual_bin[1]
    tmp_mn_l2 = np.insert(manual_bin, 0, init_l2)
    mn_l2.append(tmp_mn_l2)
    tmp_mn_l2_mean = []
    tmp_mn_l2_std = []
    tmp_mn_trans_mean = []
    tmp_mn_trans_std = []
    tmp_mn_pred_mean = []
    tmp_mn_pred_std = []

    for bin_i in range(num_manual_bin):
        cur_mn_l2_cat = []
        for cur_model_name in cur_model_mapping:
            cur_model_idx = cur_model_mapping[cur_model_name]

            cur_l2_ref = dataset_mn_l2[i][bin_i][cur_model_idx]
            cur_mn_l2_cat.extend(cur_l2_ref)
            cur_l2_mean = np.mean(cur_l2_ref)
            cur_l2_std = np.std(cur_l2_ref)
            dataset_mn_l2[i][bin_i][cur_model_idx] = cur_l2_mean

            cur_tran_ref = dataset_mn_transfer[i][bin_i][cur_model_idx]
            cur_tran_total = len(cur_tran_ref)
            cur_tran_count = np.sum(cur_tran_ref)
            cur_tran = np.mean(cur_tran_ref)
            dataset_mn_transfer[i][bin_i][cur_model_idx] = cur_tran

            cur_pred_ref = dataset_mn_pred[i][bin_i][cur_model_idx]
            cur_pred_total = len(cur_pred_ref)
            cur_pred_count = np.sum(cur_pred_ref)
            cur_pred = np.mean(cur_pred_ref)
            dataset_mn_pred[i][bin_i][cur_model_idx] = cur_pred

            fp_l2_stat_model_mn.write(("%s\t%s\t%d\t%.4f\t%.4f\t%.4f\t%d\t%d\t%.4f\t%d\t%d\t%.4f\n" % (
                cur_name, cur_model_name, bin_i, tmp_mn_l2[bin_i], cur_l2_mean,
                cur_l2_std, cur_tran_count, cur_tran_total, cur_tran,
                cur_pred_count, cur_pred_total, cur_pred)).encode())

        tmp_mn_l2_mean.append(np.mean(cur_mn_l2_cat))
        tmp_mn_l2_std.append(np.std(cur_mn_l2_cat))
        tmp_mn_trans_mean.append(np.mean(dataset_mn_transfer[i][bin_i]))
        tmp_mn_trans_std.append(np.std(dataset_mn_transfer[i][bin_i]))
        tmp_mn_pred_mean.append(np.mean(dataset_mn_pred[i][bin_i]))
        tmp_mn_pred_std.append(np.std(dataset_mn_pred[i][bin_i]))
        fp_l2_stat_mn.write(("%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (
            cur_name, bin_i, tmp_mn_l2[bin_i], tmp_mn_l2_mean[-1], tmp_mn_l2_std[-1],
            tmp_mn_trans_mean[-1], tmp_mn_trans_std[-1], tmp_mn_pred_mean[-1], tmp_mn_pred_std[-1])).encode())

    mn_l2_mean.append(tmp_mn_l2_mean)
    mn_l2_std.append(tmp_mn_l2_std)
    mn_tran_mean.append(tmp_mn_trans_mean)
    mn_tran_std.append(tmp_mn_trans_std)
    mn_pred_mean.append(tmp_mn_pred_mean)
    mn_pred_std.append(tmp_mn_pred_std)

fp_l2_stat_model.close()
fp_l2_stat_model_dt.close()
fp_l2_stat_model_mn.close()
fp_l2_stat.close()
fp_l2_stat_dt.close()
fp_l2_stat_mn.close()


width = args.width
height = args.height
x_scale = args.x_scale
y_scale = args.y_scale


def plt_figure(output_name, x_label, y_label, l2_mean, l2_std, tran_mean, tran_std):

    fg = plt.figure(figsize=(width, height))
    ax = fg.add_subplot(111)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    line_style_cycle = ['o', '>', 'D', '*', 'p', 'P', 's', 'X', '^', 'v']
    color_style_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    err_bar_style = {
        'markersize': 8,
        'markeredgewidth': 0.9,
        'alpha': 0.7,
        'capsize': 3,
        'capthick': 0.9,
        'elinewidth': 0.9,
        'linewidth': 0.9
    }
    legend_style = {
        'ncol': args.legend_col,
        'loc': 'best',
        'borderaxespad': 0.5,
        'framealpha': 0.5
    }

    for i in range(num_dataset):
        cur_name = names[i]
        cur_l2_mean = l2_mean[i]
        cur_l2_std = l2_std[i]
        cur_tran_mean = tran_mean[i]
        cur_tran_std = tran_std[i]
        cur_style = color_style_cycle[style_index[i]] + line_style_cycle[style_index[i]]

        ax.errorbar(cur_l2_mean, cur_tran_mean, xerr=cur_l2_std, yerr=cur_tran_std, fmt=cur_style,
                    label=cur_name, **err_bar_style)

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]

    #ax.legend(handles=handles, labels=labels, **legend_style)
    fg.tight_layout(pad=0)

    fg.savefig(output_name)


plt_figure(os.path.join(_dir, output_file + "-l2_g.pdf"), r'$L_2$', 'Transferability',
           global_l2_mean, global_l2_std, global_tran_mean, global_tran_std)
plt_figure(os.path.join(_dir, output_file + "-l2_dt.pdf"), r'$L_2$', 'Transferability',
           dt_l2_mean, dt_l2_std, dt_tran_mean, dt_tran_std, )


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plt_bar_v2(output_name, x_label, y_label, l2, tran_mean, tran_std, manual_count):

    fg = plt.figure(figsize=(width, height))
    fg_c = plt.figure(figsize=(width, height))
    ax = fg.add_subplot(111)
    ax_c = fg_c.add_subplot(111)

    xtrick_text = [str(round(_x, 2)) for _x in l2[0]]
    xtrick_text[0] = r'$\geq$0'
    xtrick_text[-1] = r'$\geq$' + xtrick_text[-1]
    print(xtrick_text)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(-0.00, 1.04)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(l2[0])
    ax.set_xticklabels(xtrick_text, rotation=args.y_rotation)

    ax_c.set_xlabel(x_label)
    ax_c.set_ylabel(args.y_count)
    ax_c.set_ylim(-4, 104)
    ax_c.set_xscale(x_scale)
    ax_c.xaxis.set_minor_formatter(NullFormatter())
    ax_c.xaxis.set_major_formatter(NullFormatter())
    ax_c.set_xticks(l2[0])
    ax_c.set_xticklabels(xtrick_text, rotation=args.y_rotation)

    if ignore_upper:
        right_lim = l2[0][-1] + (l2[0][-1] - l2[0][0]) * 0.05
        left_lim = l2[0][0] - (l2[0][-1] - l2[0][0]) * 0.05
        ax.set_xlim(left=left_lim, right=right_lim)
        ax_c.set_xlim(left=left_lim, right=right_lim)

    line_style_cycle = ['o', '>', 'D', '*', 'p', 'P', 's', 'X', '^', 'v']
    hatch_style_cycle = ['', '///', 'ooo', 'OOO', '...', '***', 'xxx', '\\\\\\', '+++', '---']
    color_style_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    err_bar_style = {
        'markersize': 8,
        'markeredgewidth': 0.9,
        'alpha': 0.7,
        'capsize': 3,
        'capthick': 0.9,
        'elinewidth': 0.9,
        'linewidth': 0.9
    }
    bar_style = {
        'linewidth': 0.8,
        'alpha': 0.8,
        'capsize': 2,
        'ecolor': (0, 0, 0, 0.8)
    }

    legend_style = {
        'ncol': args.legend_col,
        'loc': 'best',
        'borderaxespad': 0.5,
        'framealpha': 0.5
    }

    if x_scale == 'log':
        bar_width = np.diff(np.logspace(np.log10(l2[0][0]), np.log10(manual_bin[-1]**2 / manual_bin[-2]),
                                        num=num_manual_bin * (num_dataset+3) + 1))
        shift_space = bar_width[1] / bar_width[0]
    else:
        bar_width = [(manual_bin[-1] + 2*manual_bin[1] - 3*manual_bin[0]) / num_manual_bin / (num_dataset+3)] \
                    * num_manual_bin * (num_dataset+3)
        shift_space = bar_width[0]

    if args.shift_space is not None:
        shift_space = args.shift_space
    bar_width = np.reshape(bar_width, (num_manual_bin, num_dataset+3)).transpose() * 0.90
    if ignore_upper:
        cur_bin_num = num_manual_bin-1
    else:
        cur_bin_num = num_manual_bin

    for i in range(num_dataset):
        cur_name = names[i]
        cur_l2 = np.array(l2[i])[:cur_bin_num]
        if x_scale == 'log':
            cur_l2 *= shift_space**i
        else:
            cur_l2 += i * shift_space
        cur_tran_mean = tran_mean[i][:cur_bin_num]
        cur_tran_std = tran_std[i][:cur_bin_num]
        cur_count = (np.array(manual_count[i])/np.sum(manual_count[i]))[:cur_bin_num]
        cur_count = np.cumsum(cur_count) * 100
        cur_color = color_style_cycle[style_index[i]]
        cur_hatch = hatch_style_cycle[style_index[i]]
        cur_style = cur_color + line_style_cycle[style_index[i]]

        # ax.errorbar(cur_l2, cur_tran_mean, yerr=cur_tran_std, fmt=cur_style,
        #             label=cur_name, **err_bar_style)
        print(bar_width[i])
        ax.bar(cur_l2, cur_tran_mean, width=bar_width[i][:cur_bin_num], yerr=cur_tran_std, color=(1,1,1,0), edgecolor=cur_color, align='edge', hatch=cur_hatch,
                    label=cur_name, **bar_style)
        ax_c.errorbar(cur_l2, cur_count, label=cur_name, fmt=cur_style+'-',  **err_bar_style)

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]

    handles_c, labels_c = ax_c.get_legend_handles_labels()

    fg.tight_layout(pad=0)
    fg_c.tight_layout(pad=0)

    fg.savefig(output_name)
    fg_c.savefig(output_name[:-4] + '_count.pdf')

    lg_fg = plt.figure(figsize=(width, height))
    lg_fg_c = plt.figure(figsize=(width, height))
    lg_ax = lg_fg.add_subplot(111)
    lg_ax_c = lg_fg_c.add_subplot(111)

    lg_ax.set_axis_off()
    lg_ax_c.set_axis_off()
    tmp_lg = lg_ax.legend(handles=handles, labels=labels, **legend_style)
    export_legend(tmp_lg, output_name[:-4] + '_lg.pdf')
    tmp_lg = lg_ax_c.legend(handles=handles_c, labels=labels_c, **legend_style)
    export_legend(tmp_lg, output_name[:-4] + '_count_lg.pdf')


plt_bar_v2(os.path.join(_dir, output_file + "-l2_mn.pdf"), r'$L_2$', args.y_pass,
           mn_l2, mn_tran_mean, mn_tran_std, data_set_manual_count)
