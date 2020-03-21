#!/usr/bin/python3
import os
import json
import csv
import matplotlib.pyplot as plt
import argparse
import numpy as np
from itertools import cycle

parser = argparse.ArgumentParser(description='Plot adv_diff vs transferability or confidence value.')

parser.add_argument('-d', '--dir', help='directory, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('-o', '--output', help='output name, default summary', type=str, default='summary')
parser.add_argument('-H', '--height', help='height, default 5', type=float, default=5)
parser.add_argument('-W', '--width', help='width, default 5', type=float, default=5)
parser.add_argument('--tran_low', help='transfer lower bound', type=float, default=0)
parser.add_argument('--tran_high', help='transfer upper bound', type=float, default=1)
parser.add_argument('--att_low', help='att suc lower bound', type=float, default=None)
parser.add_argument('--att_high', help='att suc upper bound', type=float, default=None)
parser.add_argument('--l2_low', help='l2 vs conf l2 upper bound', type=float, default=None)
parser.add_argument('--l2_high', help='l2 vs conf l2 upper bound', type=float, default=None)
parser.add_argument('--legend_col', help='number of columns of legend', type=int, default=1)
parser.add_argument('--marker_size', help='', type=float, default=7)
parser.add_argument('--conf', help='list of confidence', type=str, default='')
parser.add_argument('--shift_space', help='shift space between methods', type=float, default=1.0)
parser.add_argument('--style_index', help='style indexs for different methods', type=str, default='')

args = parser.parse_args()

_dir = args.dir
output_file = args.output
conf_list = [float(_x) for _x in args.conf.split(',')]

config_fp = open(os.path.join(_dir, args.config), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

style_index = np.arange(len(config)) if args.style_index == '' \
    else np.array([int(_x) for _x in args.style_index.split(',')])
style_index = style_index % 10

# open output stat summary csv
output_csv = open(os.path.join(_dir, output_file + ".csv"), 'wb')
output_csv.write("adv_model\tconfidence\ttest_model\ttransfer_rate_mean\ttransfer_rate_std"
                 "\tpred_acc_mean\tpred_acc_std\tl0_mean\tl0_std\tl1_mean\tl1_std"
                 "\tl2_mean\tl2_std\tl_inf_mean\tl_inf_std\tattack_success\tattack_success_std\n".encode())

# plot data initialization
line_style_cycle = ['o', '>', 'D', '*', 'p', 'P', 's', 'X', '^', 'v']
color_style_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
x_transfer = []
x_transfer_std = []
x_pred = []
x_pred_std = []
x_confidence = []
x_name = []
x_st_name = []
x_line_style = []
x_color_style = []

y_l0_mean = []
y_l1_mean = []
y_l2_mean = []
y_l_inf_mean = []
att_suc_mean = []

y_l0_std = []
y_l1_std = []
y_l2_std = []
y_l_inf_std = []
att_suc_std = []

i = -1
for pair in config:
    i += 1
    cur_transfer_fp = open(os.path.join(_dir, pair['transfer']), "r")
    cur_diff_fp = open(os.path.join(_dir, pair['diff']), "r")

    cur_name = None
    cur_x_transfer = None
    cur_x_transfer_std = None
    cur_x_pred = None
    cur_x_pred_std = None
    cur_x_confidence = None
    cur_line_style = line_style_cycle[style_index[i]]
    cur_color = color_style_cycle[style_index[i]]

    cur_y_l0_mean = None
    cur_y_l1_mean = None
    cur_y_l2_mean = None
    cur_y_l_inf_mean = None
    cur_att_suc_mean = None

    cur_y_l0_std = None
    cur_y_l1_std = None
    cur_y_l2_std = None
    cur_y_l_inf_std = None
    cur_att_suc_std = None

    cur_transfer_reader = csv.DictReader(cur_transfer_fp, dialect='excel-tab')
    cur_diff_reader = csv.DictReader(cur_diff_fp, dialect='excel-tab')

    for transfer_row in cur_transfer_reader:
        diff_row = next(cur_diff_reader)
        if transfer_row['adv_model'] != cur_name:
            cur_name = transfer_row['adv_model']
            x_name.append(pair['name'] + '_' + cur_name + '_' + transfer_row['test_model'])
            x_st_name.append(pair['name'])
            x_line_style.append(cur_line_style)
            x_color_style.append(cur_color)

            cur_x_transfer = []
            cur_x_transfer_std = []
            cur_x_pred = []
            cur_x_pred_std = []
            cur_x_confidence = []

            cur_y_l0_mean = []
            cur_y_l1_mean = []
            cur_y_l2_mean = []
            cur_y_l_inf_mean = []
            cur_att_suc_mean = []

            cur_y_l0_std = []
            cur_y_l1_std = []
            cur_y_l2_std = []
            cur_y_l_inf_std = []
            cur_att_suc_std = []

            x_transfer.append(cur_x_transfer)
            x_transfer_std.append(cur_x_transfer_std)
            x_pred.append(cur_x_pred)
            x_pred_std.append(cur_x_pred_std)
            x_confidence.append(cur_x_confidence)

            y_l0_mean.append(cur_y_l0_mean)
            y_l1_mean.append(cur_y_l1_mean)
            y_l2_mean.append(cur_y_l2_mean)
            y_l_inf_mean.append(cur_y_l_inf_mean)
            att_suc_mean.append(cur_att_suc_mean)

            y_l0_std.append(cur_y_l0_std)
            y_l1_std.append(cur_y_l1_std)
            y_l2_std.append(cur_y_l2_std)
            y_l_inf_std.append(cur_y_l_inf_std)
            att_suc_std.append(cur_att_suc_std)

        if float(diff_row['confidence']) not in conf_list:
            continue
        cur_x_transfer.append(float(transfer_row['transfer_rate_mean']))
        cur_x_transfer_std.append(float(transfer_row['transfer_rate_std']))
        cur_x_pred.append(float(transfer_row['pred_acc_mean']))
        cur_x_pred_std.append(float(transfer_row['pred_acc_std']))
        cur_x_confidence.append(float(diff_row['confidence']))

        cur_y_l0_mean.append(float(diff_row['l0_mean']))
        cur_y_l0_std.append(float(diff_row['l0_std']))
        cur_y_l1_mean.append(float(diff_row['l1_mean']))
        cur_y_l1_std.append(float(diff_row['l1_std']))
        cur_y_l2_mean.append(float(diff_row['l2_mean']))
        cur_y_l2_std.append(float(diff_row['l2_std']))
        cur_y_l_inf_mean.append(float(diff_row['l_inf_mean']))
        cur_y_l_inf_std.append(float(diff_row['l_inf_std']))

        cur_att_suc_mean.append(float(transfer_row['attack_success']))
        cur_att_suc_std.append(float(transfer_row['attack_success_std']))

        # write to csv file
        output_csv.write(("%s\t%d\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %
                          (cur_name, float(transfer_row['confidence']), transfer_row['test_model'],
                           cur_x_transfer[-1], cur_x_transfer_std[-1],
                           cur_x_pred[-1], cur_x_pred_std[-1],
                           cur_y_l0_mean[-1], cur_y_l0_std[-1],
                           cur_y_l1_mean[-1], cur_y_l1_std[-1],
                           cur_y_l2_mean[-1], cur_y_l2_std[-1],
                           cur_y_l_inf_mean[-1], cur_y_l_inf_std[-1],
                           cur_att_suc_mean[-1], cur_att_suc_std[-1])).encode())

    cur_transfer_fp.close()
    cur_diff_fp.close()

output_csv.close()

# plot figure


def init_plot(ax, x_label, y_label):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

width = args.width
height = args.height
tran_low = args.tran_low
tran_high = args.tran_high
att_low = args.att_low
att_high = args.att_high
l2_low = args.l2_low
l2_high = args.l2_high

fg0 = plt.figure(figsize=(width, height))
fg2 = plt.figure(figsize=(width, height))
fgInf = plt.figure(figsize=(width, height))

conf_fg0 = plt.figure(figsize=(width, height))
conf_fg2 = plt.figure(figsize=(width, height))
conf_fgInf = plt.figure(figsize=(width, height))

att_fg0 = plt.figure(figsize=(width, height))
att_fg2 = plt.figure(figsize=(width, height))
att_fgInf = plt.figure(figsize=(width, height))

ax0 = fg0.add_subplot(111)
ax2 = fg2.add_subplot(111)
axInf = fgInf.add_subplot(111)

conf_ax0 = conf_fg0.add_subplot(111)
conf_ax2 = conf_fg2.add_subplot(111)
conf_axInf = conf_fgInf.add_subplot(111)

att_ax0 = att_fg0.add_subplot(111)
att_ax2 = att_fg2.add_subplot(111)
att_axInf = att_fgInf.add_subplot(111)

init_plot(ax0, 'Transferability', r'$L_0$')
init_plot(ax2, 'Transferability', r'$L_2$')
init_plot(axInf, 'Transferability', r'$L_{\infty}$')

ax0.set_xlim(tran_low, tran_high)
ax2.set_xlim(tran_low, tran_high)
axInf.set_xlim(tran_low, tran_high)

att_ax0.set_ylim(att_low, att_high)
att_ax2.set_ylim(att_low, att_high)
att_axInf.set_ylim(att_low, att_high)

conf_list = [float(_x) for _x in conf_list]
conf_ax2.set_xticks(conf_list)
conf_ax2.set_ylim(l2_low, l2_high)
att_ax2.set_xticks(conf_list)

init_plot(conf_ax0, 'Confidence', r'$L_0$')
init_plot(conf_ax2, 'Confidence', r'$L_2$')
init_plot(conf_axInf, 'Confidence', r'$L_{\infty}$')

init_plot(att_ax0, 'Confidence', 'Attack Success')
init_plot(att_ax2, 'Confidence', 'Attack Success')
init_plot(att_axInf, 'Confidence', 'Attack Success')

err_bar_style = {
    'markersize': args.marker_size,
    'markeredgewidth': 0.9,
    'alpha': 0.7,
    'capsize': 3,
    'capthick': 0.9,
    'elinewidth': 0.9,
    'linewidth': 0.9
}

bar_style = {
    'alpha': 0.7,
    'capsize': 3,
    'linewidth': 0.9
}

for i in range(len(x_name)):
    cur_x_name = x_name[i]
    cur_x_st_name = x_st_name[i]
    cur_x_tranfer = x_transfer[i]
    cur_x_tranfer_std = x_transfer_std[i]
    cur_x_pred = x_pred[i]
    cur_x_pred_std = x_pred_std[i]
    cur_x_confidence = np.array(x_confidence[i]) + i * args.shift_space
    cur_fmt = x_color_style[i] + x_line_style[i]

    if 'l0' in cur_x_name:
        ax0.errorbar(cur_x_tranfer, y_l0_mean[i], xerr=cur_x_tranfer_std, yerr=y_l0_std[i], fmt=cur_fmt,
                     label=cur_x_st_name, **err_bar_style)
        conf_ax0.errorbar(cur_x_confidence, y_l0_mean[i], yerr=y_l0_std[i], fmt=cur_fmt,
                          label=cur_x_st_name, **err_bar_style)
        att_ax0.errorbar(cur_x_confidence, att_suc_mean[i], yerr=att_suc_std[i], fmt=cur_fmt,
                         label=cur_x_st_name, **err_bar_style)

    if 'l2' in cur_x_name:
        ax2.errorbar(cur_x_tranfer, y_l2_mean[i], xerr=cur_x_tranfer_std, yerr=y_l2_std[i], fmt=cur_fmt,
                     label=cur_x_st_name, **err_bar_style)
        conf_ax2.errorbar(cur_x_confidence, y_l2_mean[i], yerr=y_l2_std[i], fmt=cur_fmt,
                          label=cur_x_st_name, **err_bar_style)
        att_ax2.errorbar(cur_x_confidence, att_suc_mean[i], yerr=att_suc_std[i], fmt=cur_fmt,
                         label=cur_x_st_name, **err_bar_style)

    if 'li' in cur_x_name:
        axInf.errorbar(cur_x_tranfer, y_l_inf_mean[i], xerr=cur_x_tranfer_std, yerr=y_l_inf_std[i], fmt=cur_fmt,
                       label=cur_x_st_name, **err_bar_style)
        conf_axInf.errorbar(cur_x_confidence, y_l_inf_mean[i], yerr=y_l_inf_std[i], fmt=cur_fmt,
                            label=cur_x_st_name, **err_bar_style)
        att_axInf.errorbar(cur_x_confidence, att_suc_mean[i], yerr=att_suc_std[i], fmt=cur_fmt,
                           label=cur_x_st_name, **err_bar_style)


def get_legend_handles_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    return handles, labels


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


l0_handles, l0_labels = get_legend_handles_labels(ax0)
l2_handles, l2_labels = get_legend_handles_labels(ax2)
l_inf_handles, l_inf_labels = get_legend_handles_labels(axInf)

conf_l0_handles, conf_l0_labels = get_legend_handles_labels(conf_ax0)
conf_l2_handles, conf_l2_labels = get_legend_handles_labels(conf_ax2)
conf_l_inf_handles, conf_l_inf_labels = get_legend_handles_labels(conf_axInf)

att_l0_handles, att_l0_labels = get_legend_handles_labels(att_ax0)
att_l2_handles, att_l2_labels = get_legend_handles_labels(att_ax2)
att_l_inf_handles, att_l_inf_labels = get_legend_handles_labels(att_axInf)

legend_style = {
    'ncol': args.legend_col,
    'loc': 'upper left',
    'borderaxespad': 0.5,
    'framealpha': 0.5
}

fg0.tight_layout(pad=0)
fg2.tight_layout(pad=0)
fgInf.tight_layout(pad=0)

conf_fg0.tight_layout(pad=0)
conf_fg2.tight_layout(pad=0)
conf_fgInf.tight_layout(pad=0)

att_fg0.tight_layout(pad=0)
att_fg2.tight_layout(pad=0)
att_fgInf.tight_layout(pad=0)

output_img0_filename = os.path.join(_dir, output_file + "_plot_l0.pdf")
output_img2_filename = os.path.join(_dir, output_file + "_plot_l2.pdf")
output_imgInf_filename = os.path.join(_dir, output_file + "_plot_linf.pdf")

conf_img0_filename = os.path.join(_dir, output_file + "_conf_l0.pdf")
conf_img2_filename = os.path.join(_dir, output_file + "_conf_l2.pdf")
conf_imgInf_filename = os.path.join(_dir, output_file + "_conf_linf.pdf")

att_img0_filename = os.path.join(_dir, output_file + "_att_l0.pdf")
att_img2_filename = os.path.join(_dir, output_file + "_att_l2.pdf")
att_imgInf_filename = os.path.join(_dir, output_file + "_att_linf.pdf")

fg0.savefig(output_img0_filename)
fg2.savefig(output_img2_filename)
fgInf.savefig(output_imgInf_filename)

conf_fg0.savefig(conf_img0_filename)
conf_fg2.savefig(conf_img2_filename)
conf_fgInf.savefig(conf_imgInf_filename)

att_fg0.savefig(att_img0_filename)
att_fg2.savefig(att_img2_filename)
att_fgInf.savefig(att_imgInf_filename)

lg_fg = plt.figure(figsize=(width, height))
lg_ax = lg_fg.add_subplot(111)
lg_ax.set_axis_off()
tmp_lg = lg_ax.legend(handles=l2_handles, labels=l2_labels, **legend_style)
export_legend(tmp_lg, os.path.join(_dir, output_file + '_lg_l2.pdf'))
