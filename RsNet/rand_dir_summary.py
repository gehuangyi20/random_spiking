import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import json
import argparse
from itertools import cycle

import matplotlib
matplotlib.rcParams['text.usetex'] = True

parser = argparse.ArgumentParser(description='Summarize the result of random direction inference.')

parser.add_argument('-d', '--dir', help='directory, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('-o', '--output', help='output name, default summary', type=str, default='summary')
parser.add_argument('-H', '--height', help='height, default 5', type=float, default=5)
parser.add_argument('-W', '--width', help='width, default 5', type=float, default=5)
parser.add_argument('--y_low', help='y-axis lower bound', type=float, default=None)
parser.add_argument('--y_high', help='y-axis higher bound', type=float, default=None)
parser.add_argument('--noise_iter', help='add l2 times', type=int, default=1)
parser.add_argument('--legend_col', help='number of columns of legend', type=int, default=1)
parser.add_argument('--y_unchg', help='y unchg axis txt', type=str, default='Prediction Un-changed Rate')
parser.add_argument('--y_unchg_scale', help='y unchg scale', type=str, default='float')
parser.add_argument('--x_text', help='y unchg scale', type=str, default=r'$L_2$ distance')
parser.add_argument('--marker_size', help='marker size', type=float, default=8)
parser.add_argument('--markeredgewidth', help='marker edge width', type=float, default=0.9)
parser.add_argument('--capsize', help='capsize', type=float, default=3)
parser.add_argument('--ex_text', help='extra text position', type=str, default='')
parser.add_argument('--ex_text_x', help='extra text position', type=float, default=0)
parser.add_argument('--ex_text_y', help='extra text position', type=float, default=0)
parser.add_argument('--ex_text_marker', help='extra text marker', type=str, default='o')
parser.add_argument('--ex_text_color', help='extra text color', type=str, default='C1')
parser.add_argument('--reverse_x', help='whether to reverse the order of x axix', type=str, default='no')
parser.add_argument('--topk', help='topk acc', type=int, default=1)

args = parser.parse_args()

if not os.path.isdir(args.dir):
    print("Directory", args.dir, "does not exist")
    exit(0)

_dir = args.dir
output = args.output

config_fp = open(os.path.join(_dir, args.config), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

# open output stat summary csv
output_csv = open(os.path.join(_dir, output+".csv"), 'wb')
output_csv.write("name\tl2\ttest_acc\ttest_acc_std\ttest_acc_top_k\ttest_acc_top_k_std"
                 "\ttest_unchg\test_unchg_std\ttest_loss\ttest_loss_std\n".encode())

model_name = config['model_name']
model_name_st = config['model_name_st']
l2 = config['l2']

x = []
acc = []
acc_std = []
acc_topk = []
acc_topk_std = []
unchg = []
unchg_std = []
loss = []
loss_std = []
y_unchg_scale = 1 if args.y_unchg_scale == 'float' else 100


for i in range(len(model_name)):
    cur_model_name = model_name[i]
    cur_model_name_st = model_name_st[i]
    cur_x = []
    cur_acc = []
    cur_acc_std = []
    cur_acc_topk = []
    cur_acc_topk_std = []
    cur_unchg = []
    cur_unchg_std = []
    cur_loss = []
    cur_loss_std = []

    for j in range(len(l2[i])):
        cur_l2 = l2[i][j]

        filename = cur_model_name + '/' + str(cur_l2) + '.csv'
        cur_csvfile = open(os.path.join(_dir, filename), 'r')
        cur_reader = csv.DictReader(cur_csvfile, dialect='excel-tab')

        tmp_acc = []
        tmp_acc_topk = []
        tmp_unchg = []
        tmp_loss = []

        for row in cur_reader:
            tmp_acc.append(float(row['test_acc']))
            if 'test_acc_top_k' in row:
                tmp_acc_topk.append(float(row['test_acc_top_k']))
            else:
                tmp_acc_topk.append(float(row['test_acc']))
            tmp_unchg.append(float(row['test_unchg']))
            tmp_loss.append(float(row['test_loss']))

        cur_csvfile.close()

        # fix not divide the number of noise_iter when compute the loss in rand_dir_inference
        # since the experiment uses 20, default value is 20 here
        tmp_loss = np.array(tmp_loss) / args.noise_iter

        cur_l2 = float(cur_l2)
        cur_x.append(cur_l2)
        cur_acc.append(np.mean(tmp_acc))
        cur_acc_std.append(np.std(tmp_acc))
        cur_acc_topk.append(np.mean(tmp_acc_topk))
        cur_acc_topk_std.append(np.std(tmp_acc_topk))
        cur_unchg.append(np.mean(tmp_unchg))
        cur_unchg_std.append(np.std(tmp_unchg))
        cur_loss.append(np.mean(tmp_loss))
        cur_loss_std.append(np.std(tmp_loss))

        output_csv.write(("%s\t%.4f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %
                          (cur_model_name, cur_l2,
                           cur_acc[-1], cur_acc_std[-1], cur_acc_topk[-1], cur_acc_topk_std[-1],
                           cur_unchg[-1], cur_unchg_std[-1],
                           cur_loss[-1], cur_loss_std[-1])).encode())

    x.append(cur_x)
    acc.append(cur_acc)
    acc_std.append(cur_acc_std)
    acc_topk.append(cur_acc_topk)
    acc_topk_std.append(cur_acc_topk_std)
    unchg.append(cur_unchg)
    unchg_std.append(cur_unchg_std)
    loss.append(cur_loss)
    loss_std.append(cur_loss_std)

output_csv.close()

if args.reverse_x == 'yes':
    x = np.array(x) * -1 + np.max(l2)
# plot figure
width = args.width
height = args.height
y_low = args.y_low
y_high = args.y_high
fg_acc = plt.figure(figsize=(width, height))
fg_acc_topk = plt.figure(figsize=(width, height))
fg_unchg = plt.figure(figsize=(width, height))
fg_loss = plt.figure(figsize=(width, height))

l2 = np.asarray(l2, dtype=np.float32)

ax_acc = fg_acc.add_subplot(111)
ax_acc.set_xlabel(args.x_text)
ax_acc.set_ylabel('Prediction Accurary')
ax_acc.set_xbound(0, np.max(l2))
ax_acc.set_ylim(y_low, y_high)

ax_acc_topk = fg_acc_topk.add_subplot(111)
ax_acc_topk.set_xlabel(args.x_text)
ax_acc_topk.set_ylabel('Prediction Accurary Top %d' % args.topk)
ax_acc_topk.set_xbound(0, np.max(l2))
ax_acc_topk.set_ylim(y_low, y_high)

ax_unchg = fg_unchg.add_subplot(111)
ax_unchg.set_xlabel(args.x_text)
ax_unchg.set_ylabel(args.y_unchg)
ax_unchg.set_xbound(0, np.max(l2))
ax_unchg.set_ylim(y_low * y_unchg_scale, y_high * y_unchg_scale)

ax_loss = fg_loss.add_subplot(111)
ax_loss.set_xlabel(args.x_text)
ax_loss.set_ylabel('Loss')
ax_loss.set_xbound(0, np.max(l2))

if "xticks" in config:
    ax_acc.set_xticks(config["xticks"])
    ax_acc_topk.set_xticks(config["xticks"])
    ax_unchg.set_xticks(config["xticks"])
    ax_loss.set_xticks(config["xticks"])

if "xticklabels" in config:
    ax_acc.set_xticklabels(config["xticklabels"])
    ax_acc_topk.set_xticklabels(config["xticklabels"])
    ax_unchg.set_xticklabels(config["xticklabels"])
    ax_loss.set_xticklabels(config["xticklabels"])

line_style = config["line"] if "line" in config else ['o', '>', 'D', '*', 'p', 'P', 's', 'X', '^', 'v']
color_style = config["color"] if "color" in config else ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
line_style_cycle = cycle(line_style)
color_style_cycle = cycle(color_style)

lg_acc = []
shift_space = (x[0][1] - x[0][0]) / (len(model_name) + 3)
err_bar_style = {
    'markersize': args.marker_size,
    'markeredgewidth': args.markeredgewidth,
    'alpha': 0.7,
    'capsize': args.capsize,
    'capthick': args.markeredgewidth,
    'elinewidth': args.markeredgewidth,
    'linewidth': args.markeredgewidth
}


for i in range(len(model_name)):
    cur_model_name_st = model_name_st[i]
    cur_x = np.array(x[i]) + shift_space * i
    cur_acc = acc[i]
    cur_acc_std = acc_std[i]
    cur_acc_topk = acc_topk[i]
    cur_acc_topk_std = acc_topk_std[i]
    cur_unchg = np.array(unchg[i]) * y_unchg_scale
    cur_unchg_std = np.array(unchg_std[i]) * y_unchg_scale
    cur_loss = loss[i]
    cur_loss_std = loss_std[i]

    cur_style = next(line_style_cycle) + next(color_style_cycle)
    ax_acc.errorbar(cur_x, cur_acc, yerr=cur_acc_std, fmt=cur_style,
                    label=cur_model_name_st, **err_bar_style)

    ax_acc_topk.errorbar(cur_x, cur_acc_topk, yerr=cur_acc_topk_std, fmt=cur_style,
                         label=cur_model_name_st, **err_bar_style)

    ax_unchg.errorbar(cur_x, cur_unchg, yerr=cur_unchg_std, fmt=cur_style,
                      label=cur_model_name_st, **err_bar_style)

    ax_loss.errorbar(cur_x, cur_loss, yerr=cur_loss_std, fmt=cur_style,
                     label=cur_model_name_st, **err_bar_style)

if args.ex_text != '':
    ax_acc.plot(args.ex_text_x + 0.03 * np.max(l2), args.ex_text_y, args.ex_text_marker + args.ex_text_color,
                alpha=0.7, markeredgewidth=args.markeredgewidth)
    ax_acc.text(args.ex_text_x, args.ex_text_y, args.ex_text, fontsize=8, verticalalignment='center',
                bbox=dict(boxstyle="round", ec=args.ex_text_color, fc=(1, 1, 1, 0)))
    ax_unchg.plot(args.ex_text_x - 0.045 * np.max(l2), args.ex_text_y * y_unchg_scale, args.ex_text_marker + args.ex_text_color,
                  alpha=0.7, markeredgewidth=args.markeredgewidth)
    ax_unchg.text(args.ex_text_x, args.ex_text_y * y_unchg_scale, args.ex_text, fontsize=8, verticalalignment='center',
                  bbox=dict(boxstyle="round,pad=0.4", ec=args.ex_text_color, fc=(1, 1, 1, 0)))


def get_legend_handles_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    return handles, labels


legend_style = {
        'ncol': args.legend_col,
        'loc': 'best',
        'borderaxespad': 0.5,
        'framealpha': 0.5
    }


acc_handles, acc_labels = get_legend_handles_labels(ax_acc)
acc_topk_handles, acc_topk_labels = get_legend_handles_labels(ax_acc_topk)
unchg_handles, unchg_labels = get_legend_handles_labels(ax_unchg)
loss_handles, loss_labels = get_legend_handles_labels(ax_loss)

#ax_acc.legend(handles=acc_handles, labels=acc_labels, **legend_style)
#ax_unchg.legend(handles=unchg_handles, labels=unchg_labels, **legend_style)
#ax_loss.legend(handles=acc_handles, labels=acc_labels, **legend_style)

fg_acc.tight_layout(pad=0)
fg_acc_topk.tight_layout(pad=0)
fg_unchg.tight_layout(pad=0)
fg_loss.tight_layout(pad=0)

output_acc_filename = os.path.join(_dir, output + "_plot_acc.pdf")
output_acc_topk_filename = os.path.join(_dir, output + "_plot_acc_topk.pdf")
output_unchg_filename = os.path.join(_dir, output + "_plot_unchg.pdf")
output_loss_filename = os.path.join(_dir, output + "_plot_loss.pdf")

fg_acc.savefig(output_acc_filename)
fg_acc_topk.savefig(output_acc_topk_filename)
fg_unchg.savefig(output_unchg_filename)
fg_loss.savefig(output_loss_filename)

# save lg figure
lg_fg = plt.figure(figsize=(width, height))
lg_ax = lg_fg.add_subplot(111)
lg_ax.set_axis_off()
tmp_lg = lg_ax.legend(handles=acc_handles, labels=acc_labels, **legend_style)


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


output_lg_filename = os.path.join(_dir, output + "_lg.pdf")
export_legend(tmp_lg, output_lg_filename)
