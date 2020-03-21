#!/usr/bin/python3
import os
import sys
import json
import csv
import matplotlib.pyplot as plt
from itertools import cycle

_dir = sys.argv[1]
output_file = sys.argv[2]
plot_all = False if len(sys.argv) < 4 else sys.argv[3] == 'true'

config_fp = open(os.path.join(_dir, "config.json"), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

# open output stat summary csv
output_csv = open(os.path.join(_dir, output_file + ".csv"), 'wb')
output_csv.write("adv_model\tconfidence\ttest_model\ttransfer_rate_mean\ttransfer_rate_std"
                 "\tpred_acc_mean\tpred_acc_std\tl0_mean\tl0_std\tl1_mean\tl1_std"
                 "\tl2_mean\tl2_std\tl_inf_mean\tl_inf_std\n".encode())

# plot data initialization
line_style_cycle = cycle(['o-', '>--', 'D:', 'p-', 'p--', 'h:'])
x_transfer = []
x_pred = []
x_name = []
x_line_style = []
x_color_style = []

y_l0_mean = []
y_l1_mean = []
y_l2_mean = []
y_l_inf_mean = []

y_l0_std = []
y_l1_std = []
y_l2_std = []
y_l_inf_std = []

for pair in config:
    cur_transfer_fp = open(os.path.join(_dir, pair['transfer']), "r")
    cur_diff_fp = open(os.path.join(_dir, pair['diff']), "r")

    cur_name = None
    cur_x_transfer = None
    cur_x_pred = None
    cur_line_style = next(line_style_cycle)

    cur_y_l0_mean = None
    cur_y_l1_mean = None
    cur_y_l2_mean = None
    cur_y_l_inf_mean = None

    cur_y_l0_std = None
    cur_y_l1_std = None
    cur_y_l2_std = None
    cur_y_l_inf_std = None

    cur_transfer_reader = csv.DictReader(cur_transfer_fp, dialect='excel-tab')
    cur_diff_reader = csv.DictReader(cur_diff_fp, dialect='excel-tab')

    for transfer_row in cur_transfer_reader:
        diff_row = next(cur_diff_reader)
        if transfer_row['adv_model'] != cur_name:
            cur_name = transfer_row['adv_model']
            x_name.append(pair['name'] + '_' + cur_name + '_' + transfer_row['test_model'])
            x_line_style.append(cur_line_style)
            if 'l0' in cur_name:
                cur_color = 'C0'
            elif 'l2' in cur_name:
                cur_color = 'C1'
            elif 'li' in cur_name:
                cur_color = 'C2'
            else:
                cur_color = 'C3'
            x_color_style.append(cur_color)

            cur_x_transfer = []
            cur_x_pred = []

            cur_y_l0_mean = []
            cur_y_l1_mean = []
            cur_y_l2_mean = []
            cur_y_l_inf_mean = []

            cur_y_l0_std = []
            cur_y_l1_std = []
            cur_y_l2_std = []
            cur_y_l_inf_std = []

            x_transfer.append(cur_x_transfer)
            x_pred.append(cur_x_pred)

            y_l0_mean.append(cur_y_l0_mean)
            y_l1_mean.append(cur_y_l1_mean)
            y_l2_mean.append(cur_y_l2_mean)
            y_l_inf_mean.append(cur_y_l_inf_mean)

            y_l0_std.append(cur_y_l0_std)
            y_l1_std.append(cur_y_l1_std)
            y_l2_std.append(cur_y_l2_std)
            y_l_inf_std.append(cur_y_l_inf_std)

        cur_x_transfer.append(float(transfer_row['transfer_rate_mean']))
        cur_x_pred.append(float(transfer_row['pred_acc_mean']))

        cur_y_l0_mean.append(float(diff_row['l0_mean']))
        cur_y_l0_std.append(float(diff_row['l0_std']))
        cur_y_l1_mean.append(float(diff_row['l1_mean']))
        cur_y_l1_std.append(float(diff_row['l1_std']))
        cur_y_l2_mean.append(float(diff_row['l2_mean']))
        cur_y_l2_std.append(float(diff_row['l2_std']))
        cur_y_l_inf_mean.append(float(diff_row['l_inf_mean']))
        cur_y_l_inf_std.append(float(diff_row['l_inf_std']))

        # write to csv file
        output_csv.write(("%s\t%d\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %
                          (cur_name, int(transfer_row['confidence']), transfer_row['test_model'],
                           cur_x_transfer[-1], float(transfer_row['transfer_rate_std']),
                           cur_x_pred[-1], float(transfer_row['pred_acc_std']),
                           cur_y_l0_mean[-1], cur_y_l0_std[-1],
                           cur_y_l1_mean[-1], cur_y_l1_std[-1],
                           cur_y_l2_mean[-1], cur_y_l2_std[-1],
                           cur_y_l_inf_mean[-1], cur_y_l_inf_std[-1])).encode())

    cur_transfer_fp.close()
    cur_diff_fp.close()

output_csv.close()

# plot figure


def init_plot(ax, x_label, y_label):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(y_label + ' distance vs ' + x_label)
    ax.set_xbound(0, 1)


fig = plt.figure(figsize=(20, 15))

ax_transfer0 = fig.add_subplot(2, 4, 1)
ax_transfer1 = fig.add_subplot(2, 4, 2)
ax_transfer2 = fig.add_subplot(2, 4, 3)
ax_transfer3 = fig.add_subplot(2, 4, 4)
ax_pred0 = fig.add_subplot(2, 4, 5)
ax_pred1 = fig.add_subplot(2, 4, 6)
ax_pred2 = fig.add_subplot(2, 4, 7)
ax_pred3 = fig.add_subplot(2, 4, 8)

init_plot(ax_transfer0, 'transferability', 'l0')
init_plot(ax_transfer1, 'transferability', 'l1')
init_plot(ax_transfer2, 'transferability', 'l2')
init_plot(ax_transfer3, 'transferability', 'l_inf')
init_plot(ax_pred0, 'prediction', 'l0')
init_plot(ax_pred1, 'prediction', 'l1')
init_plot(ax_pred2, 'prediction', 'l2')
init_plot(ax_pred3, 'prediction', 'l_inf')

legend_t0 = []
legend_t1 = []
legend_t2 = []
legend_t3 = []
legend_p0 = []
legend_p1 = []
legend_p2 = []
legend_p3 = []
for i in range(len(x_name)):
    cur_x_name = x_name[i]
    cur_x_tranfer = x_transfer[i]
    cur_x_pred = x_pred[i]
    cur_line_style = x_line_style[i]
    cur_color_style = x_color_style[i]

    cur_legend, = ax_transfer1.plot(cur_x_tranfer, y_l1_mean[i], cur_color_style + cur_line_style, label=cur_x_name,
                                    markersize=10, alpha=0.7)
    legend_t1.append(cur_legend)
    if 'l0' in cur_x_name or plot_all:
        ax_transfer0.plot(cur_x_tranfer, y_l0_mean[i], cur_color_style + cur_line_style, label=cur_x_name,
                          markersize=10, alpha=0.7)
        legend_t0.append(cur_legend)

    if 'l2' in cur_x_name or plot_all:
        ax_transfer2.plot(cur_x_tranfer, y_l2_mean[i], cur_color_style + cur_line_style, label=cur_x_name,
                          markersize=10, alpha=0.7)
        legend_t2.append(cur_legend)
    if 'li' in cur_x_name or plot_all:
        ax_transfer3.plot(cur_x_tranfer, y_l_inf_mean[i], cur_color_style + cur_line_style, label=cur_x_name,
                          markersize=10, alpha=0.7)
        legend_t3.append(cur_legend)

    if 'l0' in cur_x_name or plot_all:
        ax_pred0.plot(cur_x_pred, y_l0_mean[i], cur_color_style + cur_line_style, label=cur_x_name, markersize=10,
                      alpha=0.7)
        legend_p0.append(cur_legend)
    ax_pred1.plot(cur_x_pred, y_l1_mean[i], cur_color_style + cur_line_style, label=cur_x_name, markersize=10,
                  alpha=0.7)
    legend_p1.append(cur_legend)
    if 'l2' in cur_x_name or plot_all:
        ax_pred2.plot(cur_x_pred, y_l2_mean[i], cur_color_style + cur_line_style, label=cur_x_name, markersize=10,
                      alpha=0.7)
        legend_p2.append(cur_legend)
    if 'li' in cur_x_name or plot_all:
        ax_pred3.plot(cur_x_pred, y_l_inf_mean[i], cur_color_style + cur_line_style, label=cur_x_name, markersize=10,
                      alpha=0.7)
        legend_p3.append(cur_legend)

ax_transfer0.legend(handles=legend_t0, bbox_to_anchor=(0.5, -0.12), ncol=1, loc='upper center', borderaxespad=0)
ax_transfer1.legend(handles=legend_t1, bbox_to_anchor=(0.5, -0.12), ncol=1, loc='upper center', borderaxespad=0)
ax_transfer2.legend(handles=legend_t2, bbox_to_anchor=(0.5, -0.12), ncol=1, loc='upper center', borderaxespad=0)
ax_transfer3.legend(handles=legend_t3, bbox_to_anchor=(0.5, -0.12), ncol=1, loc='upper center', borderaxespad=0)
ax_pred0.legend(handles=legend_p0, bbox_to_anchor=(0.5, -0.12), ncol=1, loc='upper center', borderaxespad=0)
ax_pred1.legend(handles=legend_p1, bbox_to_anchor=(0.5, -0.12), ncol=1, loc='upper center', borderaxespad=0)
ax_pred2.legend(handles=legend_p2, bbox_to_anchor=(0.5, -0.12), ncol=1, loc='upper center', borderaxespad=0)
ax_pred3.legend(handles=legend_p3, bbox_to_anchor=(0.5, -0.12), ncol=1, loc='upper center', borderaxespad=0)
fig.tight_layout()

output_img_filename = os.path.join(_dir, output_file + "_plot.pdf")
fig.savefig(output_img_filename)
