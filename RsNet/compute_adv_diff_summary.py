#!/usr/bin/python3
import argparse
import os
import json
import csv
import subprocess
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='compute adv diff summary for particular attack.')

parser.add_argument('--dirs', help='data dir, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='list.json')
parser.add_argument('--output_file', help='save filename', type=str, default='summary')

args = parser.parse_args()

_dirs = args.dirs.split(",")
output_file = args.output_file

config_fp = open(os.path.join(_dirs[0], args.config), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

# init concatenated pdf command line
output_pdf_filename = os.path.join(_dirs[0], output_file+".pdf")
output_pdf_args = ["gs", "-sDEVICE=pdfwrite", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sOutputFile="+output_pdf_filename]

# open output stat summary csv
output_csv = open(os.path.join(_dirs[0], output_file+".csv"), 'wb')
output_csv.write("adv_model\tconfidence\tl0_mean\tl0_std\tl1_mean\tl1_std"
                 "\tl2_mean\tl2_std\tl_inf_mean\tl_inf_std\tvalid\n".encode())

# plot data initialization
x = []
x_name = []

y_l0_mean = []
y_l1_mean = []
y_l2_mean = []
y_l_inf_mean = []

y_l0_std = []
y_l1_std = []
y_l2_std = []
y_l_inf_std = []

for attack_set in config:
    cur_attack_set_name = attack_set['name']
    cur_attack_set_conf_list = attack_set['conf']

    x_name.append(cur_attack_set_name)
    x.append([float(_x) for _x in cur_attack_set_conf_list])

    cur_y_l0_mean = []
    cur_y_l1_mean = []
    cur_y_l2_mean = []
    cur_y_l_inf_mean = []

    cur_y_l0_std = []
    cur_y_l1_std = []
    cur_y_l2_std = []
    cur_y_l_inf_std = []

    for cur_conf in cur_attack_set_conf_list:
        cur_attack_name = cur_attack_set_name + '_' + str(cur_conf)

        tmp_l0 = []
        tmp_l1 = []
        tmp_l2 = []
        tmp_l_inf = []
        tmp_valid = []

        for dir_i in range(len(_dirs)):
            # concatenate command line file
            output_pdf_args.append(os.path.join(_dirs[dir_i], cur_attack_name, 'diff.pdf'))

            # open current attack summary file
            cur_fp = open(os.path.join(_dirs[dir_i], cur_attack_name, "raw.csv"), 'r')
            cur_reader = csv.DictReader(cur_fp, dialect='excel-tab')

            for row in cur_reader:
                tmp_l0.append(float(row['l0']))
                tmp_l1.append(float(row['l1']))
                tmp_l2.append(float(row['l2']))
                tmp_l_inf.append(float(row['l_inf']))

            cur_fp.close()

            cur_fp = open(os.path.join(_dirs[dir_i], cur_attack_name, "summary.csv"), 'r')
            cur_reader = csv.DictReader(cur_fp, dialect='excel-tab')
            row = next(cur_reader)
            tmp_valid.append(row['valid'] + '/' + row['all'])

        cur_y_l0_mean.append(np.mean(tmp_l0))
        cur_y_l1_mean.append(np.mean(tmp_l1))
        cur_y_l2_mean.append(np.mean(tmp_l2))
        cur_y_l_inf_mean.append(np.mean(tmp_l_inf))

        cur_y_l0_std.append(np.std(tmp_l0))
        cur_y_l1_std.append(np.std(tmp_l1))
        cur_y_l2_std.append(np.std(tmp_l2))
        cur_y_l_inf_std.append(np.std(tmp_l_inf))

        # write to csv file
        output_csv.write(("%s\t%.4f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\n" %
                         (cur_attack_set_name, float(cur_conf), cur_y_l0_mean[-1], cur_y_l0_std[-1],
                          cur_y_l1_mean[-1], cur_y_l1_std[-1],
                          cur_y_l2_mean[-1], cur_y_l2_std[-1],
                          cur_y_l_inf_mean[-1], cur_y_l_inf_std[-1], ','.join(tmp_valid))).encode())

    y_l0_mean.append(cur_y_l0_mean)
    y_l1_mean.append(cur_y_l1_mean)
    y_l2_mean.append(cur_y_l2_mean)
    y_l_inf_mean.append(cur_y_l_inf_mean)

    y_l0_std.append(cur_y_l0_std)
    y_l1_std.append(cur_y_l1_std)
    y_l2_std.append(cur_y_l2_std)
    y_l_inf_std.append(cur_y_l_inf_std)

output_csv.close()

# cat diff pdf files for each attack
subprocess.run(output_pdf_args, stdout=subprocess.PIPE)

# plot figure

fig = plt.figure(figsize=(20, 5))

ax0 = fig.add_subplot(1, 4, 1)
ax1 = fig.add_subplot(1, 4, 2)
ax2 = fig.add_subplot(1, 4, 3)
ax3 = fig.add_subplot(1, 4, 4)

ax0.set_xlabel("confidence")
ax0.set_ylabel("l0")
ax0.set_title('l0 distance vs confidence')
ax0.set_xticks(x[0])

ax1.set_xlabel("confidence")
ax1.set_ylabel("l1")
ax1.set_title('l1 distance vs confidence')
ax1.set_xticks(x[0])

ax2.set_xlabel("confidence")
ax2.set_ylabel("l2")
ax2.set_title('l2 distance vs confidence')
ax2.set_xticks(x[0])

ax3.set_xlabel("confidence")
ax3.set_ylabel("l_inf")
ax3.set_title('l_inf distance vs confidence')
ax3.set_xticks(x[0])

legend = []
for i in range(len(x_name)):
    cur_x_name = x_name[i]
    cur_conf_list = x[i]

    cur_legend, = ax0.plot(cur_conf_list, y_l0_mean[i], 'o-', label=cur_x_name)
    legend.append(cur_legend)

    ax1.plot(cur_conf_list, y_l1_mean[i], 'o-')
    ax2.plot(cur_conf_list, y_l2_mean[i], 'o-')
    ax3.plot(cur_conf_list, y_l_inf_mean[i], 'o-')

ax3.legend(handles=legend, bbox_to_anchor=(1.05, 1), ncol=1, loc='upper left', borderaxespad=0)
fig.tight_layout()

output_img_filename = os.path.join(_dirs[0], output_file+"_plot.pdf")
fig.savefig(output_img_filename)
