import os
import sys
import matplotlib.pyplot as plt
import csv
import numpy as np
import json
import math

argc = len(sys.argv)
if argc < 2:
    print('usage: plot_transferability [dir] [filename]')
    sys.exit()

_dir = sys.argv[1]
img_filename = sys.argv[2] if argc >= 3 else 'summary.png'

config_fp = open(os.path.join(_dir, "config.json"), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

adv_model_name = config['adv_model_name']
adv_model_name_st = config['adv_model_name_st']
adv_confidence = config['adv_confidence']

test_model_name = config['test_model_name']
test_model_name_st = config['test_model_name_st']

adv_model_len = len(adv_model_name)

category = 0
for cur_adv_confidence in adv_confidence:
    category += len(cur_adv_confidence)

x = []
z_transfer = []
z_predict = []
z_attack_success = []

for i in range(category):
    x.append([])
    z_transfer.append([])
    z_predict.append([])
    z_attack_success.append([])

idx = 0
num_model = 0
for i in range(len(adv_model_name)):
    cur_adv_model_name = adv_model_name[i]
    cur_adv_model_name_st = adv_model_name_st[i]
    for j in range(len(adv_confidence[i])):
        cur_adv_confidence = adv_confidence[i][j]
        num_model += 1

        for k in range(len(test_model_name)):
            cur_test_model_name = test_model_name[k]
            cur_test_model_name_st = test_model_name_st[k]

            filename = cur_test_model_name + '_transferability_' + cur_adv_model_name + '_' + \
                       str(cur_adv_confidence) + '.csv'
            cur_csvfile = open(os.path.join(_dir, filename), 'r')
            cur_reader = csv.DictReader(cur_csvfile, dialect='excel-tab')

            t = 0
            for row in cur_reader:
                t += 1
                x[idx].append(float(k))
                z_transfer[idx].append(float(row['float_adv_acc']))
                z_predict[idx].append(float(row['float_pred_acc']))
                z_attack_success[idx].append(float(row['attack_success_rate']))
            cur_csvfile.close()

        idx += 1


fig = plt.figure(figsize=(15, 5))

ax0 = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2)
ax2 = fig.add_subplot(1, 3, 3)

ax0.set_xlabel("adv_model")
ax0.set_ylabel("transfer")
ax0.set_title('Adv Transferability')
ax0.set_xticks(np.arange(len(test_model_name_st)))
ax0.set_xticklabels(test_model_name_st)


ax1.set_xlabel("adv_model")
ax1.set_ylabel("prediction")
ax1.set_title('Adv Prediction Acc on Test Model')
ax1.set_xticks(np.arange(len(test_model_name_st)))
ax1.set_xticklabels(test_model_name_st)

ax2.set_xlabel("adv_model")
ax2.set_ylabel("attack_success")
ax2.set_title('Attack Success Rate on Target Model')
ax2.set_xticks(np.arange(len(test_model_name_st)))
ax2.set_xticklabels(test_model_name_st)

legend = []
idx = 0
for i in range(len(adv_model_name)):
    cur_adv_model_name = adv_model_name[i]
    cur_adv_model_name_st = adv_model_name_st[i]
    for j in range(len(adv_confidence[i])):
        cur_adv_confidence = adv_confidence[i][j]

        cur_x = np.asarray(x[idx])
        cur_x += idx / num_model
        cur_legend = ax0.scatter(cur_x, z_transfer[idx], label=cur_adv_model_name_st + "_" + str(cur_adv_confidence),
                                 s=20)
        legend.append(cur_legend)
        ax1.scatter(cur_x, z_predict[idx], s=20)
        ax2.scatter(cur_x, z_attack_success[idx], s=20)
        idx += 1

ax0.legend(handles=legend, bbox_to_anchor=(0., 1.05, 1., .102), ncol=3, loc=3)

fig.tight_layout()
fig.subplots_adjust(top=0.95-0.05*math.ceil(len(legend)/3), bottom=.1)
fig.suptitle(img_filename)

if img_filename:
    fig.savefig(os.path.join(_dir, img_filename))
