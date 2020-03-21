#!/usr/bin/python3
import os
import json
import csv
import argparse
import numpy as np
import re


parser = argparse.ArgumentParser(description='create table for adv_diff vs transferability cross methods.')

parser.add_argument('-d', '--dir', help='directory, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('-o', '--output', help='output name, default summary', type=str, default='summary_bin')
parser.add_argument('--suffix', help='dataset suffix', type=str, default='')

args = parser.parse_args()

_dir = args.dir
output_file = args.output

config_fp = open(os.path.join(_dir, args.config), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

# mkdir
if not os.path.exists(os.path.dirname(os.path.join(_dir, output_file))):
    os.makedirs(os.path.dirname(os.path.join(_dir, output_file)))

_bin = []
att = []

for mthd in config:
    cur_raw_trans_fp = open(os.path.join(_dir, mthd['transfer']), "r")
    cur_transfer_reader = csv.DictReader(cur_raw_trans_fp, dialect='excel-tab')

    cur_att = {
        "name": mthd['name']

    }
    cur_att_def = []
    cur_data = {}
    cur_data_std = {}
    cur_pred = {}
    cur_pred_std = {}
    for transfer_row in cur_transfer_reader:
        tmp_def_name = args.suffix + re.sub('[^A-Za-z]+', '', transfer_row['dataset'])
        tmp_bin = int(transfer_row['bin'])
        _bin.append(tmp_bin)
        if tmp_def_name not in cur_att_def:
            cur_att_def.append(tmp_def_name)
            cur_data[tmp_def_name] = {}
            cur_data_std[tmp_def_name] = {}
            cur_pred[tmp_def_name] = {}
            cur_pred_std[tmp_def_name] = {}
        cur_data[tmp_def_name][tmp_bin] = float(transfer_row['trans_rate_mean'])*100
        cur_data_std[tmp_def_name][tmp_bin] = float(transfer_row['trans_rate_std'])*100
        cur_pred[tmp_def_name][tmp_bin] = float(transfer_row['pred_rate_mean']) * 100
        cur_pred_std[tmp_def_name][tmp_bin] = float(transfer_row['pred_rate_std']) * 100

    cur_att['def'] = cur_att_def
    cur_att['data'] = cur_data
    cur_att['data_std'] = cur_data_std
    cur_att['pred'] = cur_pred
    cur_att['pred_std'] = cur_pred_std

    att.append(cur_att)

unique_bin = np.unique(_bin)

for cur_bin in _bin:
    cur_fp = open(os.path.join(_dir, output_file + str(cur_bin) + '.csv'), "wb")
    cur_pred_fp = open(os.path.join(_dir, output_file + "_pred" + str(cur_bin) + '.csv'), "wb")

    cur_fp.write('att'.encode())
    cur_pred_fp.write('att'.encode())
    for cur_def_name in att[0]['def']:
        cur_fp.write(('|' + cur_def_name + '|' + cur_def_name + 'std').encode())
        cur_pred_fp.write(('|' + cur_def_name + '|' + cur_def_name + 'std').encode())
    cur_fp.write('\n'.encode())
    cur_pred_fp.write('\n'.encode())

    for cur_att in att:
        skip = False
        for cur_def_name in cur_att['def']:
            if str(cur_att['data'][cur_def_name][cur_bin]) == 'nan':
                skip = True
                break
        if skip:
            continue
        cur_fp.write(cur_att['name'].encode())
        cur_pred_fp.write(cur_att['name'].encode())
        for cur_def_name in cur_att['def']:
            cur_fp.write(('|' + str(cur_att['data'][cur_def_name][cur_bin]) +
                          '|' + str(cur_att['data_std'][cur_def_name][cur_bin])).encode())
            cur_pred_fp.write(('|' + str(cur_att['pred'][cur_def_name][cur_bin]) +
                               '|' + str(cur_att['pred_std'][cur_def_name][cur_bin])).encode())
        cur_fp.write('\n'.encode())
        cur_pred_fp.write('\n'.encode())

    cur_fp.close()
    cur_pred_fp.close()
