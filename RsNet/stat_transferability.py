import argparse
import os
import csv
import numpy as np
import json
import itertools

parser = argparse.ArgumentParser(description='stat transferability for models.')

parser.add_argument('--dirs', help='data dir, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('--ex_int', help='exclusive integer, required', type=str, default="-1")
parser.add_argument('--is_combine', help='whether combine the transferability of two adv data set',
                    type=str, default="yes")
parser.add_argument('--output_file', help='save filename', type=str, default='summary.csv')
parser.add_argument('--dir_cycle', help='model_dir cycle', type=str, default=None)
parser.add_argument('--validate_num', help='number of target model used for validation', type=int, default=0)
parser.add_argument('--validate_pass', help='lower bound of numder of target model should be passed', type=int, default=0)

args = parser.parse_args()

validate_num = args.validate_num
validate_pass = args.validate_pass

_dirs = args.dirs.split(",")
ex_row_idx = [int(__) for __ in args.ex_int.split(",")] if args.ex_int != '' else []
ex_row_idx_len = len(ex_row_idx)
output_filename = args.output_file
is_combine = args.is_combine == 'yes'
dir_cycle = args.dir_cycle.split(',') if args.dir_cycle is not None else [str(_x) for _x in np.arange(len(_dirs))]

configs = []
model_dict = {}

for cur_dir in _dirs:
    config_fp = open(os.path.join(cur_dir, args.config), "rb")
    json_str = config_fp.read()
    config_fp.close()
    configs.append(json.loads(json_str.decode()))

# assume two dir has same structure
config = configs[0]
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
z_predict_topk = []
z_det = []
z_attack_success = []
z_attack_count = []

for i in range(category):
    x.append([])
    z_transfer.append([])
    z_predict.append([])
    z_predict_topk.append([])
    z_det.append([])
    z_attack_success.append([])
    z_attack_count.append([])

idx = 0
num_model = 0
output_file = open(os.path.join(_dirs[0], output_filename), 'wb')
output_file.write("adv_model\tconfidence\ttest_model\ttransfer_rate_mean\ttransfer_rate_std"
                  "\tpred_acc_mean\tpred_acc_std\tpred_acc_topk_mean\tpred_acc_tok_std"
                  "\tdet_mean\tdet_std\tattack_success\tattack_success_std\n".encode())

if output_filename.endswith(".csv"):
    output_raw_name = output_filename[:-4] + "-raw.csv"
    output_passed_name = output_filename[:-4] + "-passed.csv"
else:
    output_raw_name = output_filename + "-raw.csv"
    output_passed_name = output_filename + "-pass.csv"
output_raw = open(os.path.join(_dirs[0], output_raw_name), 'wb')
output_raw.write("model_id\tidx\tl0\tl1\tl2\tl_inf\tfloat_pred\tfloor_pred\tceil_pred\tround_pred\tfloat_tran\t"
                 "floor_tran\tceil_trans\tround_trans\tfloat_det\tfloor_det\tceil_det\tround_det\t"
                 "float_pred_topk\tfloor_pred_topk\tceil_pred_topk\tround_pred_topk\tpassed\n".encode())
output_passed = open(os.path.join(_dirs[0], output_passed_name), 'wb')
output_passed.write("model_id\tidx\tl0\tl1\tl2\tl_inf\tfloat_pred\tfloor_pred\tceil_pred\tround_pred\tfloat_tran\t"
                    "floor_tran\tceil_trans\tround_trans\tfloat_det\tfloor_det\tceil_det\tround_det\t"
                    "float_pred_topk\tfloor_pred_topk\tceil_pred_topk\tround_pred_topk\n".encode())


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
            raw_filename = filename[:-4] + "-raw.csv"

            tmp_data = {}

            for dir_i in range(len(_dirs)):
                cur_csvfile = open(os.path.join(_dirs[dir_i], filename), 'r')
                cur_reader = csv.DictReader(cur_csvfile, dialect='excel-tab')

                t = 0
                cur_ex_row_idx = ex_row_idx[dir_i] if dir_i < ex_row_idx_len else -1
                for row in cur_reader:
                    if is_combine and t in ex_row_idx:
                        t += 1
                        continue
                    elif t == cur_ex_row_idx:
                        t += 1
                        continue
                    else:
                        t += 1

                    tmp_name = row['name']
                    tmp_att_stat = [int(__x) for __x in row['attack_success_count'].split('/')]
                    tmp_adv_count = round(float(row['round_adv_acc']) * tmp_att_stat[0])
                    tmp_pred_count = round(float(row['round_pred_acc']) * tmp_att_stat[0])
                    # handle detection rate, old version does not have detection rate data,
                    # for non detector model, it does not matter.
                    if 'round_det_acc' in row:
                        tmp_det_count = round(float(row['round_det_acc']) * tmp_att_stat[0])
                    else:
                        tmp_det_count = 0

                    # handle topk prediction rate, old version does not have topk prediction rate data,
                    # we use top 1 data instead. Also for dataset other than imagenet such as mnist,
                    # fashion, and cifar, we only report top 1 accuracy.
                    if 'round_pred_acc_topk' in row:
                        tmp_pred_topk_count = round(float(row['round_pred_acc_topk']) * tmp_att_stat[0])
                    else:
                        tmp_pred_topk_count = tmp_pred_count

                    if 'magnet_highthrs' in _dirs[dir_i]:
                        tmp_name += '_magnet'
                    elif 'transfer_rc' in _dirs[dir_i]:
                        tmp_name += '_rc'

                    if is_combine:
                        if tmp_name not in tmp_data:
                            tmp_data[tmp_name] = {
                                'k': float(k),
                                'adv': tmp_adv_count,
                                'pred': tmp_pred_count,
                                'pred_topk': tmp_pred_topk_count,
                                'det': tmp_det_count,
                                'att_suc': tmp_att_stat[0],
                                'att_tot': tmp_att_stat[1]
                            }
                        else:
                            cur_tmp_data = tmp_data[tmp_name]
                            cur_tmp_data['adv'] += tmp_adv_count
                            cur_tmp_data['pred'] += tmp_pred_count
                            cur_tmp_data['pred_topk'] += tmp_pred_topk_count
                            cur_tmp_data['det'] += tmp_det_count
                            cur_tmp_data['att_suc'] += tmp_att_stat[0]
                            cur_tmp_data['att_tot'] += tmp_att_stat[1]
                    else:
                        x[idx].append(float(k))
                        z_transfer[idx].append(tmp_adv_count/tmp_att_stat[0])
                        z_predict[idx].append(tmp_pred_count/tmp_att_stat[0])
                        z_predict_topk[idx].append(tmp_pred_topk_count/tmp_att_stat[0])
                        z_det[idx].append(tmp_det_count/tmp_att_stat[0])

                    z_attack_success[idx].append(tmp_att_stat[0]/tmp_att_stat[1])
                cur_csvfile.close()

                cur_raw_csvfile = open(os.path.join(_dirs[dir_i], raw_filename), 'r')
                cur_raw_reader = csv.reader(cur_raw_csvfile, dialect='excel-tab')
                next(cur_raw_reader, None)

                cur_ex_row_idx = str(cur_ex_row_idx)
                str_ex_row_idx = [str(__x) for __x in ex_row_idx]
                model_prefix = dir_cycle[dir_i]
                tmp_trans_data = {}
                tmp_mid_set = []

                for row in cur_raw_reader:
                    if is_combine and row[0] in str_ex_row_idx:
                        continue
                    elif (not is_combine) and row[0] == cur_ex_row_idx:
                        continue

                    tmp_model_id = row[0]
                    if tmp_model_id not in tmp_trans_data:
                        tmp_trans_data[tmp_model_id] = []
                        tmp_mid_set.append(tmp_model_id)
                    if len(row) < 15:
                        row.extend(['0', '0', '0', '0'])
                    if len(row) < 19:
                        row.extend(row[6:10])
                    tmp_trans_data[tmp_model_id].append(row)

                    # output_raw.write((model_prefix + '_' + '\t'.join(row) + '\n').encode())
                cur_raw_csvfile.close()

                if validate_num == 0:
                    for mid in tmp_mid_set:
                        for row in tmp_trans_data[mid]:
                            output_raw.write((model_prefix + '_' + '\t'.join(row) + '\t1\n').encode())
                            output_passed.write((model_prefix + '_' + '\t'.join(row) + '\n').encode())
                else:
                    for cur_mid_set in itertools.combinations(tmp_mid_set, validate_num):
                        list_pass = []
                        for row_i in range(len(tmp_trans_data[cur_mid_set[0]])):
                            tmp_pass_count = 0
                            for cur_mid in cur_mid_set:
                                if tmp_trans_data[cur_mid][row_i][13] == '1':
                                    tmp_pass_count += 1
                            if tmp_pass_count >= validate_pass:
                                list_pass.append(1)
                            else:
                                list_pass.append(0)

                        for cur_mid in tmp_mid_set:
                            # skip the data in validation dataset
                            if cur_mid in cur_mid_set:
                                continue
                            row_i = 0
                            for row in tmp_trans_data[cur_mid]:
                                output_raw.write((model_prefix + '_' + '\t'.join(row) + '\t' +
                                                  str(list_pass[row_i]) + '\n').encode())
                                if list_pass[row_i] == 1:
                                    output_passed.write((model_prefix + '_' + '\t'.join(row) + '\n').encode())
                                row_i += 1

            # stat transferability in combine condition
            if is_combine:
                for cur_tmp_data_name in tmp_data:
                    cur_tmp_data = tmp_data[cur_tmp_data_name]
                    x[idx].append(cur_tmp_data['k'])
                    z_transfer[idx].append(cur_tmp_data['adv'] / cur_tmp_data['att_suc'])
                    z_predict[idx].append(cur_tmp_data['pred'] / cur_tmp_data['att_suc'])
                    z_predict_topk[idx].append(cur_tmp_data['pred_topk'] / cur_tmp_data['att_suc'])
                    z_det[idx].append(cur_tmp_data['det'] / cur_tmp_data['att_suc'])

            cur_adv_confidence = float(cur_adv_confidence)
            output_file.write(("%s\t%.4f\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %
                               (cur_adv_model_name, cur_adv_confidence, cur_test_model_name,
                                np.mean(z_transfer[idx]), np.std(z_transfer[idx]),
                                np.mean(z_predict[idx]), np.std(z_predict[idx]),
                                np.mean(z_predict_topk[idx]), np.std(z_predict_topk[idx]),
                                np.mean(z_det[idx]), np.std(z_det[idx]),
                                np.mean(z_attack_success[idx]), np.std(z_attack_success[idx]))).encode())
        idx += 1

output_file.close()
output_raw.close()
output_passed.close()
