#!/usr/bin/python3
import argparse
import os
import utils
import subprocess
from multiprocessing import Pool


parser = argparse.ArgumentParser(description='compute adv transfer cross multiple methods all in one')

parser.add_argument('--dir', help='data dir, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('-j', help='number of parallel process', type=int, default=1)

args = parser.parse_args()

_dir = args.dir
_threads = args.j

config = utils.load_json(os.path.join(_dir, args.config))
dir_diff_vs_trans = config['dir_diff_vs_trans']
dir_trans_all = config['dir_trans_all']
dir_cross_all = config['dir_cross_all']
dir_trans_all_rel_path = config['dir_trans_all_rel_path']
dir_cross_all_rel_path = config['dir_cross_all_rel_path']
dir_trans_all_summary = config['dir_trans_all_summary']
att_model_count = len(config['model_list'])
att_mthd_count = len(config['att_mthd'])

trans_tasks = []
for cur_att in config['att_mthd']:

    cur_att_name = cur_att['name']
    cur_att_dir = cur_att['dir']
    # create diff vs trans dir and transfer all method dir
    dir_path_diff_vs_trans = os.path.join(_dir, cur_att_dir, dir_diff_vs_trans)
    dir_path_trans_all = os.path.join(_dir, cur_att_dir, dir_trans_all)
    os.makedirs(dir_path_diff_vs_trans, exist_ok=True)
    os.makedirs(dir_path_trans_all, exist_ok=True)

    all_dir_cycle = []
    all_dir = []
    diff_trans_conf = []
    cur_mthd_id = 0
    for cur_def in config['def_mthd']:
        cur_def_name = cur_def['name']
        cur_def_dir = cur_def['dir']
        cur_mthd_dirs = []

        # make the json config file for diff vs transferability
        os.makedirs(os.path.join(_dir, cur_att_dir, cur_def_dir), exist_ok=True)
        cur_def_conf = {
            "name": cur_def_name,
            "transfer": os.path.join(cur_def['rel_path'], cur_def_dir, cur_def['summary']),
            "diff": cur_att['diff']
        }
        diff_trans_conf.append(cur_def_conf)

        for model_i in config['model_list']:
            cur_def_model_dir = os.path.join(_dir, cur_att_dir, cur_def_dir + '_' + model_i)
            cur_mthd_dirs.append(cur_def_model_dir)
            os.makedirs(cur_def_model_dir, exist_ok=True)
            utils.save_json(os.path.join(cur_def_model_dir, 'config.json'),
                            config['trans_conf'], indent=4)

        # add task for att, def, task_i
        all_dir_cycle.extend([str(cur_mthd_id)]*att_model_count)
        cur_mthd_id += 1
        cur_dir_cycle = ','.join(['0']*att_model_count)
        cur_mthd_dirs_str = ','.join(cur_mthd_dirs)
        cur_task_id = 0
        for cur_task_conf in config['task_list']:
            trans_tasks.append({
                'type': "stat_trans",
                'att': cur_att_name,
                'def': cur_def_name,
                'task': cur_task_id,
                'args': ['python3', 'stat_transferability.py', '--dirs=' + cur_mthd_dirs_str,
                         '--dir_cycle='+cur_dir_cycle, cur_task_conf['options'],
                         '--output='+os.path.join(cur_def['rel_path'], cur_def_dir, cur_task_conf['output'])]
            })
            cur_task_id += 1

        all_dir.extend(cur_mthd_dirs)

    diff_trans_conf.append({
        "name": "All",
        "transfer": os.path.join(dir_trans_all_rel_path, dir_trans_all, dir_trans_all_summary),
        "diff": cur_att['diff']
    })
    utils.save_json(os.path.join(_dir, cur_att_dir, dir_diff_vs_trans, 'config.json'), diff_trans_conf, indent=4)

    # add task for att (sum up all mthds), task_i
    all_dir_str = ','.join(all_dir)
    all_dir_cycle_str = ','.join(all_dir_cycle)
    cur_task_id = 0
    for cur_task_conf in config['task_list']:
        trans_tasks.append({
            'type': "stat_trans",
            'att': cur_att_name,
            'def': "all",
            'task': cur_task_id,
            'args': ['python3', 'stat_transferability.py', '--dirs=' + all_dir_str,
                     '--dir_cycle='+all_dir_cycle_str, cur_task_conf['options'],
                     '--output='+os.path.join(dir_trans_all_rel_path,
                                              dir_trans_all, cur_task_conf['output'])]
        })
        cur_task_id += 1

# add task for def (sum up all att), task_i
g_all_dir = []
g_all_dir_cycle = []
cur_mthd_id = 0
for cur_def in config['def_mthd']:
    cur_def_name = cur_def['name']
    cur_def_dir = cur_def['dir']
    dir_path_cross_all = os.path.join(_dir, dir_cross_all, cur_def_dir)
    os.makedirs(dir_path_cross_all, exist_ok=True)
    all_dir = []

    for cur_att in config['att_mthd']:
        cur_att_dir = cur_att['dir']
        for model_i in config['model_list']:
            cur_att_model_dir = os.path.join(_dir, cur_att_dir, cur_def_dir + '_' + model_i)
            all_dir.append(cur_att_model_dir)

    g_all_dir_cycle.extend([str(cur_mthd_id)] * att_model_count * att_mthd_count)
    cur_mthd_id += 1
    all_dir_cycle = ','.join(['0'] * att_model_count * att_mthd_count)
    all_dir_str = ','.join(all_dir)
    cur_task_id = 0
    for cur_task_conf in config['task_list']:
        trans_tasks.append({
            'type': "stat_trans",
            'att': "all",
            'def': cur_def_name,
            'task': cur_task_id,
            'args': ['python3', 'stat_transferability.py', '--dirs=' + all_dir_str,
                     '--dir_cycle=' + all_dir_cycle, cur_task_conf['options'],
                     '--output='+os.path.join(dir_cross_all_rel_path, dir_trans_all_rel_path,
                                              dir_cross_all, cur_def_dir, cur_task_conf['output'])]
        })
        cur_task_id += 1

    g_all_dir.extend(all_dir)

dir_path_diff_vs_trans = os.path.join(_dir, dir_cross_all, dir_diff_vs_trans)
dir_path_trans_all = os.path.join(_dir, dir_cross_all, dir_trans_all)
os.makedirs(dir_path_diff_vs_trans, exist_ok=True)
os.makedirs(dir_path_trans_all, exist_ok=True)

all_diff_trans_conf = []
for cur_def in config['def_mthd']:
    cur_def_name = cur_def['name']
    cur_def_dir = cur_def['dir']
    cur_mthd_dirs = []

    # make the json config file for diff vs transferability
    cur_def_conf = {
        "name": cur_def_name,
        "transfer": os.path.join(cur_def['rel_path'], cur_def_dir, cur_def['summary']),
        "diff": config['diff_all']
    }
    all_diff_trans_conf.append(cur_def_conf)

all_diff_trans_conf.append({
    "name": "All",
    "transfer": os.path.join(dir_trans_all_rel_path, dir_trans_all, dir_trans_all_summary),
    "diff": config['diff_all']
})
utils.save_json(os.path.join(_dir, dir_cross_all, dir_diff_vs_trans, 'config.json'), all_diff_trans_conf, indent=4)

g_all_dir_str = ','.join(g_all_dir)
g_all_dir_cycle_str = ','.join(g_all_dir_cycle)
cur_task_id = 0
for cur_task_conf in config['task_list']:
    trans_tasks.append({
        'type': "stat_trans",
        'att': "all",
        'def': "all",
        'task': cur_task_id,
        'args': ['python3', 'stat_transferability.py', '--dirs=' + g_all_dir_str,
                 '--dir_cycle=' + g_all_dir_cycle_str, cur_task_conf['options'],
                 '--output='+os.path.join(dir_cross_all_rel_path, dir_trans_all_rel_path,
                                          dir_cross_all, dir_trans_all, cur_task_conf['output'])]
    })
    cur_task_id += 1


def diff_vs_trans_worker(config):
    p = subprocess.Popen(' '.join(config['args']), stderr=None, stdout=None, shell=True)
    p.wait()
    print('Finish(', p.returncode, '):', config['task'], config['type'], config['att'], config['def'])


def run_task(func, tasks):
    pool = Pool(_threads)
    for i in tasks:
        pool.apply_async(func, args=(i,))
    pool.close()
    pool.join()


run_task(diff_vs_trans_worker, trans_tasks)


# add task for adv diff bin vs trans
diff_bin_vs_trans_tasks = []
tmp_att_mthd = config['att_mthd'].copy()
tmp_att_mthd.append({
    'name': 'all',
    'dir': dir_cross_all
})

cur_task_id = 0
for cur_att in tmp_att_mthd:
    cur_att_name = cur_att['name']
    cur_att_dir = cur_att['dir']

    for cur_task_conf in config['diff_bin_vs_trans']:
        diff_bin_vs_trans_tasks.append({
            'type': "diff_bin_vs_trans",
            'att': cur_att_name,
            'def': "all",
            'task': cur_task_id,
            'args': ['python3', 'compute_adv_diff_bin_vs_trans.py',
                     '--dir=' + os.path.join(_dir, cur_att_dir, dir_diff_vs_trans), cur_task_conf]
        })
        cur_task_id += 1

# attack task for evaluate white box attack
config_wh = []
wh_dir = os.path.join(_dir, config['wh_dir'])
wh_dir_sum = os.path.join(wh_dir, config['wh_dir_sum'])
os.makedirs(wh_dir_sum, exist_ok=True)
for tran_self in config['wh']:
    cur_att = config['att_mthd'][tran_self[0]]
    cur_def = config['def_mthd'][tran_self[1]]
    config_wh.append({
        "name": cur_att['name'],
        "transfer": os.path.join(cur_def['rel_path'], cur_def['dir'], cur_def['summary']),
        "diff": "../" + cur_att['diff']
    })
    dest_dir = os.path.join(wh_dir, cur_def['dir'])
    src_dir = os.path.join(cur_def['rel_path'], cur_att['dir'], cur_def['dir'])
    if os.path.exists(dest_dir):
        os.unlink(dest_dir)
    os.symlink(src_dir, dest_dir)

# symlink cross trans all dir
dest_dir = os.path.join(wh_dir, dir_trans_all)
src_dir = os.path.join(dir_cross_all_rel_path, dir_cross_all, dir_trans_all)
if os.path.exists(dest_dir):
    os.unlink(dest_dir)
os.symlink(src_dir, dest_dir)

utils.save_json(os.path.join(wh_dir_sum, 'config.json'), config_wh, indent=4)

# add task for stating the transferability self attack
tran_self_dir = os.path.join(wh_dir, "tran_self")
os.makedirs(tran_self_dir, exist_ok=True)

tran_self_dirs = []
tran_self_dir_cycle = []
cur_mthd_id = 0
for tran_self in config['tran_self']:
    cur_att = config['att_mthd'][tran_self[0]]
    cur_def = config['def_mthd'][tran_self[1]]

    cur_att_dir = cur_att['dir']
    cur_def_dir = cur_def['dir']
    for model_i in config['model_list']:
        cur_def_model_dir = os.path.join(_dir, cur_att_dir, cur_def_dir + '_' + model_i)
        tran_self_dirs.append(cur_def_model_dir)
    tran_self_dir_cycle.extend([str(cur_mthd_id)] * att_model_count)
    cur_mthd_id += 1

cur_task_id = 0
tran_self_dir_str = ','.join(tran_self_dirs)
tran_self_dir_cycle_str = ','.join(tran_self_dir_cycle)
for cur_task_conf in config['task_list']:
    diff_bin_vs_trans_tasks.append({
        'type': "trans_self",
        'att': "self",
        'def': "self",
        'task': cur_task_id,
        'args': ['python3', 'stat_transferability.py', '--dirs=' + tran_self_dir_str,
                 '--dir_cycle=' + tran_self_dir_cycle_str, cur_task_conf['options'],
                 '--output='+os.path.join(dir_cross_all_rel_path, dir_trans_all_rel_path, config['wh_dir'],
                                          "tran_self", cur_task_conf['output'])]
    })
    cur_task_id += 1

#tmp_tasks = []
for cur_whitebox_task in config['wh_task_list']:
    #diff_bin_vs_trans_tasks.append({
    cur_task_name = cur_whitebox_task["name"]
    os.makedirs(os.path.join(wh_dir_sum, cur_task_name), exist_ok=True)
    diff_bin_vs_trans_tasks.append({
        'type': "white_box",
        'att': cur_task_name,
        'def': "",
        'task': cur_task_id,
        'args': ['python3', cur_whitebox_task['cmd'],
                 '--dir=' + wh_dir_sum, "--config=config.json", cur_whitebox_task['args']]
    })
    cur_task_id += 1
#run_task(diff_vs_trans_worker, tmp_tasks)
run_task(diff_vs_trans_worker, diff_bin_vs_trans_tasks)


# task for bin vs trans cross
diff_bin_vs_trans_cross_task = []
for tmp_conf in config['diff_bin_vs_trans_cross']:
    cur_task_conf = []
    for cur_att in tmp_att_mthd:
        cur_task_conf.append({
            "name": cur_att['name'],
            "transfer": os.path.join(cur_att['dir'], dir_diff_vs_trans, tmp_conf[1])
        })
    utils.save_json(os.path.join(_dir, tmp_conf[0]), cur_task_conf, indent=4)
    diff_bin_vs_trans_cross_task.append({
        'type': "diff_bin_vs_trans_cross",
        'att': "all",
        'def': "all",
        'task': cur_task_id,
        'args': ['python3', 'compute_adv_diff_vs_tran_cross.py',
                 '--dir='+_dir, '--config='+tmp_conf[0], tmp_conf[2]]
    })

run_task(diff_vs_trans_worker, diff_bin_vs_trans_cross_task)
