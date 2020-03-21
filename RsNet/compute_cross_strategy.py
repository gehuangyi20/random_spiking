#!/usr/bin/python3
import argparse
import os
import utils
import subprocess
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='compute adv transfer strategy')

parser.add_argument('--dir', help='data dir, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('-j', help='number of parallel process', type=int, default=1)

args = parser.parse_args()

_dir = args.dir
_threads = args.j

config = utils.load_json(os.path.join(_dir, args.config))

tasks = []
task_arg = config['task_arg']
cur_task_id = 0

for cur_mthd in config['mthd']:
    cur_mthd_name = cur_mthd['name']
    cur_mthd_transfer = cur_mthd['transfer']
    cur_mthd_diff = cur_mthd['diff']

    cur_mthd_config = []

    for cur_strategy in config['strategy']:
        cur_mthd_config.append({
            "name": cur_strategy['name'],
            "transfer": cur_strategy['transfer'] % cur_mthd_transfer,
            "diff": cur_strategy['diff'] % cur_mthd_diff
        })

    os.makedirs(os.path.join(_dir, cur_mthd_name), exist_ok=True)
    utils.save_json(os.path.join(_dir, cur_mthd_name, 'config.json'), cur_mthd_config, indent=4)

    dest = os.path.join(_dir, cur_mthd_name + '-l2_mn.pdf')
    src = os.path.join(cur_mthd_name, cur_mthd_name + '-l2_mn.pdf')
    if os.path.exists(dest):
        os.unlink(dest)
    os.symlink(src, dest)

    tasks.append({
        'type': "strategy",
        'name': cur_mthd_name,
        'task': cur_task_id,
        'args': ['python3', 'compute_adv_diff_bin_vs_trans.py',
                 '--dir=' + os.path.join(_dir, cur_mthd_name), '--output=' + cur_mthd_name, task_arg]
    })
    cur_task_id += 1


def worker(config):
    p = subprocess.Popen(' '.join(config['args']), stderr=None, stdout=None, shell=True)
    p.wait()
    print('Finish(', p.returncode, '):', config['task'], config['type'], config['name'])


def run_task(func, tasks):
    pool = Pool(_threads)
    for i in tasks:
        pool.apply_async(func, args=(i,))
    pool.close()
    pool.join()


run_task(worker, tasks)
