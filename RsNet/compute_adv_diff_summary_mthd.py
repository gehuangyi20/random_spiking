#!/usr/bin/python3
import argparse
import os
import utils
import subprocess
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='compute adv diff summary for multiple methods')

parser.add_argument('--dir', help='data dir, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='config.json')
parser.add_argument('-j', help='number of parallel process', type=int, default=1)

args = parser.parse_args()

_dir = args.dir
_threads = args.j

config = utils.load_json(os.path.join(_dir, args.config))
set_name = config['set_name']
att_dir = config['att_dir']
att_name_prefix = config['att_name_prefix']
att_name_rel_path = config['att_name_rel_path'] if 'att_name_rel_path' in config else ''

# process individual adv diff tasks
adv_diff_tasks = []
for i in range(len(config['name'])):
    mthd_name = config['name'][i]
    if 'options_diff' in config:
        options_diff = config['options_diff'][i]
    else:
        options_diff = []

    if 'options_img' in config:
        options_img = config['options_img'][i]
    else:
        options_img = []

    for model_id in config['id']:
        mthd_dir = os.path.join(_dir, mthd_name, str(model_id))
        # mthd_dir = os.path.join(_dir, mthd_name)
        os.makedirs(mthd_dir, exist_ok=True)
        model_conf = [
            {
                "name": att_name_prefix,
                "conf": config['conf']
            }
        ]
        utils.save_json(os.path.join(mthd_dir, 'list.json'), model_conf, indent=4)

        for conf in config['conf']:
            if "att_name_format" in config:
                cur_arg = []
                for arg_t in config['att_name_format'][1]:
                    if arg_t == 'c':
                        cur_arg.append(conf)
                    elif arg_t == 'm':
                        cur_arg.append(att_name_prefix)
                att_name = config['att_name_format'][0] % tuple(cur_arg)
            else:
                att_name = '_'.join([att_name_prefix, model_id, conf])
            out_dir = os.path.join(mthd_dir, att_name_prefix+'_'+conf)
            if 'att_dir_option' in config:
                cur_att_dir_arg = []
                for arg_t in config['att_dir_option']:
                    if arg_t == 'c':
                        cur_att_dir_arg.append(conf)
                cur_att_dir = att_dir % tuple(cur_att_dir_arg)
            else:
                cur_att_dir = att_dir

            if "name_format" in config:
                cur_arg = []
                for arg_t in config['name_format'][1]:
                    if arg_t == 'id':
                        cur_arg.append(model_id)
                    elif arg_t == 'm':
                        cur_arg.append(mthd_name)
                cur_mthd_name = config['name_format'][0] % tuple(cur_arg)
            else:
                cur_mthd_name = mthd_name

            # compute adv diff
            cur_args = ['python3', 'compute_adv_diff.py', '--dir=' + cur_att_dir, '--name=' +os.path.join(att_name_rel_path, cur_mthd_name),
                        '--attack_name='+att_name, '--set_name='+set_name, '--is_normalize=yes',
                        '--out_dir='+out_dir]
            cur_args.extend(options_diff)
            adv_diff_tasks.append({
                'type': "adv_diff",
                'mthd': cur_mthd_name,
                'model_id': model_id,
                'conf': conf,
                'args': cur_args
            })

            # show adv image
            cur_args = ['python3', '../showMnistImage.py', '--dir=' + os.path.join(cur_att_dir, att_name_rel_path, cur_mthd_name),
                        '--data_dir=' + cur_att_dir,
                        '--start_idx=' + config['show_img_start'], '--count=' + config['show_img_num'],
                        '--att_name=' + att_name, '--col=' + config['show_img_col'],
                        '--set_name=' + set_name, '--step=' + config['show_img_step'],
                        '--duplicate=' + config['show_img_duplicate'],
                        '--output=' + os.path.join(mthd_dir, '..', att_name+'.png')]
            cur_args.extend(options_img)
            adv_diff_tasks.append({
                'type': "adv_show_img",
                'mthd': cur_mthd_name,
                'model_id': model_id,
                'conf': conf,
                'args': cur_args,
                "stdout": os.path.join(mthd_dir, '..', "sum_"+att_name+'.txt')
            })


def adv_diff_worker(config):
    if "stdout" in config:
        fp = open(config['stdout'], 'wb')
    else:
        fp = None
    print(' '.join(config['args']))
    p = subprocess.Popen(config['args'], stderr=subprocess.DEVNULL, stdout=fp)
    p.wait()
    if "stdout" in config:
        fp.close()
    print('Finish(', p.returncode, '):', config['type'], config['mthd'], config['model_id'], config['conf'])


def run_task(func, tasks):
    pool = Pool(_threads)
    for i in tasks:
        pool.apply_async(func, args=(i,))
    pool.close()
    pool.join()


run_task(adv_diff_worker, adv_diff_tasks)


# summarize adv diff for each methos
adv_diff_sum_tasks = []
for mthd_name in config['name']:
    cur_dir = []
    for model_id in config['id']:
        cur_dir.append(os.path.join(_dir, mthd_name, str(model_id)))
    cur_dir = ','.join(cur_dir)

    adv_diff_sum_tasks.append({
        'type': "adv_diff_sum",
        'mthd': mthd_name,
        'args': ['python3', 'compute_adv_diff_summary.py', '--dirs=' + cur_dir, '--output_file='+config['sum_out']]
    })


def adv_diff_sum_worker(config):
    p = subprocess.Popen(config['args'], stdout=subprocess.DEVNULL)
    p.wait()
    print('Finish(', p.returncode, '):', config['type'], config['mthd'])


run_task(adv_diff_sum_worker, adv_diff_sum_tasks)
