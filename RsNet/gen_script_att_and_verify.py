import os
import json
import argparse

parser = argparse.ArgumentParser(description='generate attack and verify task script')

parser.add_argument('-d', '--dir', help='directory, required', type=str, default=None)
parser.add_argument('-c', '--config', help='config file, default config.json', type=str, default='att.json')
parser.add_argument('-o', '--output', help='output script file, default sh_att.sh', type=str, default='sh_attack_')

args = parser.parse_args()

_dir = args.dir
output_file = args.output

config_fp = open(os.path.join(_dir, args.config), "rb")
json_str = config_fp.read()
config_fp.close()

config = json.loads(json_str.decode())

g_options = " ".join(config['g_options']) + " "

for cur_task in config['task']:
    iter_sub_task = cur_task['iter_subtask'] if 'iter_subtask' in cur_task else ['0']
    task_options = " ".join(cur_task['options']) + " "
    out_fp = open(os.path.join(_dir, output_file + cur_task['name'] + '.sh'), "wb")

    if 'iter_options' in cur_task:
        cur_iter_len = len(cur_task['iter_options'])
    else:
        for cur_iter_data in cur_task['iter']:
            if 'options' in cur_iter_data:
                cur_iter_len = len(cur_iter_data['options'])
                break

    for cur_sub_task in config['sub_task']:
        out_fp.write(("#%s\n" % cur_sub_task['name']).encode())

        cur_model_start = cur_sub_task["model_start"] if "model_start" in cur_sub_task else 0

        for cur_iter_sub_task in iter_sub_task:

            for i in range(cur_iter_len):
                sub_task_options = " ".join(cur_sub_task['options']) + " "
                out_fp.write(g_options.encode())
                out_fp.write(task_options.encode())
                out_fp.write(sub_task_options.encode())

                for cur_model in cur_sub_task['model']:
                    cur_model_option = cur_model['prefix']
                    cur_model_list = []
                    for model_i in range(cur_model_start, cur_model_start + cur_sub_task['model_len']):
                        cur_model_arg = []
                        for arg_t in cur_model['arg']:
                            if arg_t == 'p':
                                cur_model_arg.append(_dir)
                            else:
                                cur_model_arg.append(model_i)
                        cur_model_list.append(cur_model['format'] % tuple(cur_model_arg))

                    cur_model_option += ','.join(cur_model_list)
                    out_fp.write(("%s " % cur_model_option).encode())

                iter_sub_task_data = []
                if 'iter' in cur_task:
                    iter_sub_task_data.extend(cur_task['iter'])
                if 'iter' in cur_sub_task:
                    iter_sub_task_data.extend(cur_sub_task['iter'])
                for cur_iter_data in iter_sub_task_data:
                    if 'options' in cur_iter_data:
                        cur_iter_option = cur_iter_data['options'][i]
                    else:
                        cur_iter_option = cur_task['iter_options'][i]

                    cur_iter_text_fmt = cur_iter_data['format']
                    cur_iter_text_arg = []

                    for arg_t in cur_iter_data['arg']:
                        if arg_t == 'p':
                            cur_iter_text_arg.append(_dir)
                        elif arg_t == 'tn':
                            cur_iter_text_arg.append(cur_task['name'])
                        elif arg_t == 'o':
                            cur_iter_text_arg.append(cur_iter_option)
                        else:
                            cur_iter_text_arg.append(cur_iter_sub_task)

                    out_fp.write(((cur_iter_text_fmt + ' ') % tuple(cur_iter_text_arg)).encode())

                out_fp.write("\n".encode())

            out_fp.write("\n".encode())

    out_fp.close()

