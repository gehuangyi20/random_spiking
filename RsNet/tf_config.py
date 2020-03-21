import os
import tensorflow as tf
from tensorflow.python.client import device_lib

gpu_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
# If true, the allocator does not pre-allocate the entire specified
# GPU memory region, instead starting small and growing as needed.
gpu_config.gpu_options.allow_growth = True
# pre-allocate 40% of the GPU memory, may gow later
gpu_config.gpu_options.per_process_gpu_memory_fraction = 1

_LOCAL_DEVICES_ = None

CHANNELS_LAST = "channels_last"
CHANNELS_FIRST = "channels_first"


def setup_visibile_gpus(devices=None):
    """
    setup visible gpus to tensorflow, must set this value before calling tensorflow session
    :param devices: Default None, not effective
    :return:
    """
    if devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


def get_available_gpus():
    global _LOCAL_DEVICES_
    _LOCAL_DEVICES_ = device_lib.list_local_devices()
    return [x.name for x in _LOCAL_DEVICES_ if x.device_type == 'GPU']


def get_gpu_name(gpu=None):
    devices = get_available_gpus()
    if gpu is None:
        if len(devices) > 0:
            gpu = devices[0]
        else:
            gpu = '/cpu:0'
    elif isinstance(gpu, int):
        if len(devices) == 0:
            gpu = '/cpu:0'
        elif len(devices) > gpu:
            gpu = devices[gpu]
        else:
            raise ValueError('gpu idx', gpu, 'is a valid device name')
    elif isinstance(gpu, str):
        if gpu not in devices and gpu != '/cpu:0':
            raise ValueError(gpu, 'is a valid device name')
    else:
        raise ValueError('gpu is either full name or gpu index')

    return gpu
