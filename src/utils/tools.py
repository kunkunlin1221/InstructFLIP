import json
import time
from itertools import chain
from typing import Any

import cv2
import torch
import torch.nn as nn
import yaml


def set_everything():
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1)
    torch.set_printoptions(4, sci_mode=False)
    torch.set_float32_matmul_precision('high')
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True


def load_yaml(fpath):
    with open(fpath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def save_yaml(obj, fpath):
    with open(fpath, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)


def colorstr(obj: Any, color='blue', fmt='bold'):
    '''
    This function is make colorful string for python.

    Args:
        obj (Any): The object you want to make it print colorful.
        color (str, optional):
            The print color of obj. Defaults to 'blue'.
            Options = {
                'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white',
                'bright_black', 'bright_red'b 'bright_green' 'bright_yellow',
                'bright_blue', 'bright_magenta', 'bright_cyan', bright_white',
            }
        fmt (str, optional):
            The print format of obj. Defaults to 'bold'.
            Options = {
                'bold', 'underline'
            }

    Returns:
        string: color string.
    '''
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
    }
    formats = {
        'bold': '\033[1m',
        'Italic': '\033[3m',
        'underline': '\033[4m',
    }
    if color not in colors.keys():
        raise KeyError(f'color must be one of {colors.keys()}')
    if fmt not in formats.keys():
        raise KeyError(f'fmt must be one of {formats.keys()}')
    return f'{colors[color]}{formats[fmt]}{obj}\033[0m'


def restore_from_ckpt(model: nn.Module, model_ckpt: dict, verbose: bool = False):
    ori_state_dict = model.state_dict()
    unexpected = []

    for k, v in model_ckpt.items():
        target = ori_state_dict.get(k, None)
        if target is not None and target.shape == v.shape:
            ori_state_dict[k] = v
        else:
            if target is not None:
                unexpected.append(f"ckpt's {k} = {v.shape}, but model's {k} = {target.shape}")
            else:
                unexpected.append(f"target of {k} is None")
    if verbose and len(unexpected):
        print(colorstr(f'Unexpected key and value: \n{unexpected}', 'red'))

    sucessful = model.load_state_dict(ori_state_dict, strict=True)

    if not sucessful:
        raise KeyError(f'load_state_dict has an error.')


def get_current_time():
    return time.strftime("%Y%m%d%H%M", time.localtime())


def flatten_list(l: list) -> list:
    '''
    Function to flatten a list.

    Args:
        l (List[List[...]]):
            Nested lists that needs to be flattened.

    Returns:
        flatten list (list): flatted list.
    '''
    out = list(chain(*l))
    if len(out) and isinstance(out[0], list):
        out = flatten_list(out)
    return out


def save_json(obj, fpath):
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data
