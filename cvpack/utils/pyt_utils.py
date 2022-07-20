# encoding: utf-8

import os
import sys
import time
import argparse

from collections import OrderedDict, defaultdict

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def parse_torch_devices(input_devices):
    """
    Parse user's devices input string to standard format for Torch.
    e.g. [gpu0, gpu1, ...]
    """

    from .logger import get_logger
    logger = get_logger()
    import torch
    if input_devices.endswith('*'):
        devices = list(range(torch.cuda.device_count()))
        return devices

    devices = []
    for d in input_devices.split(','):
        if '-' in d:
            start_device, end_device = d.split('-')[0], d.split('-')[1]
            assert start_device != ''
            assert end_device != ''
            start_device, end_device = int(start_device), int(end_device)
            assert start_device < end_device
            assert end_device < torch.cuda.device_count()
            for sd in range(start_device, end_device + 1):
                devices.append(sd)
        else:
            device = int(d)
            print(device)
            assert device < torch.cuda.device_count()
            devices.append(device)

    logger.info('using devices {}'.format(', '.join([str(d) for d in devices])))

    return devices


def link_file(src, target):
    """
    symbol link the source directorie to target
    """
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))


def ensure_dir(path):
    """
    create directories if *path* does not exist
    """
    if not os.path.isdir(path):
        os.makedirs(path)
