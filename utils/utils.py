import os
import torch
import random
import numpy as np
import json
import deepspeed

from transformers import set_seed,AutoTokenizer

from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def print_rank_0(msg,rank=0):
    if rank<=0:
        print(msg)

def to_device(batch,device):
    