import os
import random
import re
import time
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from . import ram

__model_path__ = "saved/models"


def get_device(model):
    return next(model.parameters()).device


def cast_list(array, squeeze=True):
    if isinstance(array, torch.Tensor):
        return cast_list(array.detach().cpu().numpy(), squeeze)
    if isinstance(array, list):
        return cast_list(np.array(array), squeeze)
    if isinstance(array, np.ndarray):
        if squeeze:
            array = array.squeeze().tolist()
            return array if isinstance(array, list) else [array]
        else:
            return array.tolist()


def cast_item(array):
    if isinstance(array, np.ndarray):
        array = array.tolist()
    while True:
        if isinstance(array, list):
            if len(array) != 1:
                raise Exception("More than one item!")
            array = array[0]
        else:
            break
    return array


def allocate_cuda_device(cuda_idx) -> torch.device:
    if torch.cuda.is_available() and cuda_idx >= 0:
        device = torch.device("cuda:{}".format(cuda_idx))
    else:
        device = torch.device("cpu")
    return device


def set_gpu_device(device_id):
    torch.cuda.set_device(device_id)


def register_layer_hook(layer):
    # grad_in/grad_out/inputs are tuples, outputs is a tensor
    def fw_hook_layers(module, inputs, outputs):
        ram.append('LAYER_HOOK.fw', outputs)

    def bw_hook_layers(module, grad_in, grad_out):
        ram.append('LAYER_HOOK.bw', grad_out[0])

    fw_hook = layer.register_forward_hook(fw_hook_layers)
    bw_hook = layer.register_backward_hook(bw_hook_layers)
    return [fw_hook, bw_hook]


def read_layer_hook(order):
    fw = ram.read('LAYER_HOOK.fw')[order]
    bw = ram.read('LAYER_HOOK.bw')[-(order + 1)]
    return fw, bw


def reset_layer_hook():
    ram.reset("LAYER_HOOK")


def gpu(*x):
    if torch.cuda.is_available():
        if len(x) == 1:
            return x[0].cuda()
        else:
            return map(lambda m: m.cuda(), x)
    else:
        if len(x) == 1:
            return x[0]
        else:
            return x


def exist_model(saved_model_name):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    for file in os.listdir(__model_path__):
        file = file[:-5]
        name = file.split('@')[0]
        ckpt = int(file.split('@')[1])
        if name == saved_model_name:
            return True
    return False


def load_model(model, saved_model_name, checkpoint=-1):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    if checkpoint == -1:
        for file in os.listdir(__model_path__):
            if file[-5:] == '.ckpt':
                file = file[:-5]
                name = file.split('@')[0]
                ckpt = int(file.split('@')[1])
                if name == saved_model_name and ckpt > checkpoint:
                    checkpoint = ckpt
    path = "{}/{}@{}.ckpt".format(__model_path__, saved_model_name, checkpoint)
    if not os.path.exists(path):
        print("Checkpoint {} not found.".format(path))
    else:
        print("Restore model from checkpoint {}.".format(path))
        if not torch.cuda.is_available():
            model.load_state_dict(
                torch.load(path, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path))
    return checkpoint


def save_model(model, saved_model_name, checkpoint=-1):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    if checkpoint == -1:
        checkpoint = 0
    path = "{}/{}@{}.ckpt".format(__model_path__, saved_model_name, checkpoint)
    torch.save(model.state_dict(), path)
    print("Model saved to {}.".format(path))
    return checkpoint + 1


class ModelManager:
    def __init__(self, model, model_name, seconds=-1, init_ckpt=-1):
        self.model = model
        self.model_name = model_name
        self.seconds = seconds
        self.last_time = time.time()
        self.ckpt = load_model(model=self.model,
                               saved_model_name=self.model_name,
                               checkpoint=init_ckpt)

    def save(self):
        curr_time = time.time()
        if curr_time - self.last_time > self.seconds:
            save_model(model=self.model,
                       saved_model_name=self.model_name,
                       checkpoint=self.ckpt)
            self.last_time = curr_time

    def next_ckpt(self):
        self.ckpt += 1


def find_module(model, name: str, match_type='partial'):
    modules = []
    for idx, (mod_name, module) in enumerate(model.named_modules()):
        if match_type == 'full':
            if name == mod_name:
                modules.append(module)
        elif match_type == 'partial':
            if name in mod_name:
                modules.append(module)
        elif match_type == 're':
            if re.match(name, mod_name):
                modules.append(module)
    return modules


def load_word2vec(word_dict: Dict[str, int], word2vec_path, dim, norm=True):
    print("Load word2vec from {}".format(word2vec_path))
    pre_embedding = np.random.normal(0, 1, (len(word_dict), dim))
    # x = 0
    found = 0
    emb_size = -1
    error_num = 0
    for line in tqdm(open(word2vec_path, errors='ignore')):
        # x += 1
        # print("Process line {} in file {}".format(x, word2vec_path))
        split = re.split(r"\s+", line.strip())
        if emb_size == -1:
            emb_size = len(split) - 1
        if len(split) != emb_size + 1 or len(split) < 10:
            error_num += 1
            continue
        word = split[0]
        if word in word_dict:
            found += 1
            pre_embedding[word_dict[word]] = np.array(
                list(map(float, split[1:])))
    print("Error line: ", error_num)
    print("Pre_train match case: {:.4f}".format(found / len(word_dict)))
    if norm:
        pre_embedding = pre_embedding / np.std(pre_embedding)
    return torch.from_numpy(pre_embedding)


def idx_to_msk(idx, num_classes):
    """
    Examples:
    >>> idx: [1, 2, 3]
    msk: [[0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
    """
    assert idx.dim() == 1, "the dimension of idx must be 1, e.g. [1, 2, 3]"
    msk = torch.zeros((idx.size(0), num_classes), device=idx.device)
    msk.scatter_(1, idx.view(-1, 1), 1)
    msk = msk.byte()
    return msk


def msk_to_idx(msk):
    assert msk.sum() == msk.size(0), \
        "only one element is allowed to be 1 in each row"
    return msk.nonzero()[:, 1].flatten()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
