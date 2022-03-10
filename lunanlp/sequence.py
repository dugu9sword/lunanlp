import random
from typing import List


def random_drop(idx: List, drop_rate) -> List:
    assert 0.0 < drop_rate < 0.5
    ret = list(
        filter(lambda x: x is not None,
               map(lambda x: None if random.random() < drop_rate else x, idx)))
    if len(ret) == 0:
        return ret
    return ret


def batch_drop(idx: List[List], drop_rate) -> List[List]:
    return list(map(lambda x: random_drop(x, drop_rate), idx))


def batch_pad(idx: List[List], pad_ele=0, pad_len=None) -> List[List]:
    if pad_len is None:
        pad_len = max(map(len, idx))
    return list(map(lambda x: x + [pad_ele] * (pad_len - len(x)), idx))


def batch_mask(idx: List[List], mask_zero=True) -> List[List]:
    if mask_zero:
        good_ele, mask_ele = 1, 0
    else:
        good_ele, mask_ele = 0, 1
    max_len = max(map(len, idx))
    return list(
        map(lambda x: [good_ele] * len(x) + [mask_ele] * (max_len - len(x)),
            idx))


def batch_mask_by_len(lens: List[int], mask_zero=True) -> List[List]:
    if mask_zero:
        good_ele, mask_ele = 1, 0
    else:
        good_ele, mask_ele = 0, 1
    max_len = max(lens)
    return list(
        map(lambda x: [good_ele] * x + [mask_ele] * (max_len - x), lens))


def batch_append(idx: List[List], append_ele, before=False) -> List[List]:
    if not before:
        return list(map(lambda x: x + [append_ele], idx))
    else:
        return list(map(lambda x: [append_ele] + x, idx))


def batch_lens(idx: List[List]) -> List:
    return list(map(len, idx))


def as_batch(idx: List) -> List[List]:
    return [idx]


def flatten_lst(lst: List[List]) -> List:
    return [i for sub_lst in lst for i in sub_lst]
