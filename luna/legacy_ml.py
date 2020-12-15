import time
import random
from typing import List
import numpy as np
import os
import re


# def checkpoint_paths(path, pattern=r'checkpoint@(\d+)\.pt'):
#     """Retrieves all checkpoints found in `path` directory.
#     Checkpoints are identified by matching filename to the specified pattern. If
#     the pattern contains groups, the result will be sorted by the first group in
#     descending order.
#     """
#     pt_regexp = re.compile(pattern)
#     files = os.listdir(path)

#     entries = []
#     for i, f in enumerate(files):
#         m = pt_regexp.fullmatch(f)
#         if m is not None:
#             idx = int(m.group(1)) if len(m.groups()) > 0 else i
#             entries.append((idx, m.group(0)))
#     return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


# # model_name = /aaa/bbb/ccc/model
# # find:
# #    /aaa/bbb/ccc/model.1
# #    /aaa/bbb/ccc/model.2
# #  * /aaa/bbb/ccc/model.best


# def fetch_best_ckpt_name(model_path):
#     model_name = model_path + '.best'
#     if os.path.exists(model_name):
#         print("Found checkpoint {}".format(model_name))
#     else:
#         model_name = fetch_last_ckpt_name(model_path)
#         print("Best checkpoint not found, use latest {} instead".format(
#             model_name))
#     return model_name


# def fetch_last_ckpt_name(model_path):
#     splash_index = model_path.rindex('/')
#     model_folder = model_path[:splash_index]
#     model_file = model_path[splash_index + 1:]
#     files = checkpoint_paths(model_folder, r'{}.(\d+)'.format(model_file))
#     return files[0]


class DataSet:
    def __init__(self):
        self.data = []
        self.__next = 0

    def next_batch(self, batch_size, fill_batch=True):
        if self.__next + batch_size > len(self.data):
            if fill_batch:
                ret = self.data[self.size - batch_size:self.size]
            else:
                ret = self.data[self.__next:self.size]
            self.__next = self.size
        else:
            ret = self.data[self.__next:self.__next + batch_size]
            self.__next += batch_size
        return ret

    @property
    def size(self):
        return len(self.data)

    @property
    def finished(self):
        return self.__next == self.size

    def reset(self, shuffle=True):
        self.__next = 0
        if shuffle:
            random.shuffle(self.data)


class ProgressManager:
    def __init__(self, total):
        self.__start = time.time()
        self.__prev_prev = time.time()
        self.__prev = time.time()
        self.__total = total
        self.__complete = 0

    def update(self, batch_num):
        self.__complete += batch_num
        self.__prev_prev = self.__prev
        self.__prev = time.time()

    @property
    def batch_time(self):
        return self.__prev - self.__prev_prev

    @property
    def cost_time(self):
        return self.__prev - self.__start

    @property
    def rest_time(self):
        return self.cost_time / self.__complete * (self.__total -
                                                   self.__complete)

    @property
    def complete_num(self):
        return self.__complete

    @property
    def total_num(self):
        return self.__total


class TrainingStopObserver:
    def __init__(self,
                 lower_is_better,
                 can_stop_val=None,
                 must_stop_val=None,
                 min_epoch=None,
                 max_epoch=None,
                 epoch_num=None):
        self.history_values = []
        self.history_infos = []
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.epoch_num = epoch_num
        self.lower_is_better = lower_is_better
        self.can_stop_val = can_stop_val
        self.must_stop_val = must_stop_val

    def check_stop(self, value, info=None) -> bool:
        self.history_values.append(value)
        self.history_infos.append(info)
        if self.can_stop_val is not None:
            if self.lower_is_better and value > self.can_stop_val:
                return False
            if not self.lower_is_better and value < self.can_stop_val:
                return False
        if self.must_stop_val is not None:
            if self.lower_is_better and value < self.must_stop_val:
                return True
            if not self.lower_is_better and value > self.must_stop_val:
                return True
        if self.max_epoch is not None and len(
                self.history_values) > self.max_epoch:
            return True
        if self.min_epoch is not None and len(
                self.history_values) <= self.min_epoch:
            return False
        lower = value < np.mean(self.history_values[-(self.epoch_num + 1):-1])
        if self.lower_is_better:
            return False if lower else True
        else:
            return True if lower else False

    def select_best_point(self):
        if self.lower_is_better:
            chosen_id = int(np.argmin(self.history_values[self.min_epoch:]))
        else:
            chosen_id = int(np.argmax(self.history_values[self.min_epoch:]))
        return self.history_values[
            self.min_epoch + chosen_id], self.history_infos[self.min_epoch +
                                                            chosen_id]


def hit(scores: List[List], gold: List, k: int):
    corr = 0
    total = len(gold)
    for score, label in zip(scores, gold):
        if label in list(reversed(np.argsort(score)))[:k]:
            corr += 1
    return corr / total


# def get_prf(pred: List, gold: List):
#     precision, recall, f1, _ = sklearn_prf(gold, pred, beta=1, average=None)
#     return precision.tolist(), recall.tolist(), f1.tolist()

# def show_prf(pred: List, gold: List, classes):
#     precision, recall, f1, _ = sklearn_prf(gold, pred, beta=1, labels=classes)
#     head = "{:4}|{:15}|{:10}|{:10}|{:10}"
#     content = "{:4}|{:15}|{:10f}|{:10f}|{:10f}"
#     print(Color.cyan(head.format("ID", "Class", "Precision", "Recall", "F1")))
#     for i in range(len(classes)):
#         print(Color.white(content.format(i, classes[i], precision[i], recall[i], f1[i])))

# def score2rank(scores) -> list:
#     return np.argmax(scores, 1).tolist()

# def accuracy(scores: List[List], gold: List):
#     return hit(scores, gold, 1)