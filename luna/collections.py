from typing import List, Union
import numpy as np
import inspect


class Aggregator:
    """You may use an Aggregator to aggregate values any where,
        and reduce them through any way you like. The tool prevent you from
        writing many dirty code to track values in different places/iterations.

    Example:

        Without an Aggregator:
            >>> key1_list, key2_list, key3_list = [], [], []
            >>> # In an iteration, you collect them as:
            >>> key1_list.append(1)
            >>> key2_list.append(2)
            >>> key3_list.append(5)
            >>> # while in another iteration,
            >>> key1_list.append(2)
            >>> key2_list.append(2)
            >>> key3_list.append(4)
            >>> # ...
            
        With an Aggregator:
            >>> agg = Aggregator()
            >>> # In an iteration, you collect them as:
            >>> agg.aggregate((key1, 1), (key2, 2), (key3, 5) ...)
            >>> # while in another iteration,
            >>> agg.aggregate((key1, 3), (key2, 2), (key3, 5) ...)
            >>> agg.aggregate((key1, 5), (key2, 4), (key3, 5) ...)
            >>> # ...

        And finally, you can reduce the values:
            >>> agg.aggregated(key1)
            [1, 3, 5]
            >>> agg.aggregated(key1, 'mean')
            3
            >>> agg.aggregated(key1, np.sum)
            9
            >>> agg.mean(key1)
            3
    """
    def __init__(self):
        self.__kv_mode = False
        self.__keys = None
        self.__saved = None

    @property
    def size(self):
        return len(self.__saved[0])

    def has_key(self, key):
        if self.__keys is None:
            return False
        if not self.__kv_mode:
            return False
        return key in self.__keys

    def aggregate(self, *args):
        # First called, init the collector and decide the key mode
        if self.__saved is None:
            if Aggregator.__args_kv_mode(*args):
                self.__kv_mode = True
                self.__keys = list(map(lambda x: x[0], args))
            # else:
            #     self.keys = ['__{}' for i in range(len(args))]
            self.__saved = [[] for _ in range(len(args))]
        # Later called
        if Aggregator.__args_kv_mode(*args) != self.__kv_mode:
            raise Exception("you must always specify a key or not")
        for i in range(len(args)):
            if self.__kv_mode:
                saved_id = self.__keys.index(args[i][0])
                to_save = args[i][1]
            else:
                saved_id = i
                to_save = args[i]
            if isinstance(to_save, list):
                self.__saved[saved_id].extend(to_save)
            else:
                self.__saved[saved_id].append(to_save)

    @staticmethod
    def __args_kv_mode(*args):
        # print("args is {}".format(args))
        has_key_num = 0
        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 2 and isinstance(
                    arg[0], str):
                has_key_num += 1
        if has_key_num == len(args):
            return True
        if has_key_num == 0:
            return False
        raise Exception("you must specify a key for all args or not")

    def mean(self, key):
        return self.aggregated(key, 'mean')

    def std(self, key):
        return self.aggregated(key, 'std')

    def sum(self, key):
        return self.aggregated(key, 'sum')

    def list(self, key):
        return self.aggregated(key)

    def aggregated(self, key=None, reduce: Union[str, callable] = 'no'):
        if reduce == 'no':

            def reduce_fn(x):
                return x
        elif reduce == 'mean':
            reduce_fn = np.mean
        elif reduce == 'sum':
            reduce_fn = np.sum
        elif reduce == 'std':
            reduce_fn = np.std
        elif inspect.isfunction(reduce):
            reduce_fn = reduce
        else:
            raise Exception(
                'reduce must be None, mean, sum, std or a function.')

        if key is None:
            if not self.__kv_mode:
                if len(self.__saved) == 1:
                    return reduce_fn(self.__saved[0])
                else:
                    return tuple(reduce_fn(x) for x in self.__saved)
            else:
                raise Exception("you must specify a key")
        elif key is not None:
            if self.__kv_mode:
                saved_id = self.__keys.index(key)
                return reduce_fn(self.__saved[saved_id])
            else:
                raise Exception("you cannot specify a key")


def group_fields(lst: List[object],
                 keys: Union[str, List[str]] = None,
                 indices: Union[int, List[int]] = None):
    assert keys is None or indices is None
    is_single = False
    if keys:
        if not isinstance(keys, list):
            keys = [keys]
            is_single = True
        indices = []
        for key in keys:
            obj_type = type(lst[0])
            idx = obj_type._fields.index(key)
            indices.append(idx)
    else:
        if not isinstance(indices, list):
            indices = [indices]
            is_single = True
    rets = []
    for idx in indices:
        rets.append(list(map(lambda item: item[idx], lst)))
    if is_single:
        return rets[0]
    else:
        return rets


class CherryPicker:
    def __init__(self, lower_is_better, compare_fn=None):
        self.lower_is_better = lower_is_better
        self.history_values = []
        self.history_infos = []
        self.compare_fn = compare_fn

    def add(self, value, info):
        self.history_infos.append(info)
        self.history_values.append(value)

    @property
    def size(self):
        return len(self.history_values)

    def select_best_point(self):
        if self.size == 0:
            raise Exception("Nothing to pick.")
        # np.argmin selects the first occurrence of the min
        if self.compare_fn is None:
            if self.lower_is_better:
                chosen_id = int(np.argmin(self.history_values))
            else:
                chosen_id = int(np.argmax(self.history_values))
        else:
            chosen_id = len(self.history_values) - 1
            chosen_val = self.history_values[-1]
            for i in reversed(range(len(self.history_values))):
                if self.lower_is_better:
                    if self.compare_fn(self.history_values[i],
                                       chosen_val) <= 0:
                        chosen_id = i
                        chosen_val = self.history_values[chosen_id]
                else:
                    if self.compare_fn(self.history_values[i],
                                       chosen_val) >= 0:
                        chosen_id = i
                        chosen_val = self.history_values[chosen_id]
        return chosen_id, self.history_values[chosen_id], self.history_infos[
            chosen_id]

