from itertools import islice
from typing import Callable, Iterable, Iterator, List, Union

import numpy as np


def group_fields(lst: List[object],
                 keys: Union[str, List[str]] = None,
                 indices: Union[int, List[int]] = None):
    """
    Examples:
        >>> ilst = [('a', 0), ('b', 1), ('c', 2)]
        >>> group_fields(ilst, 0)
        ['a', 'b', 'c']
        >>> group_fields(ilst, [0, 1])
        [['a', 'b', 'c'], [0, 1, 2]]
        >>> klst = [{'x': 0, 'y': 1}, {'x': 2, 'y': 3}]
        >>> group_fields(klst, ['y', 'x'])
        [[1, 3], [0, 2]]
    """
    assert (keys is None) ^ (indices is None)
    is_single = False
    if keys is not None:
        if not isinstance(keys, list):
            keys = [keys]
            is_single = True
        indices = keys
    if indices is not None:
        if not isinstance(indices, list):
            indices = [indices]
            is_single = True
    ret = []
    for idx in indices:
        ret.append(list(map(lambda ele: ele[idx], lst)))
    if is_single:
        return ret[0]
    else:
        return ret


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


def lazy_group_by_size(iterable: Iterable, group_size: int) -> Iterator[List]:
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


def lazy_group_by_max_tokens(
    iterable: Iterator,  #
    max_tokens: int,
    len_func: Callable = lambda x: x[0].count(" ")
) -> Iterator[List]:
    batch = []
    count = 0
    for instance in iter(iterable):
        batch.append(instance)
        count += len_func(instance)
        if count > max_tokens:
            yield batch
            batch = []
            count = 0
    if len(batch) > 0:
        yield batch


if __name__ == "__main__":
    import doctest
    doctest.testmod()
