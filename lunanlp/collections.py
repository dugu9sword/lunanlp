from itertools import islice
from typing import Callable, Iterable, Iterator, List, Union


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


def locate_chunk(num_total, num_chunk, chunk_id):
    start = num_total // num_chunk * chunk_id
    end = num_total // num_chunk * (chunk_id + 1)
    if chunk_id == num_chunk - 1:
        end = num_total
    return start, end


def chunks(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
