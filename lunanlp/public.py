import logging
import os
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from . import ram
import torch

import arrow
import numpy as np
import psutil

# arg_required = object()
# arg_optional = object()
# arg_place_holder = object()

logger = logging.getLogger(__name__)

CACHE_DIR = Path("saved/vars")


@contextmanager
def time_record(sth=None):
    start = time.time()
    yield
    end = time.time()
    if sth:
        print(sth, "cost {:.3} seconds".format(end - start))
    else:
        print("cost {:.3} seconds".format(end - start))

@contextmanager
def timeit(sth, times=10):
    start = time.time()
    yield
    end = time.time()
    key = f"__TIMEIT__{sth}"
    if not ram.has(key):
        ram.write(key, (1, end - start))
    else:
        agg_num, agg_cost = ram.read(key)
        agg_num += 1
        agg_cost += end - start
        ram.write(key, (agg_num , agg_cost))
        if agg_num == times:
            print(sth, "cost {:.3} seconds per call".format(agg_cost / agg_num))
            ram.pop(key)


def not_executed(flag):
    flag = f"_EXECUTE_ONCE_FLAG_{flag}"
    if ram.has_flag(flag):
        return False
    else:
        ram.set_flag(flag)
        return True


def print_once(msg):
    flag = f"_PRINT_ONCE_FLAG_{msg}"
    if not ram.has_flag(flag):
        print(msg)
    ram.set_flag(flag)


def _get_path(path=None) -> Path:
    if path is None:
        path = CACHE_DIR
    else:
        if not isinstance(path, Path):
            path = Path(path)
    return path


def save_var(variable, name, path=None):
    path = _get_path(path)
    if not path.exists():
        os.makedirs(path, exist_ok=True)
    pickle.dump(variable, open(path / (name + '.pkl'), "wb"))


def load_var(name, path=None):
    path = _get_path(path)
    return pickle.load(open(path / (name + '.pkl'), "rb"))


def exist_var(name, path=None):
    path = _get_path(path)
    return os.path.exists(path / (name + '.pkl'))


def clear_var(name, path=None):
    path = _get_path(path)
    os.remove(path / (name + '.pkl'))


def auto_create(name, func, refresh_cache=False, path=None):
    if refresh_cache and exist_var(name, path):
        logger.warning("clear existed cache for {}".format(name))
        clear_var(name, path)
    if exist_var(name, path):
        logger.warning("cache for {} exists".format(name))
        with time_record("load {} from cache".format(name)):
            obj = load_var(name, path)
    else:
        logger.warning("cache for {} does not exist".format(name))
        with time_record("create {} and save to cache".format(name)):
            obj = func()
            save_var(obj, name, path)
    return obj


def shutdown_logging(repo_name):
    for key, logger in logging.root.manager.loggerDict.items():
        if isinstance(key, str) and key.startswith(repo_name):
            logging.getLogger(key).setLevel(logging.ERROR)


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


def each_caller_run_once(f):
    def wrapper(*args, **kwargs):
        if args[0] not in wrapper.callers:
            wrapper.callers.add(args[0])
            return f(*args, **kwargs)
    wrapper.callers = set()
    return wrapper


@contextmanager
def numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextmanager
def torch_seed(seed):
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)


def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property


def check_os(platform):
    if platform == 'win' and is_win():
        allow = True
    elif platform == 'unix' and is_unix():
        allow = True
    else:
        allow = False
    if allow:

        def inner(func):
            return func

        return inner
    else:
        raise Exception("only support {}".format(platform))


def is_win():
    return psutil.WINDOWS


def is_unix():
    return psutil.LINUX | psutil.MACOS


def time_stamp():
    return arrow.now().format('MMMDD_HH-mm-ss')


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


def show_mem():
    top = psutil.Process(os.getpid())
    info = top.memory_full_info()
    memory = info.uss / 1024. / 1024.
    print('Memory: {:.2f} MB'.format(memory))


def wait_for_debug():
    if "DBGPY" not in globals():
        import debugpy
        globals()["DBGPY"]=1
        debugpy.listen(("127.0.0.1", 5678))
        debugpy.wait_for_client()
        debugpy.breakpoint()

# def deprecated(message: str = ''):
#     """
#     This is a decorator which can be used to mark functions
#     as deprecated. It will result in a warning being emitted
#     when the function is used first time and filter is set for show DeprecationWarning.
#     """
#     def decorator_wrapper(func):
#         @functools.wraps(func)
#         def function_wrapper(*args, **kwargs):
#             current_call_source = '|'.join(
#                 traceback.format_stack(inspect.currentframe()))
#             if current_call_source not in function_wrapper.last_call_source:
#                 warnings.warn("Function {} is now deprecated! {}".format(
#                     func.__name__, message),
#                               category=DeprecationWarning,
#                               stacklevel=2)
#                 function_wrapper.last_call_source.add(current_call_source)
#             return func(*args, **kwargs)
#         function_wrapper.last_call_source = set()
#         return function_wrapper
#     return decorator_wrapper