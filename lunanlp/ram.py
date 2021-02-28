"""
The RAM system is used to conveniently create globally temporary values in
any place of a code.

The values to store in a RAM have the below features:
    - Users do not want to **declare it explicitly** in the program, which
        makes the code rather dirty.
    - Users want to **share** it across functions, or even files.
    - Users use it **temporarily**, such as for debugging
    - Users want to **reuse** a group of values several times, while **reset** each
        value in the group before reusing them will add a great overhead to the code.
"""

__memory = {}


def list_keys():
    return sorted(list(__memory.keys()))


def write(k, v):
    __memory[k] = v


def pop(k):
    return __memory.pop(k)


def append(k, v):
    if k not in __memory:
        __memory[k] = []
    __memory[k].append(v)


def inc(k):
    if k not in __memory:
        __memory[k] = 0
    __memory[k] = __memory[k] + 1


def read(k):
    return __memory[k]


def has(k):
    return k in __memory


def flag_name(k):
    return f"RAM_FLAG_{k}"


def set_flag(k):
    write(flag_name(k), True)


def reset_flag(k):
    if has(flag_name(k)):
        pop(flag_name(k))


def has_flag(k, verbose_once=False):
    ret = has(flag_name(k)) and read(flag_name(k)) is True
    if verbose_once and not has_flag(f"VERBOSE_ONCE_{flag_name(k)}"):
        print(
            f"INFO: check the flag {k}={ret}, the information only occurs once."
        )
        set_flag(f"VERBOSE_ONCE_{flag_name(k)}")
    return ret


def globalize(name=None):
    if name is None:

        def wrapper(fun):
            if fun.__name__ in __memory:
                raise Exception("{} already in ram.".format(fun.__name__))
            __memory[fun.__name__] = fun
            return fun
    else:

        def wrapper(fun):
            if name in __memory:
                raise Exception("{} already in ram.".format(name))
            __memory[name] = fun
            return fun

    return wrapper


def reset(prefix=None):
    if prefix is not None:
        to_reset = []
        for key in __memory:
            if key.startswith(prefix):
                to_reset.append(key)
        for key in to_reset:
            __memory.pop(key)
    else:
        __memory.clear()
