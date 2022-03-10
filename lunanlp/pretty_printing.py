import torch
from colorama import Back, Fore


class Color(object):
    @staticmethod
    def red(s):
        return Fore.RED + str(s) + Fore.RESET

    @staticmethod
    def green(s):
        return Fore.GREEN + str(s) + Fore.RESET

    @staticmethod
    def yellow(s):
        return Fore.YELLOW + str(s) + Fore.RESET

    @staticmethod
    def blue(s):
        return Fore.BLUE + str(s) + Fore.RESET

    @staticmethod
    def magenta(s):
        return Fore.MAGENTA + str(s) + Fore.RESET

    @staticmethod
    def cyan(s):
        return Fore.CYAN + str(s) + Fore.RESET

    @staticmethod
    def white(s):
        return Fore.WHITE + str(s) + Fore.RESET

    @staticmethod
    def white_green(s):
        return Fore.WHITE + Back.GREEN + str(s) + Fore.RESET + Back.RESET


def print_tensor_dict(tensor_dict):
    """
    >>> dct = {
    >>>     "net_input": {
    >>>         "src_tokens": torch.tensor([[0, 1, 2], [3, 4, 5]]),
    >>>         "src_lengths": [2, 2],
    >>>         "mask": {
    >>>             "attn_mask": torch.tensor([True, False]),
    >>>             "padding_mask": torch.tensor([True, False]),
    >>>         }
    >>>     },
    >>>     "meta": list(range(100)),
    >>>     "target": torch.tensor([[0, 1, 2], [3, 4, 5]])
    >>> }
    >>> print_tensor_dict(dct)
    """
    indent_size = 4

    def prettyformat(k, v, indent=0):
        if isinstance(v, dict):
            if k is not None:
                _ret = ["{}{}:".format(indent * " ", k)]
            else:
                _ret = []
            indent += indent_size
            for ele in v:
                _ret.extend(prettyformat(ele, v[ele], indent))
            indent -= indent_size
            return _ret
        elif isinstance(v, torch.Tensor):
            v_str = "{}, {}".format(type(v), v.shape).replace("torch.", "")
            return ["{}{}: {}".format(indent * " ", k, v_str)]
        else:
            v_str = "{}".format(v)
            if len(v_str) > 50:
                v_str = "{}, {}...".format(type(v), v_str[:50])
            return ["{}{}: {}".format(indent * " ", k, v_str)]

    print("\n".join(prettyformat(None, tensor_dict, 0)))
