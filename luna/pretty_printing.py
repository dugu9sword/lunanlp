from colorama import Fore, Back
from tabulate import tabulate
import inspect


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


def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [
            var_name for var_name, var_val in fi.frame.f_locals.items()
            if var_val is var
        ]
        if len(names) > 0:
            return names[0]


def print_as_table(obj):
    """
    Examples:

        >>> succ_num = 914
        >>> fail_num = 123
        >>> print_as_table([succ_num, fail_num])
        --------  ---
        succ_num  914
        fail_num  123
        --------  ---
    """
    if isinstance(obj, list):
        print(tabulate([[retrieve_name(ele), ele] for ele in obj]))
    elif isinstance(obj, dict):
        print(tabulate([[k, v] for k, v in obj.items()]))


def print_num_list(lst, fmt=":6.2f"):
    fmter = "{{{}}}".format(fmt)
    str_lst = [fmter.format(ele) for ele in lst]
    print(" ".join(str_lst))
    