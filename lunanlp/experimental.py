import importlib
import sys
from contextlib import contextmanager


@contextmanager
def safe_import(folder, module_name):
    """
        Example:
        >>> with safe_import("foo", "transformers"):
        >>> # import transformers
        >>> pass
        >>>  
        >>> with safe_import("foo", "transformers"):
        >>>     import another_file
        >>>     sys.modules.pop("another_file")
        >>> 
        >>> with safe_import("bar", "transformers"):
        >>>     import transformers
    """
    print(f"Safe import from {folder} ...")
    sys.path.insert(0, folder)
    importlib.import_module(module_name)

    yield

    sys.path.remove(folder)
    keys = []
    for key in sys.modules:
        if key.startswith(module_name):
            keys.append(key)
    for key in keys:
        sys.modules.pop(key)
