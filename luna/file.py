import os
from pathlib import Path
import pickle
from contextlib import contextmanager
import time
import re
import logging
from .public import load_var, save_var, exist_var

logger = logging.getLogger(__name__)



def _safe_readline(f):
    """
    Adopted from fairseq/binarizer.py
    """
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def safe_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    return folder_path


def safe_file(file_path):
    head, _ = os.path.split(file_path)
    safe_folder(head)
    return file_path


def head(filename, n=10):
    lines = []
    with open(file=filename) as f:
        for _ in range(n):
            line = f.readline()
            if line == '':
                break
            lines.append(line)
    return lines


def tail(filename, n=10):
    lines = []
    BLOCK_SIZE = 1024 * n
    with open(file=filename) as f:
        f.seek(0, os.SEEK_END)
        filesize = f.tell()
        pos = filesize - BLOCK_SIZE
        while True:
            pos = max(0, pos)
            f.seek(pos)
            _safe_readline(f)
            lines = f.readlines()
            if len(lines) >= n:
                break
            else:
                if pos == 0:
                    break
                else:
                    pos -= BLOCK_SIZE
        return lines[-n:]


class RandomAccessFile:
    def __init__(self, filename) -> None:
        raf_cache_name = "raf@" + re.sub(r"[:./\\]", "_", filename)
        last_modified = Path(filename).stat().st_mtime
        self.offsets = None
        self.fp = open(filename)
        # try to read from cache
        if exist_var(raf_cache_name):
            logger.warning(f"RandomAccessFile: load cache for {filename}")
            raf_cache = load_var(raf_cache_name)
            if raf_cache['last_modified'] == last_modified:
                self.offsets = raf_cache['offsets']
            else:
                logger.warning(f"RandomAccessFile: cache for {filename} is dirty, refresh it")

        if self.offsets is None:
            self.offsets = self._gen_offsets()
            raf_cache = {
                "last_modified": last_modified,
                "offsets": self.offsets
            }
            save_var(raf_cache, raf_cache_name)

    def _gen_offsets(self):
        offsets = []
        while True:
            offsets.append(self.fp.tell())
            line = self.fp.readline()
            if line == '':
                break
        return offsets

    def __getitem__(self, key):
        self.fp.seek(self.offsets[key])
        line = self.fp.readline()
        return line

    def __len__(self):
        return len(self.offsets)
