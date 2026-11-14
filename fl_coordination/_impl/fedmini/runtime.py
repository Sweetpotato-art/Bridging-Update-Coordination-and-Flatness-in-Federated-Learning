"""Complete sequential runs, validation selection, isolated weights and exact resume."""
import contextlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from fl_coordination._impl.fedpairs.common import json_write, save, load, moved, rng, set_rng
from .data import fingerprint

METHODS = [('fedfv', False, 'FedFV'), ('fedfv', True, 'F-FedFV'),
           ('fedmgda', False, 'FedMGDA+'), ('fedmgda', True, 'F-FedMGDA+')]


def seed_all(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


@contextlib.contextmanager
def run_lock(path):
    """OS lock releases on process exit, including crashes; no stale PID lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            raise RuntimeError('This method is already running: ' + str(path)) from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_history(path, history):
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                 for row in history), encoding='utf-8')
    os.replace(temporary, path)
