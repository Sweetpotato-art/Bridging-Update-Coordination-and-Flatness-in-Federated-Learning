"""Small standard-library helpers, usable before PyTorch is installed."""
import contextlib
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEAM = ROOT / 'fl_coordination' / 'configs'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


from fl_coordination._impl.fedmini.runtime import run_lock as lock
