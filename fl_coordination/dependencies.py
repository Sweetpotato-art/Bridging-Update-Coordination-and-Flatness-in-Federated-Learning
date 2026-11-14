"""Retrieve only the author files used by published numerical adapters."""
import hashlib
import json
import os
import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path

from .recipes import CONFIGS


@lru_cache(maxsize=None)
def author_file(name):
    record = json.loads((CONFIGS/'upstreams.json').read_text(encoding='utf-8'))[name]
    cache = Path(os.environ.get('FL_AUTHOR_CACHE', Path.home()/'.cache'/'fl_coordination'/'authors'))
    target = cache / record['sha256'] / name

    def verified(raw):
        return hashlib.sha256(raw).hexdigest() == record['sha256']

    if target.is_file() and verified(target.read_bytes()):
        return target
    with urllib.request.urlopen(record['url'], timeout=60) as response:
        raw = response.read()
    if not verified(raw):
        raise RuntimeError(f'Author source checksum mismatch for {name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
