"""Atomic HTTPS download and SHA256 validation of compressed AND raw Fashion data."""
import gzip
import os
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

from .common import TEAM, digest, lock, read, write

MIRRORS = ('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
           'https://fashion-mnist.s3.eu-central-1.amazonaws.com/')
DEFAULT_ROOT = Path('datasets/fashion-mnist')


def valid(path, spec):
    return path.is_file() and path.stat().st_size == spec['size'] and digest(path) == spec['sha256']


def download(filename, output, spec):
    errors = []
    for mirror in MIRRORS:
        for attempt in range(3):
            fd, name = tempfile.mkstemp(prefix=filename+'.', suffix='.part', dir=output.parent)
            os.close(fd)
            temp = Path(name)
            try:
                request = urllib.request.Request(mirror+filename, headers={'User-Agent': 'coordination_flatness-data/1'})
                with urllib.request.urlopen(request, timeout=45) as src, temp.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024*1024)
                if not valid(temp, spec):
                    raise RuntimeError('Downloaded archive has the wrong size or SHA256')
                os.replace(temp, output)
                return
            except Exception as exc:
                errors.append(f'{mirror} attempt {attempt+1}: {exc}')
                if attempt < 2:
                    time.sleep(1+attempt)
            finally:
                temp.unlink(missing_ok=True)
    raise RuntimeError('Fashion-MNIST download failed; rerun to retry.\n'+'\n'.join(errors))


def ensure(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    specs = read(TEAM/'fashion_files.json')
    raw = root/'FashionMNIST'/'raw'
    raw.mkdir(parents=True, exist_ok=True)
    with lock(root/'.data.lock'):
        for name, spec in specs.items():
            if not name.endswith('.gz'):
                continue
            archive = raw/name
            if not valid(archive, spec):
                print('Fetching verified Fashion-MNIST: '+name, flush=True)
                download(name, archive, spec)
            target = raw/name[:-3]
            if not valid(target, specs[target.name]):
                fd, temporary = tempfile.mkstemp(prefix=target.name+'.', suffix='.part', dir=raw)
                os.close(fd)
                temp = Path(temporary)
                try:
                    with gzip.open(archive, 'rb') as src, temp.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    if not valid(temp, specs[target.name]):
                        raise RuntimeError('Extracted Fashion data differs from the original seed-2 data')
                    os.replace(temp, target)
                finally:
                    temp.unlink(missing_ok=True)
        # Torchvision may otherwise accept stale legacy processed/*.pt files.
        # Do not delete user files or silently use an unchecked alternate dataset.
        if (root/'FashionMNIST'/'processed'/'training.pt').exists():
            raise RuntimeError('Legacy processed/training.pt exists; use a fresh --data-root for verified raw data')
        result = dict(dataset='Fashion-MNIST', root=str(root), files=specs, verified=True)
        write(root/'data_verified.json', result)
        return result
