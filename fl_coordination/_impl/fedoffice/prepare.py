"""Download the TLL Office-Home mirror and official torchvision pretrained weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.request

DOMAINS = ('Art', 'Clipart', 'Product', 'Real_World')
COUNTS = (2427, 4365, 4439, 4357)
URLS = {
    'Art': 'https://cloud.tsinghua.edu.cn/f/6a006656b9a14567ade2/?dl=1',
    'Clipart': 'https://cloud.tsinghua.edu.cn/f/ae88aa31d2d7411dad79/?dl=1',
    'Product': 'https://cloud.tsinghua.edu.cn/f/f219b0ff35e142b3ab48/?dl=1',
    'Real_World': 'https://cloud.tsinghua.edu.cn/f/6c19f3f15bb24ed3951a/?dl=1',
}
MIRROR_SOURCE = 'https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/vision/datasets/officehome.py'


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                append = offset > 0 and response.status == 206
                expected = response.headers.get('Content-Length')
                count = 0
                with part.open('ab' if append else 'wb') as stream:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        count += len(block)
                if expected is not None and count != int(expected):
                    raise IOError('Incomplete download')
            part.replace(path)
            return
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2)


def prepare(root, model_root=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for domain, expected in zip(DOMAINS, COUNTS):
        marker = root / f'.{domain}.complete.json'
        if not marker.exists():
            archive = root / 'archives' / f'{domain}.tgz'
            print(f'Download/check {domain}', flush=True)
            download(URLS[domain], archive)
            with tarfile.open(archive, 'r:gz') as tf:
                for member in tf.getmembers():
                    destination = (root / member.name).resolve()
                    if (not destination.is_relative_to(root) or
                            not (member.isfile() or member.isdir())):
                        raise ValueError('Unsafe archive entry: ' + member.name)
                tf.extractall(root)
            files = sorted((root/domain).rglob('*.jpg'))
            if len(files) != expected:
                raise RuntimeError(f'{domain}: expected {expected} JPEGs, found {len(files)}')
            value = {'url': URLS[domain]}
            # The mirror publishes no digest; record actual content provenance, never call it an official checksum.
            h = hashlib.sha256()
            with archive.open('rb') as stream:
                for block in iter(lambda: stream.read(1048576), b''):
                    h.update(block)
            value.update(archive_sha256=h.hexdigest(), count=expected, mirror_source=MIRROR_SOURCE)
            marker.write_text(json.dumps(value, indent=2), encoding='utf-8')
        records[domain] = json.loads(marker.read_text(encoding='utf-8'))
    from torchvision.models import resnet18, ResNet18_Weights
    if model_root is not None:
        os.environ['TORCH_HOME'] = str(Path(model_root).resolve())
    # torchvision verifies the official hash prefix when downloading.
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print(json.dumps({'data_root': str(root), 'model_root':str(model_root), 'domains': records}), flush=True)
