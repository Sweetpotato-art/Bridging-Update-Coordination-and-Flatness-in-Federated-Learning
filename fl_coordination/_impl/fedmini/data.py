"""Deterministic, disjoint, same-label-space two-client Fashion-MNIST split."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST

from fl_coordination._impl.fedpairs.common import json_write


class HTTPSFashionMNIST(FashionMNIST):
    # Keep torchvision's MD5 integrity checks; use the data owner's HTTPS mirror.
    mirrors = ['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
               'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/']

    @property
    def raw_folder(self):
        return str(Path(self.root) / 'FashionMNIST' / 'raw')

    @property
    def processed_folder(self):
        return str(Path(self.root) / 'FashionMNIST' / 'processed')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def partition(train_labels, test_labels, seed=2, val_fraction=.1, majority=.8):
    """Even classes favor client 0; odd classes favor client 1; both keep all 10."""
    if not 0 < val_fraction < .5 or not .5 <= majority < 1:
        raise ValueError('Invalid validation fraction or complementary class proportion')
    result = {split: [[], []] for split in ['fit', 'validation', 'test']}
    for source, labels in [('train', np.asarray(train_labels)), ('test', np.asarray(test_labels))]:
        if set(np.unique(labels)) != set(range(10)):
            raise ValueError('Expected the same ten labels 0..9')
        gen = np.random.default_rng(np.random.SeedSequence([seed, 101 if source == 'train' else 202]))
        for label in range(10):
            indices = gen.permutation(np.flatnonzero(labels == label))
            cut = round(len(indices) * (majority if label % 2 == 0 else 1. - majority))
            for cid, local in enumerate([indices[:cut], indices[cut:]]):
                if source == 'train':
                    nval = round(len(local) * val_fraction)
                    if nval < 1 or len(local) - nval < 1:
                        raise ValueError('Every client/class needs fitting and validation samples')
                    result['validation'][cid].extend(local[:nval].tolist())
                    result['fit'][cid].extend(local[nval:].tolist())
                else:
                    result['test'][cid].extend(local.tolist())
    return result


class Data:
    def __init__(self, train_images, train_labels, test_images, test_labels, config):
        self.images = {'fit': train_images, 'validation': train_images, 'test': test_images}
        self.labels = {'fit': train_labels, 'validation': train_labels, 'test': test_labels}
        self.config = config
        self.indices = partition(train_labels.numpy(), test_labels.numpy(), config['seed'],
                                 config['validation_fraction'], config['majority_fraction'])
        self.steps_per_epoch = math.ceil(len(self.indices['fit'][0]) / config['batch_size'])
        if any(math.ceil(len(ids) / config['batch_size']) != self.steps_per_epoch
               for ids in self.indices['fit']):
            raise ValueError('This full-participation recipe requires equal local batch counts')
        digest = hashlib.sha256()
        for tensor in [train_images, train_labels, test_images, test_labels]:
            digest.update(tensor.contiguous().numpy().tobytes())
        self.manifest = dict(version=1, data_sha256=digest.hexdigest(), seed=config['seed'],
                             validation_fraction=config['validation_fraction'],
                             majority_fraction=config['majority_fraction'], indices=self.indices,
                             counts={split: [len(v) for v in ids] for split, ids in self.indices.items()},
                             label_histograms={split: [torch.bincount(self.labels[split][ids], minlength=10).tolist()
                                                       for ids in clients] for split, clients in self.indices.items()})
        self.digest = fingerprint(self.manifest)

    @classmethod
    def load(cls, root, config, download=False):
        train = HTTPSFashionMNIST(str(root), train=True, download=download)
        test = HTTPSFashionMNIST(str(root), train=False, download=download)
        if len(train) != 60000 or len(test) != 10000:
            raise ValueError('Unexpected Fashion-MNIST size')
        return cls(train.data, train.targets, test.data, test.targets, config)

    def save_manifest(self, path):
        path = Path(path)
        value = dict(self.manifest, manifest_sha256=self.digest)
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise RuntimeError('Existing split manifest differs; choose a new experiment name')
        if not path.exists():
            json_write(path, value)

    def epoch_indices(self, epoch):
        result = []
        for cid, ids in enumerate(self.indices['fit']):
            gen = np.random.default_rng(np.random.SeedSequence([self.config['seed'], 303, epoch, cid]))
            order = gen.permutation(np.array(ids, dtype=np.int64))
            result.append([v for v in np.array_split(order, range(self.config['batch_size'], len(order),
                                                                 self.config['batch_size']))])
        return result

    def batch(self, split, indices, device):
        # Resize once per whole batch on the chosen device; no random augmentations.
        indices = torch.as_tensor(np.asarray(indices).copy(), dtype=torch.long)
        x = self.images[split][indices].to(device=device, dtype=torch.float32).unsqueeze(1) / 255.
        x = F.interpolate(x, size=(self.config['image_size'],) * 2, mode='bilinear',
                          align_corners=False, antialias=True)
        y = self.labels[split][indices].to(device)
        return x, y

    def evaluation_batches(self, split, cid, device):
        ids = self.indices[split][cid]
        size = self.config['eval_batch_size']
        for start in range(0, len(ids), size):
            yield self.batch(split, ids[start:start + size], device)
