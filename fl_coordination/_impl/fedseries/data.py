"""Five disjoint clients, persistent local heads, and matched per-client coverage."""
import hashlib
import math

import numpy as np
import torch

from fl_coordination._impl.fedmini.data import Data as MiniData, fingerprint


def partition(train_labels, test_labels, seed=2, val_fraction=.1, majority=.8):
    """Allocate each class to one preferred client and four other clients.

    In the paper configuration, the preferred client receives 80% of its
    preferred class samples, and each other client receives 5%.
    """
    if not 0 < val_fraction < .5 or not .2 < majority < 1:
        raise ValueError('Invalid validation or preferred-class fraction')
    result = {name: [[] for _ in range(5)] for name in ('fit', 'validation', 'test')}
    for source, labels in [('train', np.asarray(train_labels)), ('test', np.asarray(test_labels))]:
        if set(np.unique(labels)) != set(range(10)):
            raise ValueError('Expected all ten Fashion-MNIST classes')
        gen = np.random.default_rng(np.random.SeedSequence([seed, 1101 if source == 'train' else 1202]))
        for label in range(10):
            indices = gen.permutation(np.flatnonzero(labels == label))
            owner = label // 2
            owner_count = round(len(indices) * majority)
            others = [cid for cid in range(5) if cid != owner]
            pieces = {owner: indices[:owner_count]}
            pieces.update(zip(others, np.array_split(indices[owner_count:], 4)))
            for cid, local in pieces.items():
                if source == 'train':
                    nval = round(len(local) * val_fraction)
                    if nval < 1 or len(local) - nval < 1:
                        raise ValueError('Each client/class needs training and validation data')
                    result['validation'][cid].extend(local[:nval].tolist())
                    result['fit'][cid].extend(local[nval:].tolist())
                else:
                    if not len(local):
                        raise ValueError('Each client/class needs test data')
                    result['test'][cid].extend(local.tolist())
    return result


class Data(MiniData):
    def __init__(self, train_images, train_labels, test_images, test_labels, config):
        if config['clients'] != 5:
            raise ValueError('Fashion-MNIST requires five clients')
        self.images = {'fit': train_images, 'validation': train_images, 'test': test_images}
        self.labels = {'fit': train_labels, 'validation': train_labels, 'test': test_labels}
        self.config = config
        self.indices = partition(train_labels.numpy(), test_labels.numpy(), config['seed'],
                                 config['validation_fraction'], config['majority_fraction'])
        self.fit_sizes = [len(ids) for ids in self.indices['fit']]
        self.steps_per_client = [math.ceil(n / config['batch_size']) for n in self.fit_sizes]
        if len(set(self.fit_sizes)) != 1:
            raise ValueError('Fashion-MNIST requires equal fitting-set sizes')
        self.steps_per_epoch = self.steps_per_client[0]
        self.samples_per_epoch = sum(self.fit_sizes)
        digest = hashlib.sha256()
        for tensor in (train_images, train_labels, test_images, test_labels):
            digest.update(tensor.contiguous().numpy().tobytes())
        self.manifest = dict(version='series-a-five-client-v1', data_sha256=digest.hexdigest(),
                            seed=config['seed'], preferred_classes=[[2*c, 2*c+1] for c in range(5)],
                            validation_fraction=config['validation_fraction'],
                            majority_fraction=config['majority_fraction'], indices=self.indices,
                            counts={name: [len(ids) for ids in groups] for name, groups in self.indices.items()},
                            label_histograms={name: [torch.bincount(self.labels[name][ids], minlength=10).tolist()
                                                     for ids in groups] for name, groups in self.indices.items()})
        self.digest = fingerprint(self.manifest)


def epoch_plan(data, config, epoch):
    """Group each client's fitting batches into local steps for one data epoch.

    When selecting a subset of clients, prioritize those with more remaining
    blocks and break ties using a seeded generator. Keep partial final blocks.
    """
    batches = data.epoch_indices(epoch)
    k, m = config['local_steps'], config['participants']
    if not 1 <= m <= config['clients'] or k < 1:
        raise ValueError('Invalid participation count or local-step count')
    blocks = [[b[i:i+k] for i in range(0, len(b), k)] for b in batches]
    positions = [0] * config['clients']
    gen = np.random.default_rng(np.random.SeedSequence([config['seed'], 1404, epoch, m, k]))
    plan = []
    while True:
        available = [c for c, b in enumerate(blocks) if positions[c] < len(b)]
        if not available:
            break
        permutation = gen.permutation(available).tolist()
        ordered = sorted(permutation, key=lambda c: -(len(blocks[c]) - positions[c]))
        ids = sorted(ordered[:m])
        plan.append({c: blocks[c][positions[c]] for c in ids})
        for c in ids:
            positions[c] += 1
    return plan


def plan_summary(data, config, epoch=0):
    plan = epoch_plan(data, config, epoch)
    steps, examples, visits = [0]*5, [0]*5, [0]*5
    digest = hashlib.sha256()
    sizes, local_lengths = {}, {}
    for block in plan:
        sizes[str(len(block))] = sizes.get(str(len(block)), 0) + 1
        for cid, batches in block.items():
            visits[cid] += 1
            steps[cid] += len(batches)
            local_lengths[str(len(batches))] = local_lengths.get(str(len(batches)), 0) + 1
            digest.update(np.asarray([cid, len(batches)], dtype='<i8').tobytes())
            for ids in batches:
                examples[cid] += len(ids)
                digest.update(np.asarray([len(ids)], dtype='<i8').tobytes())
                digest.update(np.asarray(ids, dtype='<i8').tobytes())
    if examples != data.fit_sizes or steps != data.steps_per_client:
        raise RuntimeError('Plan does not cover each client exactly once')
    return dict(rounds=len(plan), client_steps=steps, client_samples=examples,
                client_participations=visits, participants_histogram=sizes,
                local_steps_histogram=local_lengths, sha256=digest.hexdigest())
