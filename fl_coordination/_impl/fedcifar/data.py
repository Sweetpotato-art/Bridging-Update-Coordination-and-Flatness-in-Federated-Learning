"""Disjoint class-Dirichlet clients, train-only validation, stateless augmentation."""
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR100

from fl_coordination._impl.fedmini.data import Data as BaseData, fingerprint


def apportioned_counts(total, probabilities):
    """Largest-remainder integer allocation; no sample is dropped or duplicated."""
    real = total * np.asarray(probabilities, dtype=np.float64)
    counts = np.floor(real).astype(np.int64)
    missing = total - int(counts.sum())
    order = np.argsort(-(real - counts), kind='stable')
    counts[order[:missing]] += 1
    if counts.sum() != total:
        raise RuntimeError('Invalid class allocation')
    return counts


def partition(train_labels, test_labels, config):
    n, classes = config['clients'], config['num_classes']
    seed, alpha = config['seed'], config['dirichlet_alpha']
    if n < 2 or alpha <= 0 or not 0 < config['validation_fraction'] < .5:
        raise ValueError('Invalid partition configuration')
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    if any(set(np.unique(labels)) != set(range(classes)) for labels in (train_labels, test_labels)):
        raise ValueError('All labels must use the same complete class vocabulary')
    weights = np.random.default_rng(np.random.SeedSequence([seed, 2101])).dirichlet(
        np.full(n, alpha), size=classes)
    train_rng = np.random.default_rng(np.random.SeedSequence([seed, 2102]))
    test_rng = np.random.default_rng(np.random.SeedSequence([seed, 2103]))
    result = {s: [[] for _ in range(n)] for s in ('fit', 'validation', 'test')}
    for label in range(classes):
        train = train_rng.permutation(np.flatnonzero(train_labels == label))
        test = test_rng.permutation(np.flatnonzero(test_labels == label))
        nval = round(len(train) * config['validation_fraction'])
        if not 0 < nval < len(train):
            raise ValueError('Each class needs both fitting and held-out examples')
        pools = {'fit': train[nval:], 'validation': train[:nval], 'test': test}
        for split, pool in pools.items():
            counts = apportioned_counts(len(pool), weights[label])
            pieces = np.split(pool, np.cumsum(counts)[:-1])
            for cid, indices in enumerate(pieces):
                result[split][cid].extend(indices.tolist())
    if any(not ids for clients in result.values() for ids in clients):
        raise ValueError('Partition produced an empty client split; use an explicit new configuration')
    return result, weights


class Data(BaseData):
    def __init__(self, train_images, train_labels, test_images, test_labels, config):
        self.config = config
        self.images = {'fit': train_images, 'validation': train_images, 'test': test_images}
        self.labels = {'fit': train_labels, 'validation': train_labels, 'test': test_labels}
        self.indices, weights = partition(train_labels.numpy(), test_labels.numpy(), config)
        self.fit_sizes = [len(ids) for ids in self.indices['fit']]
        self.steps_per_client = [math.ceil(n / config['batch_size']) for n in self.fit_sizes]
        self.samples_per_epoch = sum(self.fit_sizes)
        digest = hashlib.sha256()
        for tensor in (train_images, train_labels, test_images, test_labels):
            digest.update(tensor.contiguous().numpy().tobytes())
        self.manifest = dict(
            version='cifar100-class-dirichlet-v1', data_sha256=digest.hexdigest(),
            seed=config['seed'], clients=config['clients'], num_classes=config['num_classes'],
            dirichlet_alpha=config['dirichlet_alpha'], validation_fraction=config['validation_fraction'],
            class_client_probabilities=weights.tolist(), allocation='largest_remainder',
            validation='class-stratified holdout from official training split before client allocation',
            test='official test split; same class-client probabilities, disjoint client ownership',
            indices=self.indices,
            counts={s: [len(ids) for ids in groups] for s, groups in self.indices.items()},
            label_histograms={s: [torch.bincount(self.labels[s][ids], minlength=config['num_classes']).tolist()
                                  for ids in groups] for s, groups in self.indices.items()})
        self.digest = fingerprint(self.manifest)

    @classmethod
    def load(cls, root, config, download=False):
        train = CIFAR100(str(root), train=True, download=download)
        test = CIFAR100(str(root), train=False, download=download)
        if len(train) != 50000 or len(test) != 10000 or train.data.shape[1:] != (32, 32, 3):
            raise ValueError('Expected official CIFAR-100 50,000/10,000 RGB images')
        return cls(torch.from_numpy(train.data), torch.tensor(train.targets, dtype=torch.long),
                   torch.from_numpy(test.data), torch.tensor(test.targets, dtype=torch.long), config)

    def batch(self, split, indices, device, *, epoch=None, cid=None):
        ids = np.asarray(indices, dtype=np.int64)
        if not len(ids):
            raise ValueError('Empty mini-batch')
        index = torch.from_numpy(ids.copy())
        x = self.images[split][index].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.
        if split == 'fit' and self.config['augmentation'] != 'none':
            if epoch is None or cid is None:
                raise ValueError('Training augmentation needs explicit epoch and client ID')
            # Separate RNG: extra solver/backward calls cannot change augmentation.
            gen = np.random.default_rng(np.random.SeedSequence(
                [self.config['seed'], 2201, epoch, cid, int(ids[0]), len(ids)]))
            offsets = gen.integers(0, 9, size=(len(ids), 2))
            flip = torch.as_tensor(gen.random(len(ids)) < .5, device=device)
            padded = F.pad(x, (4, 4, 4, 4), mode='constant', value=0)
            h, w = x.shape[-2:]
            rows = torch.as_tensor(offsets[:, 0], device=device)[:, None, None] + torch.arange(h, device=device)[None, :, None]
            cols = torch.as_tensor(offsets[:, 1], device=device)[:, None, None] + torch.arange(w, device=device)[None, None, :]
            samples = torch.arange(len(ids), device=device)[:, None, None]
            x = padded.permute(0, 2, 3, 1)[samples, rows, cols].permute(0, 3, 1, 2)
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        mean = x.new_tensor(self.config['normalization_mean']).view(1, 3, 1, 1)
        std = x.new_tensor(self.config['normalization_std']).view(1, 3, 1, 1)
        return (x - mean) / std, self.labels[split][index].to(device)


def epoch_plan(data, config, epoch):
    """All clients participate; each fitting example is used exactly once.

    Unequal client sizes require unequal K_i. Spread each client's mini-batches
    across the same number of rounds, with 1 <= K_i <= the declared maximum.
    This is an explicit scheduling adaptation, not fixed-K FedAvg.
    """
    if config['participants'] != config['clients']:
        raise ValueError('This recipe is full participation')
    batches = data.epoch_indices(epoch)
    rounds = max(math.ceil(len(b) / config['local_steps']) for b in batches)
    if min(map(len, batches)) < rounds:
        raise ValueError('Full participation/exact coverage/K limit cannot all hold for this split; '
                         'choose a new batch size or partition explicitly, never repeat data')
    groups = []
    for client in batches:
        cuts = np.array_split(np.arange(len(client)), rounds)
        groups.append([[client[int(i)] for i in part] for part in cuts])
    return [{cid: groups[cid][r] for cid in range(config['clients'])} for r in range(rounds)]


def plan_summary(data, config, epoch=0):
    plan = epoch_plan(data, config, epoch)
    n = config['clients']
    steps, examples, visits = [0]*n, [0]*n, [0]*n
    digest, lengths = hashlib.sha256(), {}
    for block in plan:
        if sorted(block) != list(range(n)):
            raise RuntimeError('Missing client in a full-participation round')
        for cid, batches in block.items():
            visits[cid] += 1
            steps[cid] += len(batches)
            if not 1 <= len(batches) <= config['local_steps']:
                raise RuntimeError('Local step limit violated')
            lengths[str(len(batches))] = lengths.get(str(len(batches)), 0) + 1
            digest.update(np.asarray([cid, len(batches)], dtype='<i8').tobytes())
            for ids in batches:
                examples[cid] += len(ids)
                digest.update(np.asarray([len(ids)], dtype='<i8').tobytes())
                digest.update(np.asarray(ids, dtype='<i8').tobytes())
    if steps != data.steps_per_client or examples != data.fit_sizes:
        raise RuntimeError('Plan does not exactly cover all client training samples')
    return dict(rounds=len(plan), client_steps=steps, client_samples=examples,
                client_participations=visits, participants_histogram={str(n): len(plan)},
                local_steps_histogram=lengths, sha256=digest.hexdigest())
