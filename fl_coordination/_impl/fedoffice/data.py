"""Domain-owned, class-stratified disjoint splits and stateless image augmentation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from fl_coordination._impl.fedmini.data import Data as BaseData, fingerprint
from fl_coordination._impl.fedcifar.data import epoch_plan, plan_summary
from .prepare import DOMAINS, COUNTS


class Data(BaseData):
    def __init__(self, root, config, *, strict_counts=True):
        self.root, self.config = Path(root).resolve(), config
        self.paths, labels, domains = [], [], []
        classes = sorted(p.name for p in (self.root/DOMAINS[0]).iterdir() if p.is_dir())
        if len(classes) != config['num_classes']:
            raise ValueError('Unexpected class vocabulary')
        content = []
        for cid, domain in enumerate(DOMAINS):
            folder = self.root/domain
            if sorted(p.name for p in folder.iterdir() if p.is_dir()) != classes:
                raise ValueError('All four domains must have identical class names')
            before = len(self.paths)
            for label, name in enumerate(classes):
                images = sorted((folder/name).glob('*.jpg'))
                if len(images) < 5:
                    raise ValueError('Each domain/class needs fitting, validation and test samples')
                for path in images:
                    self.paths.append(path)
                    labels.append(label)
                    domains.append(cid)
                    content.append([path.relative_to(self.root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()])
            if strict_counts and len(self.paths)-before != COUNTS[cid]:
                raise ValueError(f'Wrong complete image count in {domain}')
        self.targets = np.asarray(labels, dtype=np.int64)
        owners = np.asarray(domains, dtype=np.int64)
        self.indices = {s:[[] for _ in DOMAINS] for s in ('fit','validation','test')}
        for cid in range(4):
            gen = np.random.default_rng(np.random.SeedSequence([config['seed'], 4101, cid]))
            for label in range(len(classes)):
                ids = gen.permutation(np.flatnonzero((owners==cid)&(self.targets==label)))
                nv = max(1, round(len(ids)*config['validation_fraction']))
                nt = max(1, round(len(ids)*config['test_fraction']))
                if nv+nt >= len(ids):
                    raise ValueError('No fitting examples after holdout')
                for s, pool in [('validation',ids[:nv]),('test',ids[nv:nv+nt]),('fit',ids[nv+nt:])]:
                    self.indices[s][cid].extend(pool.tolist())
        self.fit_sizes = [len(v) for v in self.indices['fit']]
        self.samples_per_epoch = sum(self.fit_sizes)
        self.steps_per_client = [math.ceil(n/config['batch_size']) for n in self.fit_sizes]
        self.manifest = dict(version='officehome-four-domains-stratified-v1', seed=config['seed'],
                             domain_names=list(DOMAINS), classes=classes, files=content,
                             indices=self.indices, validation_fraction=config['validation_fraction'],
                             test_fraction=config['test_fraction'],
                             counts={s:[len(v) for v in groups] for s,groups in self.indices.items()},
                             label_histograms={s:[np.bincount(self.targets[v],minlength=len(classes)).tolist() for v in groups]
                                               for s,groups in self.indices.items()},
                             evaluation='Held-out images from all four participating domains; not unseen-domain generalization')
        self.digest = fingerprint(self.manifest)
        self.pool = ThreadPoolExecutor(max_workers=config.get('io_workers',4))

    def close(self):
        self.pool.shutdown(wait=True)

    def _image(self, item):
        split, idx, epoch, cid = item
        with Image.open(self.paths[idx]) as source:
            image = source.convert('RGB')
            w,h = image.size
            shorter = self.config['resize_shorter']
            if w <= h:
                newsize = (shorter, int(shorter*h/w))
            else:
                newsize = (int(shorter*w/h), shorter)
            image = image.resize(newsize,Image.Resampling.BILINEAR)
            size = self.config['image_size']
            if split == 'fit':
                if epoch is None or cid is None:
                    raise ValueError('Training augmentation requires epoch and client ID')
                gen = np.random.default_rng(np.random.SeedSequence([self.config['seed'],4201,epoch,cid,idx]))
                left = int(gen.integers(newsize[0]-size+1))
                top = int(gen.integers(newsize[1]-size+1))
                flip = gen.random() < .5
            else:
                left,top=(newsize[0]-size)//2,(newsize[1]-size)//2
                flip=False
            image=image.crop((left,top,left+size,top+size))
            if flip:
                image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return np.array(image,dtype=np.uint8)

    def batch(self, split, indices, device, *, epoch=None, cid=None):
        ids=[int(i) for i in indices]
        if not ids:
            raise ValueError('Empty batch')
        x=np.stack(list(self.pool.map(self._image,[(split,i,epoch,cid) for i in ids])))
        x=torch.from_numpy(x).to(device=device,dtype=torch.float32).permute(0,3,1,2)/255.
        mean=x.new_tensor(self.config['normalization_mean']).view(1,3,1,1)
        std=x.new_tensor(self.config['normalization_std']).view(1,3,1,1)
        return (x-mean)/std,torch.as_tensor(self.targets[ids].copy(),device=device,dtype=torch.long)
