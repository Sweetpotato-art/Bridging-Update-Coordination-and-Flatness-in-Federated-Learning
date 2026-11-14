"""Author client ownership, global patient isolation, and lazy full local epochs."""
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import hashlib
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from fl_coordination._impl.fedmini.data import fingerprint
from fl_coordination._impl.fedpairs.common import json_write
from .prepare import sha256


def key_seed(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, values)).encode()).digest()[:8], 'little')


def patient_id(name):
    match = re.fullmatch(r'(\d+)_(left|right)(?:_test)?', Path(name).stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError('Cannot establish patient identity from author filename: '+name)
    return match.group(1)


def safe_name(value):
    name = value.strip().replace('\\', '/')
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or ':' in name:
        raise ValueError('Unsafe image name')
    return p.as_posix()


def rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        values = [r for r in csv.reader(f) if r and r[0].strip()]
    if values and values[0][0].lower() in ('image', 'filename', 'path', 'image_id'):
        values = values[1:]
    return values


def read_names(path):
    values = [safe_name(r[0]) for r in rows(path)]
    if len(set(values)) != len(values): raise ValueError('Repeated entry in '+str(path))
    return sorted(values)


def rgb_uint8(array, *, normalized_float=None):
    x = np.asarray(array)
    if x.ndim == 2: x = np.repeat(x[..., None], 3, axis=2)
    if x.ndim != 3 or x.shape[-1] not in (1, 3, 4):
        raise ValueError('Expected HWC or grayscale author image')
    if x.shape[-1] == 1: x = np.repeat(x, 3, axis=2)
    x = x[..., :3]
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 255:
        raise ValueError('Unsupported Retina pixel range')
    if normalized_float is True and x.max() > 1: raise ValueError('Normalized image exceeds 1')
    if np.issubdtype(x.dtype, np.floating) and (normalized_float is True or (normalized_float is None and x.max() <= 1)):
        x = x*255.
    return np.rint(x).astype(np.uint8)


class ClientBatches:
    """Re-iterable CPU index plan; at most the current batch is moved to the GPU."""
    def __init__(self, data, cid, round_index, device, limit=None):
        self.data, self.cid, self.round, self.device = data, cid, round_index, device
        gen = np.random.default_rng(key_seed(data.config['seed'], 'batches', round_index, cid))
        self.ids = gen.permutation(data.indices['fit'][cid]).tolist()
        self.chunks = [self.ids[i:i+data.config['batch_size']]
                       for i in range(0, len(self.ids), data.config['batch_size'])]
        if limit is not None: self.chunks = self.chunks[:limit]

    def __len__(self): return len(self.chunks)

    def __iter__(self):
        for ids in self.chunks:
            yield self.data.batch('fit', ids, self.device, round_index=self.round, cid=self.cid)


class Data:
    def __init__(self, root, config, *, strict_counts=True):
        self.root, self.config = Path(root).resolve(), config
        if config['local_epochs'] != 1: raise ValueError('This protocol uses one complete local epoch')
        if config['participants'] != config['clients']: raise ValueError('Retina pilot uses all clients')
        if config['author_float_pixel_range'] != '0_255': raise ValueError('Author release uses 0-255 pixels')
        self.paths, self.records = [], []
        labels = {}
        for row in rows(self.root/'labels.csv'):
            name = safe_name(row[0]); value = float(row[1])
            if value not in (0, 1) or name in labels: raise ValueError('Invalid/duplicate binary label')
            labels[name] = int(value)
        folder = self.root/f'{config["clients"]}_clients'/f'split_{config["author_split"]}'
        files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith('.'))
        if len(files) != config['clients']: raise ValueError('Incorrect author client partition count')
        self.client_names = [p.name for p in files]
        groups = [read_names(p) for p in files]
        owned = [name for group in groups for name in group]
        if len(set(owned)) != len(owned): raise ValueError('Author clients share an image')
        public = read_names(self.root/'test.csv')
        if set(public) & set(owned): raise ValueError('Author training and public test images overlap')
        author_train = read_names(self.root/'train.csv') if (self.root/'train.csv').exists() else owned
        if not set(owned).issubset(author_train): raise ValueError('Client image is absent from author train.csv')
        unassigned = sorted(set(author_train)-set(owned))
        # The release contains 8,999 train images, despite the paper's rounded
        # 9,000 description. Preserve all released images and original owners.
        if strict_counts and (len(author_train), len(owned), len(public), len(unassigned)) != (8999, 8999, 3000, 0):
            raise ValueError(f'Author sample counts changed: {len(owned)} / {len(public)}')
        for phase, name, cid in [('train', n, c) for c,g in enumerate(groups) for n in g]+[('test', n, -1) for n in public]:
            path = (self.root/phase/name).resolve()
            if not path.is_relative_to(self.root): raise ValueError('Image symlink escapes data root')
            if name not in labels or not path.is_file(): raise ValueError('Missing author image or label: '+name)
            self.paths.append(path)
            self.records.append(dict(name=name, phase=phase, cid=cid, patient=patient_id(name),
                                     label=labels[name], sha256=sha256(path)))
        self.targets = np.asarray([r['label'] for r in self.records], dtype=np.int64)
        # Link patients that share an identical source file. This prevents duplicate
        # images with different patient IDs from leaking across split boundaries.
        parents = {r['patient']:r['patient'] for r in self.records}
        def find(x):
            while parents[x] != x:
                parents[x] = parents[parents[x]]; x = parents[x]
            return x
        hashes = {}
        for rec in self.records:
            old = hashes.get(rec['sha256'])
            if old is not None:
                if old['label'] != rec['label']: raise ValueError('Identical image has conflicting labels')
                a,b = find(old['patient']),find(rec['patient']); parents[max(a,b)] = min(a,b)
            hashes[rec['sha256']] = rec
        for rec in self.records: rec['group'] = find(rec['patient'])
        public_groups = {r['group'] for r in self.records if r['phase']=='test'}
        eligible = [i for i,r in enumerate(self.records) if r['phase']=='train' and r['group'] not in public_groups]
        excluded = [i for i,r in enumerate(self.records) if r['phase']=='train' and r['group'] in public_groups]
        labels_by_group = {}
        for i in eligible:
            r = self.records[i]
            labels_by_group[r['group']] = max(labels_by_group.get(r['group'], 0), r['label'])
        phase_by_group = {}
        for label in (0, 1):
            group_ids = sorted(p for p,l in labels_by_group.items() if l==label)
            gen = np.random.default_rng(key_seed(config['seed'], 'patient-split', label))
            group_ids = [group_ids[i] for i in gen.permutation(len(group_ids))]
            nv = max(1, round(len(group_ids)*config['validation_fraction']))
            nt = max(1, round(len(group_ids)*config['test_fraction']))
            if nv+nt >= len(group_ids): raise ValueError('Too few patient groups for independent holdouts')
            for phase, ids in [('validation',group_ids[:nv]), ('test',group_ids[nv:nv+nt]), ('fit',group_ids[nv+nt:])]:
                phase_by_group.update({p:phase for p in ids})
        self.indices = {s:[[] for _ in files] for s in ('fit','validation','test')}
        for idx in eligible:
            r = self.records[idx]; self.indices[phase_by_group[r['group']]][r['cid']].append(idx)
        self.public_indices = [i for i,r in enumerate(self.records) if r['phase']=='test']
        for split, groups in self.indices.items():
            for cid, ids in enumerate(groups):
                available = {r['label'] for r in self.records if r['phase']=='train' and r['cid']==cid}
                if not ids or set(self.targets[ids]) != available:
                    raise ValueError(f'{split} client {cid} lost an available class; review the predeclared patient split')
        self.fit_sizes = [len(g) for g in self.indices['fit']]
        self.steps_per_client = [math.ceil(n/config['batch_size']) for n in self.fit_sizes]
        self.manifest = dict(protocol=config['protocol'], author_split=config['author_split'], seed=config['seed'],
                             records=self.records, client_names=self.client_names, indices=self.indices,
                             public_indices=self.public_indices, excluded_public_overlap=excluded,
                             counts={s:[len(g) for g in groups] for s,groups in self.indices.items()},
                             label_histograms={s:[np.bincount(self.targets[g],minlength=2).tolist() for g in groups]
                                               for s,groups in self.indices.items()},
                             patient_counts={s:[len({self.records[i]['patient'] for i in g}) for g in groups]
                                             for s,groups in self.indices.items()},
                             author_counts=dict(train_csv=len(author_train),assigned_train=len(owned),public_test=len(public)),
                             unassigned_author_train=unassigned,
                             source_files={p.relative_to(self.root).as_posix():sha256(p) for p in
                                           [self.root/'labels.csv',self.root/'test.csv',*files,*([self.root/'train.csv'] if (self.root/'train.csv').exists() else [])]},
                             public_overlap_policy='Exclude all original-training patient/duplicate groups present in public test',
                             patient_split_policy='Conservative numeric patient ID across eye and optional original-source _test suffix; global stratified holdouts; preserve author ownership',
                             proportions=dict(validation=config['validation_fraction'],test=config['test_fraction']))
        self.digest = fingerprint(self.manifest)
        self.pool = ThreadPoolExecutor(max_workers=config.get('io_workers',4))

    def close(self): self.pool.shutdown(wait=True); self._raw.cache_clear()

    @lru_cache(maxsize=256)
    def _raw(self, idx):
        path = self.paths[idx]
        if path.suffix.lower()=='.npy':
            # Fixed release-level scale: do not brighten a dark 0-255 image
            # merely because this particular image happens to have max <= 1.
            return rgb_uint8(np.load(path, allow_pickle=False), normalized_float=False)
        with Image.open(path) as im: return np.asarray(im.convert('RGB')).copy()

    def _image(self, item):
        split, idx, round_index, cid = item
        image = Image.fromarray(self._raw(idx)); size = self.config['image_size']
        if split=='fit':
            if round_index is None or cid is None: raise ValueError('Training augmentation needs round/client identity')
            gen = np.random.default_rng(key_seed(self.config['seed'],'augment',round_index,cid,self.records[idx]['name']))
            width,height = image.size; area = width*height
            for _ in range(10):
                target = area*gen.uniform(*self.config['crop_scale'])
                ratio = math.exp(gen.uniform(math.log(.75),math.log(4/3)))
                w,h = round(math.sqrt(target*ratio)),round(math.sqrt(target/ratio))
                if 0<w<=width and 0<h<=height:
                    left,top=int(gen.integers(width-w+1)),int(gen.integers(height-h+1)); break
            else:
                w,h=width,height
                if width/height<.75: h=round(width/.75)
                elif width/height>4/3: w=round(height*4/3)
                left,top=(width-w)//2,(height-h)//2
            image = TF.resized_crop(image,top,left,h,w,[size,size],InterpolationMode.BILINEAR)
            image = TF.rotate(image,float(gen.uniform(-self.config['rotation_degrees'],self.config['rotation_degrees'])),InterpolationMode.BILINEAR)
            if gen.random()<.5: image=TF.hflip(image)
        else:
            image=TF.center_crop(TF.resize(image,size,InterpolationMode.BILINEAR),[size,size])
        return np.asarray(image,dtype=np.uint8).copy()

    def batch(self, split, ids, device, *, round_index=None, cid=None):
        ids = [int(i) for i in ids]
        x = np.stack(list(self.pool.map(self._image,[(split,i,round_index,cid) for i in ids])))
        x = torch.from_numpy(x).permute(0,3,1,2).contiguous().to(device=device,dtype=torch.float32)/255.
        mean=x.new_tensor(self.config['normalization_mean']).view(1,3,1,1)
        std=x.new_tensor(self.config['normalization_std']).view(1,3,1,1)
        return (x-mean)/std,torch.as_tensor(self.targets[ids].copy(),dtype=torch.long,device=device)

    def local_batches(self, round_index, device, *, limit=None):
        return {cid:ClientBatches(self,cid,round_index,device,limit) for cid in range(self.config['clients'])}

    def evaluation_batches(self, split, cid, device):
        if split=='public_test': ids=self.public_indices
        elif split in ('validation','test'): ids=self.indices[split][cid]
        else: raise ValueError('Invalid evaluation split')
        for start in range(0,len(ids),self.config['eval_batch_size']):
            yield self.batch(split,ids[start:start+self.config['eval_batch_size']],device)

    def save_manifest(self, path): json_write(path,self.manifest)

    def budget(self):
        return dict(rounds=self.config['rounds'],local_epochs=1,clients_per_round=self.config['clients'],
                    fit_samples=sum(self.fit_sizes),client_samples=self.fit_sizes,client_steps=self.steps_per_client,
                    training_sample_visits=self.config['rounds']*sum(self.fit_sizes),
                    local_batches=self.config['rounds']*sum(self.steps_per_client),
                    full_data_epochs=self.config['rounds'])
