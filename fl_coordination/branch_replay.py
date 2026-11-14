"""Collect scalar diagnostics from paired A3 client-gradient replay."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from fl_coordination._impl.feda3.engine import Engine
from fl_coordination._impl.fedmini.coordinators import FedFV
from fl_coordination._impl.fedmini.runtime import seed_all
from fl_coordination._impl.fedpairs.common import load
from fl_coordination._impl.fedseries.data import Data, epoch_plan


class CapturedBranches(Exception):
    """Stop the forward replay before the shared optimizer step."""


def capture(engine, state, batches, step):
    engine.load_model(state['model'])
    for optimizer, saved in zip(engine.head_optimizers, state['head_optimizers']):
        optimizer.load_state_dict(saved)
    engine.step_index = step - 1
    captured = {}

    def capture_g(vectors, losses, step, ids):
        captured.update(g=vectors, losses=losses, ids=ids)
        return torch.zeros_like(vectors[0]), {}

    def capture_h(vectors, losses, step, ids):
        assert captured['ids'] == ids
        captured['h'] = vectors
        raise CapturedBranches

    engine.g_coordinator = capture_g
    engine.h_coordinator = capture_h
    try:
        engine.step(batches)
    except CapturedBranches:
        return captured
    raise RuntimeError('Split branch capture did not run')


def diagnostics(v, config, step, device):
    lam = config['flat_lambda']
    gs, hs = torch.stack(v['g']).double(), torch.stack(v['h']).double()
    mixed = gs + lam * hs
    norms = gs.norm(dim=1) + lam * hs.norm(dim=1)
    scale = norms[:, None] * norms[None, :]
    assert (scale > 0).all()
    x = (gs @ gs.T + lam**2 * (hs @ hs.T)) / scale
    y = (mixed @ mixed.T) / scale
    pairs = []
    tol = 1e-7
    for i, j in zip(*np.triu_indices(len(gs), k=1)):
        a, b = float(x[i, j]), float(y[i, j])
        tag = ('near_zero' if min(abs(a), abs(b)) <= tol else
               'resolved' if a < 0 < b else 'introduced' if b < 0 < a else 'same')
        pairs.append(dict(client_i=v['ids'][i], client_j=v['ids'][j], x=a, y=b, tag=tag))

    def project(vectors):
        coordinator = FedFV(config, device)
        out, _ = coordinator(list(vectors), v['losses'], step, ids=v['ids'])
        return out

    def compare(a, b):
        delta = float((a-b).norm() / ((a.norm()+b.norm()) / 2))
        cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0).clamp(-1, 1))
        return dict(relative_l2=delta, angle_deg=math.degrees(math.acos(cosine)))

    linear = compare(mixed.mean(0), gs.mean(0) + lam*hs.mean(0))
    if linear['relative_l2'] > 1e-12:
        raise ValueError('Fixed equal-weight aggregation failed the linearity control')
    direct = project(mixed)
    split = project(gs) + lam*project(hs)
    fv = compare(direct, split)
    gains = [dict(client=cid, direct=float(a), split=float(b))
             for cid, a, b in zip(v['ids'], gs @ (direct / direct.norm()),
                                 gs @ (split / split.norm()))]
    return dict(step=step, pairs=pairs, linear=linear, fedfv=fv, gains=gains)



def run(result_dir, checkpoint, data_root, device):
    meta = json.loads((Path(result_dir)/'metadata.json').read_text(encoding='utf-8'))
    config = meta['config']
    if config['seed'] != 2 or config['case'] != 'A3' or config['flat_lambda'] != 1:
        raise ValueError('Use the seed-2 Fashion-MNIST ASAM separate-coordination run.')
    seed_all(2, True)
    ck = load(Path(checkpoint))
    if ck['identity_sha256'] != meta['identity_sha256'] or ck['epoch'] != 200:
        raise ValueError('The epoch-200 checkpoint must match the source run.')
    data = Data.load(data_root, config)
    if data.digest != meta['split_sha256']:
        raise ValueError('Data split differs from the training run.')
    engine = Engine(config, 'fedawa', 'split', device)
    plan = epoch_plan(data, config, ck['epoch'])
    output = []
    for offset, block in enumerate(plan, 1):
        step = ck['engine']['step'] + offset
        batches = {cid: [data.batch('fit', idx, device) for idx in chunks]
                   for cid, chunks in block.items()}
        branches = capture(engine, ck['engine'], batches, step)
        output.append(diagnostics(branches, config, step, device))
    return dict(seed=2, checkpoint_epoch=200, split=data.digest, replays=output)
