import copy
import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def moved(value, device='cpu'):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: moved(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(moved(v, device) for v in value)
    return copy.deepcopy(value)


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def preserve_rng():
    state = rng()
    try:
        yield
    finally:
        set_rng(state)


def flat_dict(state):
    return torch.cat([v.reshape(-1) for v in state.values()])


def gradients(model):
    return {k: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
            for k, p in model.named_parameters()}


def perturb(model, g, rho):
    norm = flat_dict(g).norm()
    if float(norm) <= 1e-12:
        return False
    with torch.no_grad():
        for k, p in model.named_parameters():
            p.add_(g[k], alpha=rho / float(norm))
    return True


def finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(name + ' is nonfinite; no solver fallback or silent normalization is applied')


def diagnostics(dg, dh, lam, sharpness):
    dg, dh = dg.reshape(-1), dh.reshape(-1)
    d = dg + lam * dh
    finite(d, 'Fused direction')
    a, b = float(dg.norm()), float(dh.norm())
    return dict(g_direction_norm=a, h_direction_norm=b, fused_direction_norm=float(d.norm()),
                branch_cosine=float(dg @ dh / (a*b)) if a*b > 1e-20 else None,
                mean_sam_loss_increment=float(np.mean(sharpness)),
                negative_sam_loss_increments=int(np.sum(np.asarray(sharpness) < 0)))


def fingerprint_state(state):
    h = hashlib.sha256()
    for name, v in state.items():
        h.update(name.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
