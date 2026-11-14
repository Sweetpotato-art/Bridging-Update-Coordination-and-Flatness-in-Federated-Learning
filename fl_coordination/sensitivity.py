"""Measure validation-loss changes along the shared-backbone ascent direction."""
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from fl_coordination._impl.fedmini.model import restore_buffers
from fl_coordination._impl.fedpairs.common import load

RADII = tuple(n / 2000 for n in range(11))


def probe(dataset, result_dir, checkpoint, data_root, device='cpu'):
    from fl_coordination._impl.fedmini.runtime import seed_all

    folder = Path(result_dir)
    meta = json.loads((folder / 'metadata.json').read_text())
    config = meta['config']
    seed_all(config['seed'], config.get('deterministic', True))
    if dataset == 'fashion':
        from fl_coordination._impl.fedmini.model import build_network
        from fl_coordination._impl.fedseries.data import Data

        encoder, heads = build_network(config['clients'])
        root = data_root
    elif dataset == 'cifar':
        from fl_coordination._impl.fedcifar.model import build_network
        from fl_coordination._impl.fedcifar.data import Data

        encoder, heads = build_network(config['clients'], config['head_mode'], config['num_classes'])
        root = data_root
    else:
        raise ValueError(dataset)

    records = [json.loads(row) for row in (folder / 'metrics.jsonl').read_text().splitlines() if row.strip()]
    reference = records[-1]
    saved = load(Path(checkpoint))
    if saved['identity_sha256'] != meta['identity_sha256'] or saved['epoch'] != reference['epoch']:
        raise ValueError('Checkpoint does not match the final log')
    if reference['epoch'] != config['epochs']:
        raise ValueError('Training log is incomplete')
    state = saved['model'] if 'model' in saved else saved['engine']['model']
    encoder.load_state_dict(state['encoder'])
    heads.load_state_dict(state['heads'])
    encoder.to(device).eval()
    heads.to(device).eval().requires_grad_(False)
    local_buffers = [{key: value.to(device) for key, value in group.items()}
                     for group in state['local_buffers']]
    data = Data.load(root, config)
    if data.digest != meta['split_sha256']:
        raise ValueError('Evaluation split does not match the run')
    batches = [list(data.evaluation_batches('validation', cid, device))
               for cid in range(config['clients'])]
    parameters = list(encoder.parameters())
    originals = [p.detach().clone() for p in parameters]

    def loss(backward=False):
        values = []
        for cid, chunks in enumerate(batches):
            restore_buffers(encoder, local_buffers[cid])
            count = len(data.indices['validation'][cid])
            total = 0.
            for images, labels in chunks:
                term = F.cross_entropy(heads[cid](encoder(images)), labels, reduction='sum') / count
                if backward:
                    (term / len(batches)).backward()
                total += float(term.detach())
            values.append(total)
        return values

    clean = loss(backward=True)
    logged = [row['loss'] for row in reference['validation']['clients']]
    error = max(abs(a - b) for a, b in zip(clean, logged))
    if len(clean) != len(logged) or not math.isfinite(error) or error > 1e-4:
        raise ValueError(f'Clean loss mismatch by client: {clean} vs {logged}')
    gradient_norm = math.sqrt(sum(float(p.grad.square().sum()) for p in parameters))
    parameter_norm = math.sqrt(sum(float(p.detach().square().sum()) for p in parameters))
    if not (math.isfinite(gradient_norm) and gradient_norm > 0 and math.isfinite(parameter_norm)):
        raise ValueError('Non-finite norm or zero gradient')
    clean_mean = sum(clean) / len(clean)
    points = [dict(radius=0., delta=0.)]
    with torch.no_grad():
        for radius in RADII[1:]:
            scale = radius * parameter_norm / gradient_norm
            for p, original in zip(parameters, originals):
                p.copy_(original).add_(p.grad, alpha=scale)
            perturbed = loss()
            delta = sum(perturbed) / len(perturbed) - clean_mean
            if not math.isfinite(delta):
                raise ValueError('Non-finite perturbed loss')
            points.append(dict(radius=radius, delta=delta))
    return dict(dataset=dataset, identity=meta['identity_sha256'], checkpoint=str(checkpoint),
                split=data.digest, seed=config['seed'], epoch=reference['epoch'], device=str(device),
                clean_loss=clean_mean, max_clean_client_error=error, gradient_norm=gradient_norm,
                parameter_norm=parameter_norm, points=points)
