"""One synchronized shared update and one local-head update per mini-batch."""
import copy

import numpy as np
import torch
from torch.nn import functional as F

from fl_coordination._impl.fedpairs.common import moved, finite, fingerprint_state
from .coordinators import make_coordinator
from .gradients import client_gradients
from .model import build_network, buffers, restore_buffers, assign_gradients, evaluating, clone_state


class Engine:
    def __init__(self, config, coordinator, flat, device, network=None):
        self.config, self.flat, self.device = config, flat, device
        self.encoder, self.heads = build_network(config['clients']) if network is None else network
        self.encoder.to(device)
        self.heads.to(device)
        self.local_buffers = [buffers(self.encoder) for _ in self.heads]
        if config['optimizer'] != 'adam':
            raise ValueError('The published Fashion-MNIST recipe pins Adam; a different optimizer needs a new recipe')
        options = dict(lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.shared_optimizer = torch.optim.Adam(self.encoder.parameters(), **options)
        self.head_optimizers = [torch.optim.Adam(head.parameters(), **options) for head in self.heads]
        self.g_coordinator = make_coordinator(coordinator, config, device)
        self.h_coordinator = make_coordinator(coordinator, config, device)
        self.step_index = 0
        self.initial_hash = fingerprint_state(dict(**{'encoder.' + k: v for k, v in self.encoder.state_dict().items()},
                                                  **{'heads.' + k: v for k, v in self.heads.state_dict().items()}))

    def step(self, batches):
        if len(batches) != len(self.heads):
            raise ValueError('Every client must contribute one batch')
        rho = self.config['rho'] if self.flat else 0.
        uploads = [client_gradients(self.encoder, head, batch, state, rho, self.config['adaptive_sam'])
                   for head, batch, state in zip(self.heads, batches, self.local_buffers)]
        self.local_buffers = [u.buffers for u in uploads]
        self.step_index += 1
        g, qp_g = self.g_coordinator([u.g for u in uploads], [u.loss for u in uploads], self.step_index)
        h = torch.zeros_like(g)
        qp_h = {'solver': 'inactive'}
        if self.flat and rho > 0 and self.config['flat_lambda'] != 0:
            h, qp_h = self.h_coordinator([u.h for u in uploads], [u.sharpness for u in uploads], self.step_index)
        lam = self.config['flat_lambda'] if self.flat else 0.
        direction = g + lam * h
        finite(direction, 'Fused shared direction')
        self.shared_optimizer.zero_grad(set_to_none=True)
        assign_gradients(self.encoder.parameters(), direction)
        self.shared_optimizer.step()
        for head, optimizer, upload in zip(self.heads, self.head_optimizers, uploads):
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(head.parameters(), upload.head):
                parameter.grad = gradient.clone()
            optimizer.step()
        # Catch the very first nonfinite optimizer update instead of writing bad weights.
        for root in [self.encoder, self.heads]:
            if not all(bool(torch.isfinite(p).all()) for p in root.parameters()):
                raise FloatingPointError('Optimizer produced nonfinite parameters')
        gnorm, hnorm = float(g.norm()), float(h.norm())
        pair_denom = float(uploads[0].g.norm() * uploads[1].g.norm())
        pair_dot = float(uploads[0].g @ uploads[1].g)
        return dict(step=self.step_index, losses=[u.loss for u in uploads],
                    batch_sizes=[len(y) for _, y in batches],
                    g_norm=gnorm, h_norm=hnorm, fused_norm=float(direction.norm()),
                    branch_cosine=float(g @ h) / (gnorm * hnorm) if gnorm * hnorm else None,
                    client_g_cosine=pair_dot / pair_denom if pair_denom else None,
                    client_g_conflict=int(pair_dot < 0),
                    g_dot_direction=[float(u.g @ direction) for u in uploads],
                    sharpness=[u.sharpness for u in uploads],
                    raw_g_norms=[float(u.g.norm()) for u in uploads],
                    raw_h_norms=[float(u.h.norm()) for u in uploads],
                    reconstruction_error=max(u.reconstruction_error for u in uploads),
                    qp_g=qp_g, qp_h=qp_h)

    @torch.no_grad()
    def evaluate(self, data, split):
        values = []
        with evaluating(self.encoder, self.heads):
            for cid, head in enumerate(self.heads):
                restore_buffers(self.encoder, self.local_buffers[cid])
                n, correct, loss = 0, 0, 0.
                for x, y in data.evaluation_batches(split, cid, self.device):
                    logits = head(self.encoder(x))
                    finite(logits, 'Evaluation logits')
                    correct += int((logits.argmax(1) == y).sum())
                    loss += float(F.cross_entropy(logits, y, reduction='sum'))
                    n += len(y)
                if n == 0:
                    raise ValueError('Empty evaluation client')
                values.append(dict(accuracy=100. * correct / n, loss=loss / n, samples=n))
        accuracies = [v['accuracy'] for v in values]
        return dict(clients=values, mean=float(np.mean(accuracies)), worst=min(accuracies),
                    pooled=sum(v['accuracy'] * v['samples'] for v in values) / sum(v['samples'] for v in values))

    def model_state(self):
        return clone_state(self.encoder, self.heads, self.local_buffers)

    def load_model(self, value):
        self.encoder.load_state_dict(value['encoder'])
        self.heads.load_state_dict(value['heads'])
        self.local_buffers = moved(value['local_buffers'], self.device)

    def state_dict(self):
        return dict(model=self.model_state(), step=self.step_index, initial_hash=self.initial_hash,
                    shared_optimizer=copy.deepcopy(self.shared_optimizer.state_dict()),
                    head_optimizers=[copy.deepcopy(o.state_dict()) for o in self.head_optimizers],
                    g_coordinator=copy.deepcopy(self.g_coordinator.state_dict()),
                    h_coordinator=copy.deepcopy(self.h_coordinator.state_dict()))

    def load_state_dict(self, value):
        self.load_model(value['model'])
        self.shared_optimizer.load_state_dict(value['shared_optimizer'])
        for optimizer, state in zip(self.head_optimizers, value['head_optimizers']):
            optimizer.load_state_dict(state)
        self.g_coordinator.load_state_dict(moved(value['g_coordinator'], self.device))
        self.h_coordinator.load_state_dict(moved(value['h_coordinator'], self.device))
        self.step_index = value['step']
        self.initial_hash = value['initial_hash']
