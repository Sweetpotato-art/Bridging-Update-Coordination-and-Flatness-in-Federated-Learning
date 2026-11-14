"""Local shared SGD followed by sample-weighted endpoint averaging, without Adam at the server."""
import copy
import math

import numpy as np
import torch

from fl_coordination._impl.fedmini.engine import Engine as EvaluationEngine
from fl_coordination._impl.fedmini.gradients import client_gradients
from fl_coordination._impl.fedmini.model import build_network, buffers, flatten
from fl_coordination._impl.fedpairs.common import finite, fingerprint_state
from fl_coordination._impl.fedseries.engine import pair_geometry


def method_label(coordinator, variant):
    if (coordinator, variant) != ('fedavg', 'baseline'):
        raise ValueError('This entry implements only the plain FedAvg A3 baseline')
    return 'FedAvg'


class Engine(EvaluationEngine):
    """Share trainable encoder parameters; keep heads and BN buffers local, as in A3.

    n_i is the size of client i's complete fitting subset, excluding validation.
    w_next = sum_i n_i / sum_j n_j * w_i_after_local_SGD.
    Private heads retain A3's local Adam. There is no shared/server optimizer state.
    """
    def __init__(self, config, coordinator, variant, device, network=None):
        method_label(coordinator, variant)
        if (config['case'], config['clients'], config['participants']) != ('A3', 5, 5):
            raise ValueError('FedAvg A3 requires all five clients')
        if config['optimizer'] != 'none' or config['private_optimizer'] != 'adam':
            raise ValueError('FedAvg uses parameter averaging and A3 private-head Adam')
        counts = config['client_fit_sizes']
        if len(counts) != 5 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in counts):
            raise ValueError('Positive per-client fitting sample counts are required')
        if config['weight_decay'] != 0 or not math.isfinite(config['local_learning_rate']) or config['local_learning_rate'] <= 0:
            raise ValueError('A3 local shared SGD requires positive lr and zero weight decay')
        self.config, self.device, self.flat, self.variant = config, device, False, variant
        self.weights = [n / sum(counts) for n in counts]
        self.encoder, self.heads = build_network(5) if network is None else network
        self.encoder.to(device); self.heads.to(device)
        self.local_buffers = [buffers(self.encoder) for _ in self.heads]
        self.head_optimizers = [torch.optim.Adam(h.parameters(), lr=config['learning_rate'], weight_decay=0.)
                                for h in self.heads]
        self.step_index = 0
        self.initial_hash = fingerprint_state(dict(**{'encoder.'+k:v for k,v in self.encoder.state_dict().items()},
                                                  **{'heads.'+k:v for k,v in self.heads.state_dict().items()}))

    def step(self, client_batches):
        lengths = {len(b) for b in client_batches.values()}
        if sorted(client_batches) != list(range(5)) or len(lengths) != 1:
            raise ValueError('A3 requires all clients and equal local block lengths')
        k = next(iter(lengths))
        if not 1 <= k <= self.config['local_steps']:
            raise ValueError('Local block length exceeds the A3 budget')
        shared = list(self.encoder.parameters())
        base = [p.detach().clone() for p in shared]
        merged = [torch.zeros_like(p) for p in shared]
        eta = self.config['local_learning_rate']
        uploads = []
        try:
            for cid in range(5):
                with torch.no_grad():
                    for parameter, saved in zip(shared, base):
                        parameter.copy_(saved)
                total, losses, counts = None, [], []
                for batch in client_batches[cid]:
                    upload = client_gradients(self.encoder, self.heads[cid], batch,
                                              self.local_buffers[cid], 0., self.config['adaptive_sam'])
                    self.local_buffers[cid] = upload.buffers
                    total = upload.g.clone() if total is None else total.add_(upload.g)
                    with torch.no_grad():
                        offset = 0
                        for parameter in shared:
                            n = parameter.numel()
                            parameter.add_(upload.g[offset:offset+n].view_as(parameter), alpha=-eta)
                            offset += n
                    optimizer = self.head_optimizers[cid]
                    optimizer.zero_grad(set_to_none=True)
                    for parameter, gradient in zip(self.heads[cid].parameters(), upload.head):
                        parameter.grad = gradient.clone()
                    optimizer.step()
                    counts.append(len(batch[1])); losses.append(upload.loss)
                # Average actual local endpoints, not normalized gradients or Adam steps.
                with torch.no_grad():
                    for output, parameter in zip(merged, shared):
                        finite(parameter, 'FedAvg local endpoint')
                        output.add_(parameter.detach(), alpha=self.weights[cid])
                actual = flatten([b - p.detach() for b,p in zip(base,shared)])
                expected = eta * total
                error = float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
                uploads.append(dict(g=total/k, samples=sum(counts),
                                    loss=float(np.average(losses, weights=counts)), error=error))
        finally:
            with torch.no_grad():
                for parameter, saved in zip(shared, base):
                    parameter.copy_(saved)
        for value in merged:
            finite(value, 'FedAvg averaged endpoint')
        for parameter in self.heads.parameters():
            finite(parameter, 'FedAvg private head')
        with torch.no_grad():
            for parameter, value in zip(shared, merged):
                parameter.copy_(value)
        self.step_index += 1
        actual_delta = flatten([b-p.detach() for b,p in zip(base,shared)])
        gs = [u['g'] for u in uploads]
        mean_g = sum(w*g for w,g in zip(self.weights,gs))
        norm = float(mean_g.norm())
        vector_bytes = actual_delta.numel()*actual_delta.element_size()
        return dict(round=self.step_index, client_ids=list(range(5)),
                    client_steps={str(c):k for c in range(5)},
                    client_samples={str(c):u['samples'] for c,u in enumerate(uploads)},
                    samples=sum(u['samples'] for u in uploads), local_steps=5*k, backward_passes=5*k,
                    losses=[u['loss'] for u in uploads], sharpness=[0.]*5,
                    uplink_vector_bytes=5*vector_bytes, downlink_parameter_bytes=5*vector_bytes,
                    g_norm=norm, h_norm=0., fused_norm=norm, branch_cosine=None,
                    client_g_geometry=pair_geometry(gs),
                    actual_server_step_norm=float(actual_delta.norm()),
                    reconstruction_error=0., parameter_reconstruction_relative_error=max(u['error'] for u in uploads),
                    qp_g=dict(solver='fedavg_sample_weighted_parameters',weights=list(self.weights),
                              sample_counts=list(self.config['client_fit_sizes']),server_optimizer='none'),
                    qp_h=dict(solver='inactive'))

    def state_dict(self):
        return dict(model=self.model_state(), step=self.step_index, initial_hash=self.initial_hash,
                    weights=list(self.weights),
                    head_optimizers=[copy.deepcopy(o.state_dict()) for o in self.head_optimizers])

    def load_state_dict(self, value):
        if value['weights'] != self.weights or len(value['head_optimizers']) != len(self.heads):
            raise ValueError('FedAvg checkpoint client weights/head count changed')
        self.load_model(value['model'])
        for optimizer, state in zip(self.head_optimizers, value['head_optimizers']):
            optimizer.load_state_dict(state)
        self.step_index, self.initial_hash = value['step'], value['initial_hash']
