"""Sample-weighted local endpoints with private heads and local BN statistics."""
import copy
import math

import numpy as np
import torch

from fl_coordination._impl.fedcifar.engine import NoPrivateOptimizer
from fl_coordination._impl.fedcifar.model import build_network
from fl_coordination._impl.fedmini.engine import Engine as EvaluationEngine
from fl_coordination._impl.fedmini.gradients import client_gradients
from fl_coordination._impl.fedmini.model import buffers, flatten
from fl_coordination._impl.fedpairs.common import finite, fingerprint_state
from fl_coordination._impl.fedseries.engine import pair_geometry


def method_label(coordinator, variant, config=None):
    if (coordinator, variant) != ('fedavg', 'baseline'):
        raise ValueError('This entry runs only the plain CIFAR-100 FedAvg baseline')
    return 'FedAvg'


display_label = method_label


class Engine(EvaluationEngine):
    def __init__(self, config, coordinator, variant, device, network=None):
        method_label(coordinator, variant)
        n = config['clients']
        counts = config.get('client_fit_sizes', [])
        if n < 2 or config['participants'] != n or len(counts) != n or any(
                type(v) is not int or v <= 0 for v in counts):
            raise ValueError('FedAvg requires full participation and positive complete fit-set sizes')
        if config['optimizer'] != 'none' or config['private_optimizer'] != 'adam':
            raise ValueError('FedAvg uses endpoint averaging and private-head Adam; no server optimizer')
        if config['head_mode'] not in ('private','shared') or config['bn_policy'] != 'client_local_running_statistics':
            raise ValueError('Invalid classifier or BN ownership')
        if config['rho'] != 0 or config['flat_lambda'] != 0 or config['weight_decay'] != 0:
            raise ValueError('Plain FedAvg has no SAM, branch fusion, or weight decay')
        for key in ('learning_rate','local_learning_rate'):
            if not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError('Learning rates must be positive and finite')
        self.config, self.device, self.flat, self.variant = config, device, False, variant
        self.weights = [v/sum(counts) for v in counts]
        self.encoder, self.heads = network if network is not None else build_network(
            n,config['head_mode'],config['num_classes'])
        if len(self.heads) != n:
            raise ValueError('Head count differs from client count')
        self.encoder.to(device); self.heads.to(device)
        self.local_buffers = [buffers(self.encoder) for _ in self.heads]
        self.head_optimizers = []
        for head in self.heads:
            parameters = list(head.parameters())
            if bool(parameters) != (config['head_mode'] == 'private'):
                raise ValueError('Classifier parameters are in the wrong ownership group')
            self.head_optimizers.append(torch.optim.Adam(parameters,lr=config['learning_rate'],weight_decay=0.)
                                        if parameters else NoPrivateOptimizer())
        self.step_index = 0
        self.initial_hash = fingerprint_state({
            **{'encoder.'+k:v for k,v in self.encoder.state_dict().items()},
            **{'heads.'+k:v for k,v in self.heads.state_dict().items()}})

    def step(self, client_batches):
        n = self.config['clients']
        if sorted(client_batches) != list(range(n)) or any(
                not 1 <= len(b) <= self.config['local_steps'] for b in client_batches.values()):
            raise ValueError('Every client must contribute 1..K batches; K_i may differ')
        shared = list(self.encoder.parameters())
        base = [p.detach().clone() for p in shared]
        merged = [torch.zeros_like(p) for p in shared]
        eta = self.config['local_learning_rate']
        uploads = []
        try:
            for cid in range(n):
                with torch.no_grad():
                    for parameter,saved in zip(shared,base):
                        parameter.copy_(saved)
                total, losses, counts = None, [], []
                for batch in client_batches[cid]:
                    upload = client_gradients(self.encoder,self.heads[cid],batch,
                                              self.local_buffers[cid],rho=0.,adaptive=False)
                    self.local_buffers[cid] = upload.buffers
                    total = upload.g.clone() if total is None else total.add_(upload.g)
                    with torch.no_grad():
                        offset = 0
                        for parameter in shared:
                            size = parameter.numel()
                            parameter.add_(upload.g[offset:offset+size].view_as(parameter),alpha=-eta)
                            offset += size
                    optimizer = self.head_optimizers[cid]
                    optimizer.zero_grad(set_to_none=True)
                    for parameter,gradient in zip(self.heads[cid].parameters(),upload.head):
                        parameter.grad = gradient.clone()
                    optimizer.step()
                    counts.append(len(batch[1])); losses.append(upload.loss)
                # Endpoint averaging is independent of local K_i and tail batch size.
                # The weight is n_i / sum(n_j), using complete fitting subsets only.
                with torch.no_grad():
                    for output,parameter in zip(merged,shared):
                        finite(parameter,'FedAvg local endpoint')
                        output.add_(parameter.detach(),alpha=self.weights[cid])
                actual = flatten([b-p.detach() for b,p in zip(base,shared)])
                expected = eta*total
                error = float((actual-expected).norm()/expected.norm().clamp_min(1e-12))
                k = len(client_batches[cid])
                uploads.append(dict(g=total/k,steps=k,samples=sum(counts),
                                    loss=float(np.average(losses,weights=counts)),error=error))
        finally:
            with torch.no_grad():
                for parameter,saved in zip(shared,base):
                    parameter.copy_(saved)
        for value in merged:
            finite(value,'FedAvg averaged endpoint')
        for parameter in self.heads.parameters():
            finite(parameter,'FedAvg private head')
        with torch.no_grad():
            for parameter,value in zip(shared,merged):
                parameter.copy_(value)
        self.step_index += 1
        delta = flatten([b-p.detach() for b,p in zip(base,shared)])
        gs = [u['g'] for u in uploads]
        norm = float(sum(w*g for w,g in zip(self.weights,gs)).norm())  # diagnostic only
        vector_bytes = delta.numel()*delta.element_size()
        local_steps = sum(u['steps'] for u in uploads)
        return dict(round=self.step_index,client_ids=list(range(n)),
                    client_steps={str(c):u['steps'] for c,u in enumerate(uploads)},
                    client_samples={str(c):u['samples'] for c,u in enumerate(uploads)},
                    samples=sum(u['samples'] for u in uploads),local_steps=local_steps,
                    backward_passes=local_steps,losses=[u['loss'] for u in uploads],sharpness=[0.]*n,
                    uplink_vector_bytes=n*vector_bytes,downlink_parameter_bytes=n*vector_bytes,
                    g_norm=norm,h_norm=0.,fused_norm=norm,branch_cosine=None,
                    client_g_geometry=pair_geometry(gs),actual_server_step_norm=float(delta.norm()),
                    reconstruction_error=0.,parameter_reconstruction_relative_error=max(u['error'] for u in uploads),
                    qp_g=dict(solver='fedavg_sample_weighted_parameters',weights=list(self.weights),
                              sample_counts=list(self.config['client_fit_sizes']),server_optimizer='none'),
                    qp_h=dict(solver='inactive'))

    def state_dict(self):
        return dict(model=self.model_state(),step=self.step_index,initial_hash=self.initial_hash,
                    weights=list(self.weights),head_optimizers=[copy.deepcopy(o.state_dict()) for o in self.head_optimizers])

    def load_state_dict(self, value):
        if value['weights'] != self.weights or len(value['head_optimizers']) != len(self.heads):
            raise ValueError('FedAvg checkpoint sample weights or classifier ownership changed')
        self.load_model(value['model'])
        for optimizer,state in zip(self.head_optimizers,value['head_optimizers']):
            optimizer.load_state_dict(state)
        self.step_index, self.initial_hash = value['step'],value['initial_hash']
