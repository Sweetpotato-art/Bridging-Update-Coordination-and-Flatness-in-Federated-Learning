"""Sample-weighted FedAvg with client-private classifiers."""
import copy
import numpy as np
import torch
from fl_coordination._impl.fedmini.engine import Engine as EvaluationEngine
from fl_coordination._impl.fedmini.gradients import client_gradients
from fl_coordination._impl.fedmini.model import buffers, flatten
from fl_coordination._impl.fedpairs.common import finite, fingerprint_state

METHODS = ('fedavg','fedfv','fedfv_asam')
LABELS = {'fedavg':'FedAvg','fedfv':'FedFV','fedfv_asam':'FedFV & FedASAM'}


def check_ids(config, batches):
    ids = sorted(batches)
    if len(ids)!=config['participants'] or any(type(cid) is not int or not 0<=cid<config['clients'] for cid in ids):
        raise ValueError('Incorrect number/identity of participants')
    if any(not batches[cid] for cid in ids):
        raise ValueError('Empty local workload')
    return ids


class FedAvgEngine(EvaluationEngine):
    """Average sampled clients' actual SGD endpoints, weighted by fit-set size.

    Private heads use the same persistent Adam recipe as both FedFV variants.
    No global optimizer and no renormalization over absent clients.
    """
    def __init__(self,config,device,network):
        self.config,self.device = config,device
        self.encoder,self.heads = network
        self.encoder.to(device); self.heads.to(device)
        self.local_buffers = [buffers(self.encoder) for _ in self.heads]
        self.head_optimizers = [torch.optim.Adam(h.parameters(),lr=config['learning_rate']) for h in self.heads]
        self.step_index = 0
        self.initial_hash = fingerprint_state({
            **{'encoder.'+k:v for k,v in self.encoder.state_dict().items()},
            **{'heads.'+k:v for k,v in self.heads.state_dict().items()}})

    def step(self,batches):
        ids = check_ids(self.config,batches)
        counts = [self.config['client_fit_sizes'][cid] for cid in ids]
        if any(n<=0 for n in counts):
            raise ValueError('Nonpositive client size')
        weights = [n/sum(counts) for n in counts]
        shared = list(self.encoder.parameters())
        base = [p.detach().clone() for p in shared]
        merged = [torch.zeros_like(p) for p in shared]
        records = []
        try:
            for cid,weight in zip(ids,weights):
                with torch.no_grad():
                    for p,b in zip(shared,base):p.copy_(b)
                losses,ns = [],[]
                for batch in batches[cid]:
                    u = client_gradients(self.encoder,self.heads[cid],batch,self.local_buffers[cid],0.,False)
                    self.local_buffers[cid] = u.buffers
                    with torch.no_grad():
                        offset = 0
                        for p in shared:
                            size = p.numel()
                            p.add_(u.g[offset:offset+size].view_as(p),alpha=-self.config['local_learning_rate'])
                            offset += size
                    opt = self.head_optimizers[cid]
                    opt.zero_grad(set_to_none=True)
                    for p,g in zip(self.heads[cid].parameters(),u.head):p.grad=g.clone()
                    opt.step()
                    losses.append(u.loss);ns.append(len(batch[1]))
                with torch.no_grad():
                    for output,p in zip(merged,shared):
                        finite(p,'FedAvg local endpoint')
                        output.add_(p,alpha=weight)
                records.append(dict(cid=cid,samples=sum(ns),steps=len(ns),loss=float(np.average(losses,weights=ns))))
        finally:
            with torch.no_grad():
                for p,b in zip(shared,base):p.copy_(b)
        with torch.no_grad():
            for p,value in zip(shared,merged):
                finite(value,'FedAvg averaged endpoint');p.copy_(value)
        for p in self.heads.parameters():finite(p,'Private classifier')
        self.step_index += 1
        delta = flatten([b-p.detach() for b,p in zip(base,shared)])
        steps = sum(r['steps'] for r in records)
        return dict(round=self.step_index,client_ids=ids,
                    client_samples={str(r['cid']):r['samples'] for r in records},
                    client_steps={str(r['cid']):r['steps'] for r in records},
                    samples=sum(r['samples'] for r in records),local_steps=steps,backward_passes=steps,
                    losses=[r['loss'] for r in records],actual_server_step_norm=float(delta.norm()),
                    qp_g=dict(solver='sample_weighted_local_endpoints',weights=weights,sample_counts=counts),
                    qp_h=dict(solver='inactive'))

    def state_dict(self):
        return dict(model=self.model_state(),step=self.step_index,initial_hash=self.initial_hash,
                    head_optimizers=[copy.deepcopy(o.state_dict()) for o in self.head_optimizers])

    def load_state_dict(self,state):
        if len(state['head_optimizers'])!=len(self.heads):
            raise ValueError('Client count differs on resume')
        self.load_model(state['model'])
        for opt,value in zip(self.head_optimizers,state['head_optimizers']):opt.load_state_dict(value)
        self.step_index,self.initial_hash = state['step'],state['initial_hash']
