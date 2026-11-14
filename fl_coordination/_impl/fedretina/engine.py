"""Reuse the verified trajectory coordinators and sampled-endpoint FedAvg engine."""
import numpy as np
import torch
from torch.nn import functional as F
from fl_coordination._impl.checkpointing.engine import FedAvgEngine as EndpointEngine
from fl_coordination._impl.fedseries.engine import Engine as TrajectoryEngine
from fl_coordination._impl.fedcifar.engine import Engine as CoordinatorEngine
from fl_coordination._impl.fedcifarextra.operators import UnequalStepFedAWA
from fl_coordination._impl.fedmini.model import evaluating, restore_buffers, flatten
from fl_coordination._impl.fedpairs.common import finite
from .model import build_network

DEFAULT_METHODS = ('fedavg','fedfv','fedfv_asam')
METHODS = (*DEFAULT_METHODS, 'fedawa', 'fedawa_asam')
LABELS = {'fedavg':'FedAvg','fedfv':'FedFV','fedfv_asam':'FedFV & FedASAM',
          'fedawa':'FedAWA-Agg','fedawa_asam':'FedAWA-Agg & FedASAM'}


def binary_metrics(labels, probabilities):
    y,p=np.asarray(labels,dtype=np.int64),np.asarray(probabilities,dtype=np.float64)
    if not len(y) or p.shape!=y.shape or not np.isfinite(p).all(): raise ValueError('Invalid binary predictions')
    predicted=(p>.5).astype(np.int64)  # binary argmax chooses class 0 on an exact tie
    # Return null rather than an invented AUC if a set contains only one class.
    pos,neg=int((y==1).sum()),int((y==0).sum())
    auc=balanced=None
    if pos and neg:
        order=np.argsort(p,kind='stable'); ranks=np.empty(len(p),dtype=np.float64)
        start=0
        while start<len(p):
            end=start+1
            while end<len(p) and p[order[end]]==p[order[start]]: end+=1
            ranks[order[start:end]]=(start+1+end)/2;start=end
        auc=float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg))
        balanced=50.*(float((predicted[y==0]==0).mean())+float((predicted[y==1]==1).mean()))
    return dict(accuracy=100.*float((predicted==y).mean()),auroc=auc,balanced_accuracy=balanced,
                samples=len(y),class_counts=[neg,pos])


class Evaluation:
    @torch.no_grad()
    def evaluate(self, data, split):
        values=[]
        with evaluating(self.encoder,self.heads):
            for cid,head in enumerate(self.heads):
                restore_buffers(self.encoder,self.local_buffers[cid])
                labels,probs=[],[];loss=0.
                for x,y in data.evaluation_batches(split,cid,self.device):
                    logits=head(self.encoder(x));finite(logits,'Retina evaluation logits')
                    probs.extend(logits.softmax(1)[:,1].cpu().tolist());labels.extend(y.cpu().tolist())
                    loss+=float(F.cross_entropy(logits,y,reduction='sum'))
                v=binary_metrics(labels,probs);v['loss']=loss/v['samples'];values.append(v)
        result=dict(clients=values,mean=float(np.mean([v['accuracy'] for v in values])),
                    worst=min(v['accuracy'] for v in values),
                    mean_auroc=float(np.mean([v['auroc'] for v in values])) if all(v['auroc'] is not None for v in values) else None,
                    mean_balanced_accuracy=float(np.mean([v['balanced_accuracy'] for v in values])) if all(v['balanced_accuracy'] is not None for v in values) else None)
        valid_auc=[v['auroc'] for v in values if v['auroc'] is not None]
        result.update(auroc_defined_clients=len(valid_auc),
                      mean_available_auroc=float(np.mean(valid_auc)) if valid_auc else None)
        if split=='public_test':
            result.update(evaluation='Each private head and its BN state evaluated on the same untouched public test set',
                          unique_samples=len(data.public_indices))
        else:
            result.update(pooled=sum(v['accuracy']*v['samples'] for v in values)/sum(v['samples'] for v in values),
                          evaluation='Disjoint held-out patients within each author client')
        return result


class FedAvg(Evaluation, EndpointEngine):
    def __init__(self,config,device,network=None):
        super().__init__(config,device,network or build_network(config))


class FedFV(Evaluation, TrajectoryEngine):
    def __init__(self,config,method,device,network=None):
        super().__init__(config,'fedfv','sam' if method=='fedfv_asam' else 'baseline',device,network or build_network(config))
        for group in self.shared_optimizer.param_groups: group['lr']=config['server_learning_rate']


class FedAWA(Evaluation, CoordinatorEngine):
    """Author AWA weights on full, unequal-length Retina local trajectories."""
    def __init__(self,config,method,device,network=None):
        variant='sam' if method=='fedawa_asam' else 'baseline'
        super().__init__(config,'fedawa',variant,device,network or build_network(config))
        layout=[(name,p.numel()) for name,p in self.encoder.named_parameters()]
        self.g_coordinator=UnequalStepFedAWA(config,device,layout)
        self.h_coordinator=UnequalStepFedAWA(config,device,layout)
        for group in self.shared_optimizer.param_groups: group['lr']=config['server_learning_rate']

    def step(self,client_batches):
        ids=sorted(client_batches)
        if ids!=list(range(self.config['clients'])):
            raise ValueError('Retina FedAWA requires full ordered participation')
        base=flatten(list(self.encoder.parameters())).detach()
        scales=[self.config['local_learning_rate']*len(client_batches[c]) for c in ids]
        self.g_coordinator.bind_context(base,scales)
        self.h_coordinator.bind_context(base,scales)
        # Full local epochs have unequal K_i. Do not inherit the CIFAR batch cap.
        # The shared trajectory routine supplies mean g (baseline) or mean p (ASAM).
        return TrajectoryEngine.step(self,client_batches)


def method_config(config,method,data):
    if method not in METHODS: raise ValueError(method)
    result=dict(config,client_fit_sizes=list(data.fit_sizes))
    if method not in ('fedfv_asam','fedawa_asam'): result.update(rho=0.,flat_lambda=0.)
    return result


def build_engine(config,method,device,network=None):
    if method not in METHODS: raise ValueError(method)
    if method=='fedavg': return FedAvg(config,device,network)
    if method in ('fedawa','fedawa_asam'): return FedAWA(config,method,device,network)
    return FedFV(config,method,device,network)
