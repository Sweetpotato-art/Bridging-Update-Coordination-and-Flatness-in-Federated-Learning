"""CIFAR FedAWA context for unequal K_i; the pinned author objective is unchanged."""
import contextlib
import io
from types import SimpleNamespace
import torch
from fl_coordination._impl.feda3.operators import FedAWAAgg, make_coordinator as pinned_coordinator
from fl_coordination._impl.fedpairs.common import finite


class UnequalStepFedAWA(FedAWAAgg):
    def __init__(self, config, device, layout):
        super().__init__(config, device, layout)
        counts = config['client_fit_sizes']
        if len(counts)!=self.n or any(type(n) is not int or n<=0 for n in counts):
            raise ValueError('FedAWA requires complete fitting sample counts')
        self.fractions = [n/sum(counts) for n in counts]
        # Author initializes logits with sample fractions, not their logarithms.
        self.logits = torch.tensor(self.fractions,dtype=torch.float32,device=device)

    def bind_context(self, base, scales):
        if len(scales)!=self.n or any(s<=0 for s in scales):
            raise ValueError('FedAWA needs one positive eta*K_i per client')
        self.context = base.detach(), tuple(float(s) for s in scales)

    def __call__(self, vectors, losses, step, ids=None):
        if ids!=list(range(self.n)) or self.context is None:
            raise ValueError('FedAWA needs ordered full participation and current endpoints')
        base, scales = self.context
        matrix = torch.stack(vectors)
        finite(matrix,'FedAWA shared gradients')
        if matrix.shape!=(self.n,base.numel()):
            raise ValueError('FedAWA coordinate shape mismatch')
        used = torch.softmax(self.logits,0).detach().clone()
        endpoints = [{'flat_w':base-scale*v} for scale,v in zip(scales,vectors)]
        central = SimpleNamespace(model=SimpleNamespace(get_param=lambda:{'flat_w':base}))
        with contextlib.redirect_stdout(io.StringIO()):
            merged,logits = self.native.fedawa(self.args,endpoints,self.fractions,
                                                central,step,self.logits.detach().clone())
        finite(logits,'FedAWA updated logits')
        finite(merged['flat_w'],'FedAWA merged endpoint')
        self.logits = logits.detach().clone()
        # Common-recipe aggregation consumes trajectory MEAN gradients. K_i is
        # used in the author's endpoint objective, not silently in the server step.
        direction = used @ matrix
        scaled = matrix * matrix.new_tensor(scales)[:,None]
        error = float((merged['flat_w']-(base-used@scaled)).norm()/base.norm().clamp_min(1e-12))
        finite(direction,'FedAWA common-recipe direction')
        return direction,dict(solver='author_fedawa_release',weights=used.cpu().tolist(),
            next_weights=torch.softmax(self.logits,0).cpu().tolist(),
            pseudo_endpoint_scales=list(scales),endpoint_relative_error=error,
            weight_application='release_pre_step',regularizer='weighted_client_model_cosine',
            direction_units='trajectory_mean_gradient')


def make_coordinator(name, config, device, layout):
    if name=='fedawa':
        return UnequalStepFedAWA(config,device,layout)
    return pinned_coordinator(name,config,device,layout)
