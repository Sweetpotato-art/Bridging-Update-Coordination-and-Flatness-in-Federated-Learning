"""CIFAR-100 training with FedMDFG-Agg, FedAWA-Agg, and FedLD-Agg."""
from fl_coordination._impl.fedcifar.engine import Engine as OriginalEngine
from fl_coordination._impl.fedmini.model import flatten
from .operators import make_coordinator

NAMES={'fedmgda':'FedMGDA+','fedmdfg':'FedMDFG-Agg','fedawa':'FedAWA-Agg'}


def method_label(coordinator,variant,config=None):
    if coordinator not in NAMES or variant not in ('baseline','split'):
        raise ValueError('Unsupported CIFAR-100 coordination recipe')
    return NAMES[coordinator]+('-FedSAM-split' if variant=='split' else '')


def display_label(coordinator,variant,config=None):
    method_label(coordinator,variant,config)
    return NAMES[coordinator]+(' & FedSAM' if variant=='split' else '')


class Engine(OriginalEngine):
    def __init__(self,config,coordinator,variant,device,network=None):
        method_label(coordinator,variant,config)
        if config['head_mode']!='private' or config['adaptive_sam']:
            raise ValueError('This recipe pins private heads and standard SAM')
        super().__init__(config,coordinator,variant,device,network)
        if coordinator=='fedawa':
            layout=[(name,p.numel()) for name,p in self.encoder.named_parameters()]
            self.g_coordinator=make_coordinator(coordinator,config,device,layout)
            self.h_coordinator=make_coordinator(coordinator,config,device,layout)

    def step(self,client_batches):
        if hasattr(self.g_coordinator,'bind_context'):
            if sorted(client_batches)!=list(range(self.config['clients'])):
                raise ValueError('Full participation required')
            base=flatten(list(self.encoder.parameters())).detach()
            scales=[self.config['local_learning_rate']*len(client_batches[i])
                    for i in range(self.config['clients'])]
            self.g_coordinator.bind_context(base,scales)
            self.h_coordinator.bind_context(base,scales)
        return super().step(client_batches)
