"""Direct coordination of trajectory-mean ASAM gradients for CIFAR-100."""
from fl_coordination._impl.fedcifar.engine import Engine as OriginalEngine
from fl_coordination._impl.fedcifarextra.operators import make_coordinator
from fl_coordination._impl.fedmini.model import flatten
from .config import NAMES


def method_label(coordinator,variant,config=None):
    if coordinator not in NAMES or variant!='sam':
        raise ValueError('Unsupported CIFAR-100 direct ASAM recipe')
    return NAMES[coordinator]+'-FedASAM-direct'


def display_label(coordinator,variant,config=None):
    method_label(coordinator,variant,config)
    return NAMES[coordinator]+' & FedASAM'


class Engine(OriginalEngine):
    def __init__(self,config,coordinator,variant,device,network=None):
        method_label(coordinator,variant,config)
        if not config['adaptive_sam'] or config['head_mode']!='private':
            raise ValueError('Use ASAM with private classifier heads')
        super().__init__(config,coordinator,variant,device,network)
        if coordinator=='fedawa':
            # Use each client's local step count in its FedAWA pseudo-endpoint.
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
