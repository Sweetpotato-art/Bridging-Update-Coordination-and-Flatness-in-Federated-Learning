"""Fashion-MNIST trajectory training with interchangeable coordinators."""
import torch

from fl_coordination._impl.fedmini.model import build_network, buffers, flatten
from fl_coordination._impl.fedpairs.common import fingerprint_state
from fl_coordination._impl.fedseries.engine import Engine as SeriesEngine, VARIANTS
from .operators import make_coordinator


NAMES = {'fedfv':'FedFV','fedmgda':'FedMGDA+','fedmdfg':'FedMDFG-Agg',
         'fedawa':'FedAWA-Agg','fedld':'FedLD-Agg'}
EXTRA = ('fedmdfg','fedawa','fedld')


def method_label(coordinator, variant):
    return {'baseline':'','sam':'ASAM-','split':'F-'}[variant]+NAMES[coordinator]


class Engine(SeriesEngine):
    def __init__(self, config, coordinator, variant, device, network=None):
        if variant not in VARIANTS or config['optimizer'] != 'adam':
            raise ValueError('Expected a supported Fashion-MNIST variant and the Adam optimizer')
        self.config, self.flat, self.device = config, variant != 'baseline', device
        self.variant = variant
        self.encoder, self.heads = build_network(config['clients']) if network is None else network
        self.encoder.to(device); self.heads.to(device)
        self.local_buffers = [buffers(self.encoder) for _ in self.heads]
        options = dict(lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.shared_optimizer = torch.optim.Adam(self.encoder.parameters(), **options)
        self.head_optimizers = [torch.optim.Adam(h.parameters(), **options) for h in self.heads]
        layout = [(name,p.numel()) for name,p in self.encoder.named_parameters()]
        self.g_coordinator = make_coordinator(coordinator,config,device,layout)
        self.h_coordinator = make_coordinator(coordinator,config,device,layout)
        self.step_index = 0
        self.initial_hash = fingerprint_state(dict(**{'encoder.'+k:v for k,v in self.encoder.state_dict().items()},
                                                  **{'heads.'+k:v for k,v in self.heads.state_dict().items()}))

    def step(self, client_batches):
        lengths = {len(b) for b in client_batches.values()}
        if sorted(client_batches) != list(range(self.config['clients'])) or len(lengths) != 1:
            raise ValueError('A3 requires all clients with the same local block length')
        if hasattr(self.g_coordinator, 'bind_context'):
            base = flatten(list(self.encoder.parameters()))
            scale = self.config['local_learning_rate'] * next(iter(lengths))
            self.g_coordinator.bind_context(base, scale)
            self.h_coordinator.bind_context(base, scale)
        return super().step(client_batches)
