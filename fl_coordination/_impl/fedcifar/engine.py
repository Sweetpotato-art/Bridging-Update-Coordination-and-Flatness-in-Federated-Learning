"""Reuse unchanged multi-step branch arithmetic and pinned author coordinators."""
import torch

from fl_coordination._impl.feda3.operators import make_coordinator
from fl_coordination._impl.fedmini.model import buffers
from fl_coordination._impl.fedpairs.common import fingerprint_state
from fl_coordination._impl.fedseries.engine import Engine as SharedTrajectoryEngine, VARIANTS
from .model import build_network


NAMES = {'fedfv': 'FedFV', 'fedmgda': 'FedMGDA+', 'fedld': 'FedLD-Agg'}


def method_label(coordinator, variant, config=None):
    """Stable filesystem IDs; never label standard SAM as ASAM or vice versa."""
    base = NAMES[coordinator]
    if config is None or config.get('protocol') == 'CIFAR100-ResNet20-v1':
        return {'baseline': '', 'sam': 'ASAM-', 'split': 'F-'}[variant] + base
    flatness = 'FedASAM' if config['adaptive_sam'] else 'FedSAM'
    return {'baseline': base, 'sam': base + '-' + flatness + '-direct',
            'split': base + '-' + flatness + '-split'}[variant]


def display_label(coordinator, variant, config):
    if config.get('protocol') == 'CIFAR100-ResNet20-v1':
        return method_label(coordinator, variant, config)
    base = NAMES[coordinator]
    flatness = 'FedASAM' if config['adaptive_sam'] else 'FedSAM'
    return base if variant == 'baseline' else base + ' & ' + flatness


class NoPrivateOptimizer:
    """Shared-head clients have no private trainable tensors or optimizer state."""
    def zero_grad(self, **kwargs):
        pass

    def step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError('A shared-head run cannot restore a private optimizer')


class Engine(SharedTrajectoryEngine):
    def __init__(self, config, coordinator, variant, device, network=None):
        if variant not in VARIANTS or config['head_mode'] not in ('private', 'shared'):
            raise ValueError('Unknown method variant or head mode')
        if config['optimizer'] != 'adam' or config['weight_decay'] != 0:
            raise ValueError('This recipe keeps server/private Adam without weight decay')
        self.config, self.flat, self.device = config, variant != 'baseline', device
        self.variant = variant
        self.encoder, self.heads = network if network is not None else build_network(
            config['clients'], config['head_mode'], config['num_classes'])
        if len(self.heads) != config['clients']:
            raise ValueError('Head count differs from client count')
        self.encoder.to(device)
        self.heads.to(device)
        self.local_buffers = [buffers(self.encoder) for _ in self.heads]
        options = dict(lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.shared_optimizer = torch.optim.Adam(self.encoder.parameters(), **options)
        self.head_optimizers = []
        for head in self.heads:
            parameters = list(head.parameters())
            if bool(parameters) != (config['head_mode'] == 'private'):
                raise ValueError('Classifier parameters are in the wrong ownership group')
            self.head_optimizers.append(torch.optim.Adam(parameters, **options)
                                        if parameters else NoPrivateOptimizer())
        layout = [(name, p.numel()) for name, p in self.encoder.named_parameters()]
        self.g_coordinator = make_coordinator(coordinator, config, device, layout)
        self.h_coordinator = make_coordinator(coordinator, config, device, layout)
        self.step_index = 0
        self.initial_hash = fingerprint_state({
            **{'encoder.' + k: v for k, v in self.encoder.state_dict().items()},
            **{'heads.' + k: v for k, v in self.heads.state_dict().items()}})

    def step(self, client_batches):
        if sorted(client_batches) != list(range(self.config['clients'])):
            raise ValueError('CIFAR-100 recipe requires every client in every round')
        if any(not 1 <= len(batches) <= self.config['local_steps']
               for batches in client_batches.values()):
            raise ValueError('Local mini-batch count is outside the declared limit')
        return super().step(client_batches)
