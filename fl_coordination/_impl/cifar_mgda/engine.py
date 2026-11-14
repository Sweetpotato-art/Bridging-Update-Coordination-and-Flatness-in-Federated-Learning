"""Reuse frozen CIFAR trajectories and FedGF; replace the coordinator factory."""
from types import FunctionType
from fl_coordination._impl.fedcifar.engine import Engine as FrozenEngine
from fl_coordination._impl.fedcifarglobalflat.engine import Engine as FrozenGFEngine
from fl_coordination._impl.fedglobalflat.mechanisms import Mechanism
from .config import NAMES, MECHANISMS
from .operators import make_coordinator

_init = FunctionType(FrozenEngine.__init__.__code__,
    dict(FrozenEngine.__init__.__globals__, make_coordinator=make_coordinator),
    argdefs=FrozenEngine.__init__.__defaults__)


def method_label(coordinator, variant, config):
    mechanism = config['mechanism']
    if coordinator not in NAMES or mechanism not in MECHANISMS:
        raise ValueError('Unknown combination')
    expected = 'fedgf' if mechanism == 'fedgf' else MECHANISMS[mechanism][0]
    if variant != expected:
        raise ValueError('Wrong variant for flatness mechanism')
    name = {'fedsam':'FedSAM-split', 'fedasam':'FedASAM-direct', 'fedgf':'FedGF-split'}[mechanism]
    return NAMES[coordinator]+'-'+name


def display_label(coordinator, variant, config):
    method_label(coordinator, variant, config)
    return NAMES[coordinator]+' & '+{'fedsam':'FedSAM','fedasam':'FedASAM','fedgf':'FedGF'}[config['mechanism']]


class Engine(FrozenEngine):
    def __init__(self, config, coordinator, variant, device, network=None):
        method_label(coordinator, variant, config)
        self.kind = config['mechanism']
        if config['head_mode'] != 'private' or config['adaptive_sam'] != (self.kind == 'fedasam'):
            raise ValueError('Pin private heads and the selected perturbation type')
        _init(self, config, coordinator, MECHANISMS[self.kind][0], device, network)
        if self.kind == 'fedgf':
            counts = config['client_fit_sizes']
            if len(counts) != config['clients'] or any(n <= 0 for n in counts):
                raise ValueError('FedGF requires exact positive fitting counts')
            self.fractions = [n/sum(counts) for n in counts]
            self.mechanism = Mechanism('fedgf', config, device)

    def step(self, client_batches):
        if self.kind == 'fedgf':
            return FrozenGFEngine.step(self, client_batches)
        return super().step(client_batches)

    def state_dict(self):
        value = super().state_dict()
        if self.kind == 'fedgf':
            value['flatness'] = self.mechanism.state_dict()
        return value

    def load_state_dict(self, value):
        super().load_state_dict(value)
        if self.kind == 'fedgf':
            self.mechanism.load_state_dict(value['flatness'])
