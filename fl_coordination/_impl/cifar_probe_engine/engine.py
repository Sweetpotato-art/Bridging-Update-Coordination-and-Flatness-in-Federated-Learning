"""Reuse the tested local trajectory; introduce only the new mechanism adapter."""
from types import FunctionType
from fl_coordination._impl.fedcifar.engine import Engine as BaseEngine
from fl_coordination._impl.fedcifarglobalflat.engine import Engine as FlatEngine
from fl_coordination._impl.fedcifarextra.operators import make_coordinator as original_coordinator
from fl_coordination._impl.cifar_mgda.operators import make_coordinator as ml_coordinator
from fl_coordination._impl.fedglobalflat.mechanisms import Mechanism
from .config import NAMES, FLAT_NAMES, NEW_JOBS


def make_coordinator(name,config,device,layout):
    factory=ml_coordinator if name == 'fedmgda' else original_coordinator
    return factory(name,config,device,layout)


_init=FunctionType(BaseEngine.__init__.__code__,
    dict(BaseEngine.__init__.__globals__,make_coordinator=make_coordinator),
    argdefs=BaseEngine.__init__.__defaults__)


def method_label(c,m,config=None):
    if (c,m) not in NEW_JOBS:raise ValueError('Only twelve new mechanism combinations')
    return NAMES[c]+'-'+FLAT_NAMES[m]+'-split'


def display_label(c,m,config=None):
    method_label(c,m,config)
    return NAMES[c]+' & '+FLAT_NAMES[m]


class Engine(BaseEngine):
    def __init__(self,config,coordinator,variant,device,network=None):
        method_label(coordinator,variant,config)
        if config['head_mode']!='private' or config['adaptive_sam']:
            raise ValueError('Standard L2 perturbations and private heads are required')
        _init(self,config,coordinator,'split',device,network)
        counts=config['client_fit_sizes']
        if len(counts)!=config['clients'] or any(n<=0 for n in counts):raise ValueError('Invalid fitting counts')
        self.fractions = [n/sum(counts) for n in counts]
        self.mechanism = Mechanism('fedgloss', config, device)

    def step(self, client_batches):
        return FlatEngine.step(self, client_batches)

    def state_dict(self):
        return dict(super().state_dict(),flatness=self.mechanism.state_dict())

    def load_state_dict(self,value):
        super().load_state_dict(value)
        self.mechanism.load_state_dict(value['flatness'])
