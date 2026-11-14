"""Reuse paired derivatives and author coordinators over full Retina trajectories."""
from fl_coordination._impl.fedcifar.engine import Engine as CoordinatorEngine
from fl_coordination._impl.fedseries.engine import Engine as TrajectoryEngine
from fl_coordination._impl.fedcifarglobalflat.engine import Engine as GlobalFlatEngine
from fl_coordination._impl.fedcifarextra.operators import make_coordinator
from fl_coordination._impl.fedmini.model import flatten
from fl_coordination._impl.fedretina.engine import Evaluation
from fl_coordination._impl.fedretina.model import build_network
from .config import parse_job


class SAMEngine(Evaluation, CoordinatorEngine):
    def __init__(self, cfg, coordinator, variant, device, network):
        super().__init__(cfg, coordinator, variant, device, network)
        if coordinator == 'fedawa':
            layout = [(name, p.numel()) for name, p in self.encoder.named_parameters()]
            self.g_coordinator = make_coordinator(coordinator, cfg, device, layout)
            self.h_coordinator = make_coordinator(coordinator, cfg, device, layout)

    def step(self, batches):
        ids = sorted(batches)
        if ids != list(range(self.config['clients'])) or any(not batches[c] for c in ids):
            raise ValueError('Retina requires all five complete nonempty client trajectories')
        if hasattr(self.g_coordinator, 'bind_context'):
            base = flatten(list(self.encoder.parameters())).detach()
            scales = [self.config['local_learning_rate']*len(batches[c]) for c in ids]
            self.g_coordinator.bind_context(base, scales)
            self.h_coordinator.bind_context(base, scales)
        # Full local epochs: no inherited CIFAR mini-batch truncation.
        return TrajectoryEngine.step(self, batches)


class GFEngine(Evaluation, GlobalFlatEngine):
    pass


def build_engine(cfg, job, device, network=None):
    c, v = parse_job(job)
    if cfg['coordinator'] != c or cfg['mechanism'] != v or cfg['adaptive_sam'] or cfg['head_mode'] != 'private':
        raise ValueError('Retina matrix configuration/engine mismatch')
    net = network if network is not None else build_network(cfg)
    if v == 'gf':
        engine = GFEngine(cfg, c, 'fedgf', device, net)
    else:
        engine = SAMEngine(cfg, c, 'baseline' if v == 'baseline' else 'split', device, net)
    for group in engine.shared_optimizer.param_groups:
        group['lr'] = cfg['server_learning_rate']
    return engine
