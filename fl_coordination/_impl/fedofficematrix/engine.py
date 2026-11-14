"""Office-Home training with separate coordination for SAM and FedGF."""
from fl_coordination._impl.fedcifar.engine import Engine as TrajectoryEngine
from fl_coordination._impl.fedcifarextra.operators import make_coordinator
from fl_coordination._impl.fedcifarglobalflat.engine import Engine as GFEngine
from fl_coordination._impl.fedmini.model import flatten
from fl_coordination._impl.fedoffice.model import build_network
from .config import parse_job


class SAMEngine(TrajectoryEngine):
    def __init__(self, cfg, coordinator, variant, device, network=None):
        super().__init__(cfg, coordinator, variant, device, network or build_network(cfg))
        if coordinator == 'fedawa':
            layout = [(name, p.numel()) for name, p in self.encoder.named_parameters()]
            self.g_coordinator = make_coordinator(coordinator, cfg, device, layout)
            self.h_coordinator = make_coordinator(coordinator, cfg, device, layout)

    def step(self, client_batches):
        if hasattr(self.g_coordinator, 'bind_context'):
            ids = sorted(client_batches)
            if ids != list(range(self.config['clients'])):
                raise ValueError('Full participation is required')
            base = flatten(list(self.encoder.parameters())).detach()
            scales = [self.config['local_learning_rate']*len(client_batches[c]) for c in ids]
            self.g_coordinator.bind_context(base, scales)
            self.h_coordinator.bind_context(base, scales)
        return super().step(client_batches)


def build_engine(cfg, job, device, network=None):
    c, v = parse_job(job)
    if cfg['adaptive_sam'] or cfg['head_mode'] != 'private':
        raise ValueError('This matrix uses standard SAM/GF with private heads')
    if v == 'gf':
        engine = GFEngine(cfg, c, 'fedgf', device, network or build_network(cfg))
    else:
        engine = SAMEngine(cfg, c, 'baseline' if v == 'baseline' else 'split', device, network)
    for group in engine.shared_optimizer.param_groups:
        group['lr'] = cfg['server_learning_rate']
    return engine
