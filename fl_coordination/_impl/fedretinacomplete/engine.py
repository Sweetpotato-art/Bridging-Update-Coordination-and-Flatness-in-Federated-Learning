"""Reuse full Retina trajectories, existing author coordinators and local probes."""
import math
from fl_coordination._impl.fedcifar.engine import Engine as CoordinatorEngine
from fl_coordination._impl.fedcifarglobalflat.engine import Engine as FlatEngine
from fl_coordination._impl.fedcifarextra.operators import make_coordinator
from fl_coordination._impl.fedglobalflat.mechanisms import Mechanism
from fl_coordination._impl.local_probe.engine import Engine as LocalProbeEngine
from fl_coordination._impl.fedretina.engine import Evaluation
from fl_coordination._impl.fedretina.model import build_network
from fl_coordination._impl.fedretinamatrix.engine import SAMEngine
from .config import parse_job


def initialize_flat(engine, cfg, coordinator, kind, device, network):
    CoordinatorEngine.__init__(engine, cfg, coordinator, 'split', device, network)
    if coordinator == 'fedawa':
        layout = [(name, p.numel()) for name, p in engine.encoder.named_parameters()]
        engine.g_coordinator = make_coordinator(coordinator, cfg, device, layout)
        engine.h_coordinator = make_coordinator(coordinator, cfg, device, layout)
    counts = cfg['client_fit_sizes']
    if len(counts) != cfg['clients'] or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError('Exact fitting sample counts required')
    engine.fractions = [n/sum(counts) for n in counts]
    engine.mechanism = Mechanism(kind, cfg, device)


class GFEngine(Evaluation, FlatEngine):
    def __init__(self, cfg, coordinator, device, network):
        initialize_flat(self, cfg, coordinator, 'fedgf', device, network)


class GloSSLocalEngine(Evaluation, LocalProbeEngine):
    def __init__(self, cfg, coordinator, device, network):
        local_rho = cfg['local_sam_rho']
        if not math.isfinite(local_rho) or local_rho <= 0:
            raise ValueError('Localized GloSS requires a positive local SAM radius')
        initialize_flat(self, cfg, coordinator, 'fedgloss', device, network)
        self.mechanism.local_sam_rho = local_rho
        self.mechanism.local_probe_records = []


def build_engine(cfg, job, device, network=None):
    c, v = parse_job(job)
    if (cfg['coordinator'] != c or cfg['mechanism'] != v or cfg['head_mode'] != 'private' or
            cfg['adaptive_sam'] != (v == 'asam')):
        raise ValueError('Retina configuration/engine mismatch')
    if not math.isfinite(cfg['rho']) or cfg['rho'] < 0:
        raise ValueError('Invalid perturbation radius')
    if v == 'gf' and (not math.isfinite(cfg['fedgf_threshold']) or cfg['fedgf_threshold'] <= 0):
        raise ValueError('Invalid GF calibration')
    net = network if network is not None else build_network(cfg)
    if v == 'gloss':
        engine = GloSSLocalEngine(cfg, c, device, net)
    elif v == 'gf':
        engine = GFEngine(cfg, c, device, net)
    else:
        engine = SAMEngine(cfg, c, {'baseline': 'baseline', 'sam': 'split', 'asam': 'sam'}[v], device, net)
    for group in engine.shared_optimizer.param_groups:
        group['lr'] = cfg['server_learning_rate']
    return engine
