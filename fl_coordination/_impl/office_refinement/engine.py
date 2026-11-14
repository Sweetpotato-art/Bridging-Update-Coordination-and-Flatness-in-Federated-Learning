"""Reuse existing numerical kernels without changing pinned predecessor files."""
from fl_coordination._impl.fedcifar.engine import Engine as TrajectoryEngine
from fl_coordination._impl.fedoffice.model import build_network
from fl_coordination._impl.fedofficecomplete.engine import GFEngine, GloSSLocalEngine

def build_engine(cfg, job, device, network=None):
    combination, rho = job, cfg["rho"]
    c, v = combination.rsplit('_', 1)
    if (cfg['coordinator'], cfg['mechanism'], cfg['rho']) != (c, v, rho):
        raise ValueError('Job/config identity mismatch')
    net = network if network is not None else build_network(cfg)
    if v == 'gf':
        engine = GFEngine(cfg, c, device, net)
    elif v == 'gloss':
        engine = GloSSLocalEngine(cfg, c, device, net)
        engine.mechanism.local_sam_rho = cfg['local_sam_rho']
    else:
        engine = TrajectoryEngine(cfg, c, 'sam' if v == 'asam' else 'split', device, net)
    for group in engine.shared_optimizer.param_groups:
        group['lr'] = cfg['server_learning_rate']
    return engine
