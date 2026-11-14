"""Reuse adaptive joint perturbation and direct A(mean p), including unequal-K AWA."""
from fl_coordination._impl.fedcifarasam.engine import Engine as DirectASAM
from fl_coordination._impl.fedoffice.model import build_network
from .config import parse_job


def build_engine(cfg, job, device, network=None):
    c, _ = parse_job(job)
    if not cfg['adaptive_sam'] or cfg['mechanism'] != 'asam_direct':
        raise ValueError('ASAM adaptive perturbation and direct coordination required')
    engine = DirectASAM(cfg, c, 'sam', device, network or build_network(cfg))
    for group in engine.shared_optimizer.param_groups:
        group['lr'] = cfg['server_learning_rate']
    return engine
