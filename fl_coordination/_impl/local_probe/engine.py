"""Reuse the frozen FedFV trajectory and history; replace only probe evaluation."""
import math
from types import FunctionType
from fl_coordination._impl.cifar_probe_engine.engine import Engine as OriginalEngine
from fl_coordination._impl.fedcifarglobalflat.engine import Engine as FlatEngine
from .gradients import paired_gradients

NAMES = {'fedgloss': 'FedGloSS'}
_step = FunctionType(FlatEngine.step.__code__,
    dict(FlatEngine.step.__globals__, paired_gradients=paired_gradients),
    argdefs=FlatEngine.step.__defaults__)


def method_label(c, m, config=None):
    if c != 'fedfv' or m not in NAMES: raise ValueError('Expected FedFV with localized FedGloSS')
    return 'FedFV-'+NAMES[m]+'-LocalSAM-split'


def display_label(c, m, config=None):
    method_label(c, m, config)
    return 'FedFV & '+NAMES[m]+' + local SAM (independent g/h)'


class Engine(OriginalEngine):
    def __init__(self, config, coordinator, variant, device, network=None):
        method_label(coordinator, variant, config)
        local_rho = config['local_sam_rho']
        if not math.isfinite(local_rho) or local_rho < 0:
            raise ValueError('Invalid local SAM radius')
        super().__init__(config, coordinator, variant, device, network)
        self.mechanism.local_sam_rho = local_rho
        self.mechanism.local_probe_records = []

    def step(self, client_batches):
        self.mechanism.local_probe_records = []
        record = _step(self, client_batches)
        probes = self.mechanism.local_probe_records
        if len(probes) != record['local_steps']:
            raise RuntimeError('One fresh local probe per batch is required')
        info = dict(batches=len(probes), local_rho=self.mechanism.local_sam_rho)
        for key in ('shift_norm', 'local_epsilon_norm', 'historical_local_cosine',
                    'shifted_gradient_norm', 'local_probe_loss_delta'):
            values = [p[key] for p in probes if p[key] is not None]
            info[key+'_mean'] = sum(values)/len(values) if values else None
        record['local_probe'] = info
        return record

    def state_dict(self):
        return dict(super().state_dict(), local_sam_rho=self.mechanism.local_sam_rho)

    def load_state_dict(self, value):
        if value['local_sam_rho'] != self.mechanism.local_sam_rho:
            raise ValueError('Local SAM checkpoint radius differs')
        super().load_state_dict(value)
        self.mechanism.local_probe_records = []
