"""Fashion-MNIST FedGloSS with a fresh local probe after the global shift."""
import math
from types import FunctionType

from fl_coordination._impl.fedglobalflat.engine import Engine as GlobalFlatEngine
from fl_coordination._impl.fedglobalflat.runtime import epoch_summary as original_summary
from fl_coordination._impl.local_probe.gradients import paired_gradients


# Reuse the Fashion trajectory with an isolated probe binding.
_step = FunctionType(
    GlobalFlatEngine.step.__code__,
    dict(GlobalFlatEngine.step.__globals__, paired_gradients=paired_gradients),
    argdefs=GlobalFlatEngine.step.__defaults__,
)


class LocalizedGloSSEngine(GlobalFlatEngine):
    def __init__(self, config, coordinator, variant, device, network=None):
        if variant != 'fedgloss':
            raise ValueError('The localized probe requires FedGloSS')
        local_rho = config['local_sam_rho']
        if not math.isfinite(local_rho) or local_rho < 0:
            raise ValueError('Invalid local SAM radius')
        super().__init__(config, coordinator, variant, device, network=network)
        self.mechanism.local_sam_rho = local_rho
        self.mechanism.local_probe_records = []

    def step(self, batches):
        self.mechanism.local_probe_records = []
        record = _step(self, batches)
        probes = self.mechanism.local_probe_records
        if len(probes) != record['local_steps']:
            raise RuntimeError('Every local batch must receive a fresh SAM probe')
        info = dict(batches=len(probes), local_rho=self.mechanism.local_sam_rho)
        for key in ('shift_norm', 'local_epsilon_norm', 'historical_local_cosine',
                    'shifted_gradient_norm', 'local_probe_loss_delta'):
            values = [p[key] for p in probes if p[key] is not None]
            info[key + '_mean'] = sum(values) / len(values) if values else None
        record['local_probe'] = info
        return record

    def state_dict(self):
        return dict(super().state_dict(), local_sam_rho=self.mechanism.local_sam_rho)

    def load_state_dict(self, value):
        if value['local_sam_rho'] != self.mechanism.local_sam_rho:
            raise ValueError('Saved local SAM radius differs')
        super().load_state_dict(value)
        self.mechanism.local_probe_records = []


def epoch_summary(records, clients):
    value = original_summary(records, clients)
    probes = [r['local_probe'] for r in records]
    n = sum(p['batches'] for p in probes)
    info = dict(batches=n, local_rho=probes[0]['local_rho'])
    for key in ('shift_norm', 'local_epsilon_norm', 'historical_local_cosine',
                'shifted_gradient_norm', 'local_probe_loss_delta'):
        field = key + '_mean'
        active = [p for p in probes if p[field] is not None]
        count = sum(p['batches'] for p in active)
        info[field] = sum(p[field] * p['batches'] for p in active) / count if count else None
    value['local_probe'] = info
    return value
