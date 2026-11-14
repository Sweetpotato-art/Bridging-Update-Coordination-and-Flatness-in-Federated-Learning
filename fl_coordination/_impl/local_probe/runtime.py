"""Same full epochs, held-out selection and exact resume; distinct method labels."""
from types import FunctionType
from fl_coordination._impl.fedcifarglobalflat import runtime as frozen
from fl_coordination._impl.cifar_probe_engine.runtime import epoch_summary as original_summary
from .engine import Engine, method_label, display_label


def epoch_summary(records, clients):
    value = original_summary(records, clients)
    probes = [r['local_probe'] for r in records]
    n = sum(p['batches'] for p in probes)
    summary = dict(batches=n, local_rho=probes[0]['local_rho'],
                   policy='history_then_batch_local_sam')
    for key in probes[0]:
        if not key.endswith('_mean'): continue
        active = [p for p in probes if p[key] is not None]
        summary[key] = sum(p[key]*p['batches'] for p in active)/sum(p['batches'] for p in active) if active else None
    value['local_probe'] = summary
    return value


run_method = FunctionType(frozen.run_method.__code__, dict(frozen.run_method.__globals__,
    method_label=method_label, display_label=display_label, epoch_summary=epoch_summary),
    argdefs=frozen.run_method.__defaults__)
run_method.__kwdefaults__ = dict(frozen.run_method.__kwdefaults__, engine_factory=Engine)
