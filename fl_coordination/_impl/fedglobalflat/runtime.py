"""Reuse the frozen data coverage, validation selection and atomic resume loop."""
from types import FunctionType
import numpy as np

from fl_coordination._impl.fedseries import runtime as frozen
from .engine import Engine, method_label


def epoch_summary(records, clients):
    summary = frozen.epoch_summary(records, clients)
    summary['flatness'] = dict(
        mechanism=records[0]['flatness']['mechanism'],
        perturbation_norm_mean=float(np.mean([r['perturbation_norms'] for r in records])),
        global_perturbation_norm_mean=float(np.mean([r['flatness']['global_perturbation_norm'] for r in records])),
        client_endpoint_norm_mean=float(np.mean([r['flatness']['client_endpoint_norm_mean'] for r in records])),
        fedgf_c=[r['flatness']['fedgf_c'] for r in records])
    return summary


run_method = FunctionType(frozen.run_method.__code__,
                         dict(frozen.run_method.__globals__, method_label=method_label,
                              epoch_summary=epoch_summary),
                         name='run_method', argdefs=frozen.run_method.__defaults__)
run_method.__kwdefaults__ = dict(frozen.run_method.__kwdefaults__, engine_factory=Engine)
run_method.__module__ = __name__
bind_experiment = frozen.bind_experiment
