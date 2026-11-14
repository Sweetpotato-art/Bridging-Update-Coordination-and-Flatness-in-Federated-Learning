"""Identical validation selection, atomic checkpoints, and resume semantics."""
from types import FunctionType
from fl_coordination._impl.fedcifar import runtime as ordinary
from fl_coordination._impl.fedcifarglobalflat import runtime as global_flat
from .engine import Engine, method_label, display_label


def adapt(function):
    result = FunctionType(function.__code__, dict(function.__globals__,
        method_label=method_label, display_label=display_label), argdefs=function.__defaults__)
    result.__kwdefaults__ = dict(function.__kwdefaults__, engine_factory=Engine)
    return result


sam_run, gf_run = adapt(ordinary.run_method), adapt(global_flat.run_method)


def run_method(config, data, coordinator, variant, device, result_root, weight_root,
               source_manifest, **kwargs):
    runner = gf_run if config['mechanism'] == 'fedgf' else sam_run
    return runner(config, data, coordinator, variant, device, result_root, weight_root,
                  source_manifest, **kwargs)
