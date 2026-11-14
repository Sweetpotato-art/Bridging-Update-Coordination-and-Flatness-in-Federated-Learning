"""Reuse the exact fixed-epoch loop with private descriptive bindings.

Function globals are copied, not the numerical implementation. Existing modules
and their running experiments are never edited or monkey-patched.
"""
from types import FunctionType
from fl_coordination._impl.fedofficecomplete import runtime as original
from .config import METHODS, NAMES, build_engine

def parse_baseline(method):
    if method not in METHODS: raise ValueError('Baseline-only entry')
    return method, 'baseline'

def run_method(cfg, data, method, device, result_dir, weight_dir, manifest,
               *, engine_factory=build_engine, stop_after_epoch=None):
    parse_baseline(method)
    if cfg['rho'] != 0 or cfg['flat_lambda'] != 0: raise ValueError('Baseline-only entry')
    def write(path, value):
        if path.name == 'metadata.json':
            value = dict(value, primary_reporting='Fixed Epoch100 checkpoint test',
                         coordination='Sample-weighted local endpoints' if method == 'fedavg' else 'A(mean g)',
                         seed_policy='Seed controls domain holdouts, initialization, augmentation and batch order')
        return original.json_write(path, value)
    bindings = dict(original.run_method.__globals__, parse_job=parse_baseline,
                    label=lambda m: NAMES[m], json_write=write)
    runner = FunctionType(original.run_method.__code__, bindings, 'run_baseline',
                          original.run_method.__defaults__, original.run_method.__closure__)
    runner.__kwdefaults__ = dict(original.run_method.__kwdefaults__)
    return runner(cfg, data, method, device, result_dir, weight_dir, manifest,
                  engine_factory=engine_factory, stop_after_epoch=stop_after_epoch)
