"""Unmodified CIFAR training/validation/resume loop with isolated method names."""
from types import FunctionType
from fl_coordination._impl.fedcifar import runtime as frozen
from .engine import Engine,method_label,display_label

run_method=FunctionType(frozen.run_method.__code__,
    dict(frozen.run_method.__globals__,method_label=method_label,display_label=display_label),
    name='run_method',argdefs=frozen.run_method.__defaults__)
run_method.__kwdefaults__=dict(frozen.run_method.__kwdefaults__,engine_factory=Engine)
run_method.__module__=__name__
