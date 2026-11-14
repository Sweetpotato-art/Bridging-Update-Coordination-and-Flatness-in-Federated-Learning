"""Same local-probe logging, complete data epochs, validation selection and resume."""
from types import FunctionType
from fl_coordination._impl.local_probe.runtime import run_method as original
from .engine import Engine, method_label, display_label

run_method = FunctionType(original.__code__, dict(original.__globals__,
    method_label=method_label, display_label=display_label), argdefs=original.__defaults__)
run_method.__kwdefaults__ = dict(original.__kwdefaults__, engine_factory=Engine)
