"""CIFAR-100 training and checkpointing with FedAvg."""
from types import FunctionType

from fl_coordination._impl.fedcifar import runtime as frozen
from fl_coordination._impl.fedpairs.common import json_write
from .engine import Engine, method_label, display_label


def write_result(path,value):
    if path.name in ('metadata.json','final.json'):
        value.update(aggregation='sample_weighted_local_shared_endpoints',server_optimizer='none')
    if path.name == 'metadata.json':
        counts = value['config']['client_fit_sizes']
        value.update(protocol='CIFAR-100 FedAvg with shared trainable parameters and client-local BN',
                     aggregation_weights=[n/sum(counts) for n in counts],
                     weight_basis='complete_client_fit_size_excluding_validation',
                     local_shared_optimizer='SGD',private_head_optimizer=(
                         'Adam' if value['config']['head_mode']=='private' else 'none'))
    json_write(path,value)


run_method = FunctionType(frozen.run_method.__code__,
    dict(frozen.run_method.__globals__,method_label=method_label,display_label=display_label,json_write=write_result),
    name='run_method',argdefs=frozen.run_method.__defaults__)
run_method.__kwdefaults__ = dict(frozen.run_method.__kwdefaults__,engine_factory=Engine)
run_method.__module__ = __name__
bind_experiment = frozen.bind_experiment
