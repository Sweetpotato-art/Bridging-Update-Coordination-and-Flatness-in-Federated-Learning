"""ResNet-20 backbone and classifiers for CIFAR-100."""
import ast
import copy
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from fl_coordination.dependencies import author_file


@lru_cache(None)
def author_resnet20():
    path = author_file('FedAWA_ResNet20/resnet.py')
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    names = {'BasicBlock', 'ResNet_cifar', 'ResNet20'}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError('Pinned ResNet-20 definitions are incomplete')
    # Execute the original definitions verbatim, without unrelated model imports.
    scope = dict(torch=torch, nn=nn, F=F, __name__='fedcifar.author_resnet20')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['ResNet20']


def build_network(clients=20, head_mode='private', num_classes=100):
    if clients < 1 or head_mode not in ('private', 'shared') or num_classes < 2:
        raise ValueError('Invalid network configuration')
    network = author_resnet20()(num_classes)
    if head_mode == 'private':
        classifier = network.linear
        network.linear = nn.Identity()
        # Identical initialization, distinct parameters and private Adam states.
        heads = nn.ModuleList([copy.deepcopy(classifier) for _ in range(clients)])
    else:
        # The classifier is included in both shared ASAM branch vectors.
        heads = nn.ModuleList([nn.Identity() for _ in range(clients)])
    return network, heads
