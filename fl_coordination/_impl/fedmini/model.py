"""Fashion-MNIST ResNet-18; shared parameters and client-local BN buffers."""
import copy
from contextlib import contextmanager

import torch
from torch import nn
from torchvision.models import resnet18


def build_network(clients=2):
    encoder = resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    encoder.fc = nn.Linear(512, 100)
    # The ReLU follows encoder.fc, exactly as in the pinned F-MTL forward.
    encoder = nn.Sequential(encoder, nn.ReLU())
    heads = nn.ModuleList([nn.Linear(100, 10) for _ in range(clients)])
    return encoder, heads


def buffers(module):
    return {name: value.detach().clone() for name, value in module.named_buffers()}


@torch.no_grad()
def restore_buffers(module, state):
    current = dict(module.named_buffers())
    if current.keys() != state.keys():
        raise ValueError('BN buffer schema differs from checkpoint')
    for name, value in state.items():
        current[name].copy_(value)


def flatten(values):
    return torch.cat([v.detach().reshape(-1) for v in values])


def assign_gradients(parameters, vector):
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.grad = vector[offset:offset + size].view_as(parameter).clone()
        offset += size
    if offset != vector.numel():
        raise ValueError('Shared gradient size differs from shared parameter count')


@contextmanager
def evaluating(encoder, heads):
    state = buffers(encoder)
    modes = [(module, module.training) for root in [encoder, heads] for module in root.modules()]
    encoder.eval()
    heads.eval()
    try:
        yield
    finally:
        restore_buffers(encoder, state)
        for module, mode in modes:
            module.training = mode


def clone_state(encoder, heads, local_buffers):
    return copy.deepcopy(dict(encoder=encoder.state_dict(), heads=heads.state_dict(),
                              local_buffers=local_buffers))
