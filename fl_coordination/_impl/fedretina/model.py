"""Versioned ImageNet backbones; all parameters trainable, independent private FCs."""
import copy
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def network_spec(config):
    name = config['backbone']
    specs = {
        'resnet18': dict(builder=resnet18, weights=ResNet18_Weights.IMAGENET1K_V1,
                         pretrained='ResNet18_Weights.IMAGENET1K_V1', features=512,
                         shared_parameters=11176512, blocks=[2, 2, 2, 2]),
    }
    if name not in specs or config['pretrained'] != specs[name]['pretrained']:
        raise ValueError('Backbone and declared ImageNet V1 checkpoint must match')
    return specs[name]


def build_network(config, *, pretrained=True):
    spec = network_spec(config)
    encoder = spec['builder'](weights=spec['weights'] if pretrained else None)
    encoder.fc = nn.Identity()
    head = nn.Linear(spec['features'], config['num_classes'])
    heads = nn.ModuleList([copy.deepcopy(head) for _ in range(config['clients'])])
    return encoder, heads
