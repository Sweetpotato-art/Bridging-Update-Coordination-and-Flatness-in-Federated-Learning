"""Standard ImageNet ResNet-18, replacing only its 1000-way classifier."""
import copy
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_network(config):
    if config['pretrained'] != 'ResNet18_Weights.IMAGENET1K_V1':
        raise ValueError('This experiment requires the declared pretrained checkpoint')
    encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc=nn.Identity()
    head=nn.Linear(512,config['num_classes'])
    heads=nn.ModuleList([copy.deepcopy(head) for _ in range(config['clients'])])
    return encoder,heads
