"""Paired clean/perturbed derivatives with exact parameter and RNG restoration."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from fl_coordination._impl.fedpairs.common import rng, set_rng, finite
from .model import buffers, restore_buffers, flatten


@dataclass
class ClientGradients:
    g: torch.Tensor
    h: torch.Tensor
    head: list
    loss: float
    sharpness: float
    buffers: dict
    reconstruction_error: float


def asam_perturbations(parameters, gradients, rho, adaptive=True):
    """Author formula: rho*w^2*g / (|| |w|*g ||_2 + 1e-12).

    One joint norm covers the shared encoder AND this client's private head.
    In particular this is not standard SAM with a large radius of two.
    """
    norms = [(p.detach().abs() * g if adaptive else g).norm()
             for p, g in zip(parameters, gradients)]
    scale = rho / (torch.stack(norms).norm() + 1e-12)
    result = [(p.detach().square() if adaptive else 1.) * g * scale
              for p, g in zip(parameters, gradients)]
    for value in result:
        finite(value, 'ASAM perturbation')
    return result


def client_gradients(encoder, head, batch, local_buffers, rho, adaptive=True):
    """No optimizer steps here. Even heads update only after both uploads exist."""
    encoder.train()
    head.train()
    restore_buffers(encoder, local_buffers)
    shared = list(encoder.parameters())
    private = list(head.parameters())
    parameters = shared + private
    x, y = batch
    before_rng = rng()
    loss = F.cross_entropy(head(encoder(x)), y)
    clean = [g.detach() for g in torch.autograd.grad(loss, parameters)]
    g = flatten(clean[:len(shared)])
    finite(g, 'Ordinary shared gradient')
    clean_buffers = buffers(encoder)
    after_rng = rng()
    if rho == 0:
        return ClientGradients(g, torch.zeros_like(g), clean[len(shared):],
                               float(loss.detach()), 0., clean_buffers, 0.)
    if rho < 0:
        raise ValueError('rho must be nonnegative')
    original = [p.detach().clone() for p in parameters]
    delta = asam_perturbations(parameters, clean, rho, adaptive)
    try:
        with torch.no_grad():
            for parameter, saved, change in zip(parameters, original, delta):
                parameter.copy_(saved + change)
        # Replay stochastic layers, not the data iterator. Inputs are the same tensor.
        set_rng(before_rng)
        perturbed_loss = F.cross_entropy(head(encoder(x)), y)
        perturbed = [v.detach() for v in torch.autograd.grad(perturbed_loss, parameters)]
        p = flatten(perturbed[:len(shared)])
        h = p - g
        finite(h, 'Flatness shared gradient')
        for value in perturbed[len(shared):]:
            finite(value, 'Private head gradient')
        reconstruction = float((g + h - p).norm() / p.norm().clamp_min(1e-12))
        return ClientGradients(g, h, perturbed[len(shared):], float(loss.detach()),
                               float((perturbed_loss - loss).detach()), clean_buffers, reconstruction)
    finally:
        # Copy saved values instead of subtracting epsilon (avoids roundoff drift).
        with torch.no_grad():
            for parameter, saved in zip(parameters, original):
                parameter.copy_(saved)
        # Second forward uses batch statistics but must not update running state twice.
        restore_buffers(encoder, clean_buffers)
        set_rng(after_rng)
