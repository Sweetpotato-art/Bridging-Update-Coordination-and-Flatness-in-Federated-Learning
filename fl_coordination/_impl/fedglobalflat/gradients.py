"""Clean and perturbed derivatives on identical samples and stochastic draws."""
import torch
from torch.nn import functional as F

from fl_coordination._impl.fedmini.gradients import ClientGradients
from fl_coordination._impl.fedmini.model import buffers, restore_buffers, flatten
from fl_coordination._impl.fedpairs.common import rng, set_rng, finite


def paired_gradients(encoder, head, batch, local_buffers, mechanism):
    encoder.train()
    head.train()
    restore_buffers(encoder, local_buffers)
    shared, private = list(encoder.parameters()), list(head.parameters())
    parameters = shared + private
    x, y = batch
    before_rng = rng()
    loss = F.cross_entropy(head(encoder(x)), y)
    clean = [v.detach() for v in torch.autograd.grad(loss, parameters)]
    g = flatten(clean[:len(shared)])
    finite(g, 'Clean shared gradient')
    clean_buffers, after_rng = buffers(encoder), rng()
    original = [p.detach().clone() for p in shared]
    delta = mechanism.perturbation(flatten(original), g)
    finite(delta, 'Flatness perturbation')
    if not bool(torch.count_nonzero(delta)):
        upload = ClientGradients(g, torch.zeros_like(g), clean[len(shared):],
                                 float(loss.detach()), 0., clean_buffers, 0.)
        return upload, 1, 0.
    try:
        with torch.no_grad():
            offset = 0
            for parameter, saved in zip(shared, original):
                n = parameter.numel()
                parameter.copy_(saved + delta[offset:offset+n].view_as(saved))
                offset += n
        set_rng(before_rng)
        # Replay both stochastic state AND pre-clean running buffers. Running
        # buffers retained for evaluation are the clean pass, once per batch,
        # matching the existing Fashion protocol (heads remain local).
        restore_buffers(encoder, local_buffers)
        perturbed_loss = F.cross_entropy(head(encoder(x)), y)
        perturbed = [v.detach() for v in torch.autograd.grad(perturbed_loss, parameters)]
        p = flatten(perturbed[:len(shared)])
        h = p-g
        for value in [h, *perturbed[len(shared):]]:
            finite(value, 'Perturbed derivative')
        upload = ClientGradients(g, h, perturbed[len(shared):], float(loss.detach()),
                                 float((perturbed_loss-loss).detach()), clean_buffers,
                                 float((g+h-p).norm()/p.norm().clamp_min(1e-12)))
        return upload, 2, float(delta.norm())
    finally:
        with torch.no_grad():
            for parameter, saved in zip(shared, original):
                parameter.copy_(saved)
        restore_buffers(encoder, clean_buffers)
        set_rng(after_rng)
