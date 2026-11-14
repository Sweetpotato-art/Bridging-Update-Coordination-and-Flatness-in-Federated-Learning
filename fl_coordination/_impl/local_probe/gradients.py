"""Three evaluations: clean, historical shift, then fresh SAM at that shift.

Only shared parameters are perturbed. Every probe uses identical samples,
augmentation, random draws and pre-clean BN buffers. Keep clean BN statistics
once per batch; restore parameters/RNG even if an extra probe fails.
"""
import math
import torch
from torch.nn import functional as F
from fl_coordination._impl.fedmini.gradients import ClientGradients
from fl_coordination._impl.fedmini.model import buffers, restore_buffers, flatten
from fl_coordination._impl.fedpairs.common import rng, set_rng, finite
from fl_coordination._impl.fedglobalflat.mechanisms import radius_vector


def paired_gradients(encoder, head, batch, local_buffers, mechanism):
    rho_local = mechanism.local_sam_rho
    if not math.isfinite(rho_local) or rho_local < 0:
        raise ValueError('Invalid local SAM radius')
    encoder.train(); head.train()
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
    shift = mechanism.perturbation(flatten(original), g).detach()
    finite(shift, 'Historical evaluation shift')

    def evaluate(offset):
        with torch.no_grad():
            start = 0
            for parameter, saved in zip(shared, original):
                n = parameter.numel()
                parameter.copy_(saved + offset[start:start+n].view_as(saved))
                start += n
        set_rng(before_rng)
        restore_buffers(encoder, local_buffers)
        value = F.cross_entropy(head(encoder(x)), y)
        derivatives = [v.detach() for v in torch.autograd.grad(value, parameters)]
        for derivative in derivatives: finite(derivative, 'Probe derivative')
        return value, derivatives

    try:
        calls = 1
        if bool(torch.count_nonzero(shift)):
            shifted_loss, shifted = evaluate(shift); calls += 1
        else:
            shifted_loss, shifted = loss, clean
        shifted_g = flatten(shifted[:len(shared)])
        # Stop-gradient first-order SAM approximation; no differentiation through
        # either history or the normalized local gradient.
        epsilon_local = radius_vector(shifted_g, rho_local).detach()
        total_offset = shift + epsilon_local
        if bool(torch.count_nonzero(epsilon_local)):
            perturbed_loss, perturbed = evaluate(total_offset); calls += 1
        else:
            perturbed_loss, perturbed = shifted_loss, shifted
        p = flatten(perturbed[:len(shared)])
        h = p-g
        finite(h, 'Combined correction gradient')
        shift_norm, local_norm = float(shift.norm()), float(epsilon_local.norm())
        cosine = float(shift @ epsilon_local)/(shift_norm*local_norm) if shift_norm*local_norm else None
        mechanism.local_probe_records.append(dict(
            shift_norm=shift_norm, local_epsilon_norm=local_norm,
            historical_local_cosine=cosine, shifted_gradient_norm=float(shifted_g.norm()),
            local_probe_loss_delta=float((perturbed_loss-shifted_loss).detach())))
        upload = ClientGradients(g, h, perturbed[len(shared):], float(loss.detach()),
            float((perturbed_loss-loss).detach()), clean_buffers,
            float((g+h-p).norm()/p.norm().clamp_min(1e-12)))
        return upload, calls, float(total_offset.norm())
    finally:
        with torch.no_grad():
            for parameter, saved in zip(shared, original): parameter.copy_(saved)
        restore_buffers(encoder, clean_buffers)
        set_rng(after_rng)
