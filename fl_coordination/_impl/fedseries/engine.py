"""Same-point K=1 and trajectory-averaged K>1 coordination, without private uploads."""
import numpy as np
import torch

from fl_coordination._impl.fedmini.engine import Engine as MiniEngine
from fl_coordination._impl.fedmini.gradients import client_gradients
from fl_coordination._impl.fedmini.model import assign_gradients, flatten
from fl_coordination._impl.fedpairs.common import finite


VARIANTS = ('baseline', 'sam', 'split')


def method_label(coordinator, variant):
    base = {'fedfv': 'FedFV', 'fedmgda': 'FedMGDA+'}[coordinator]
    return {'baseline': '', 'sam': 'ASAM-', 'split': 'F-'}[variant] + base


def pair_geometry(vectors):
    dots, cosines = [], []
    for i in range(len(vectors)):
        for j in range(i):
            dot = float(vectors[i] @ vectors[j])
            denom = float(vectors[i].norm() * vectors[j].norm())
            if denom > 0:
                dots.append(dot)
                cosines.append(dot / denom)
    return dict(conflict_fraction=float(np.mean(np.asarray(dots) < 0)) if dots else None,
                mean_cosine=float(np.mean(cosines)) if cosines else None)


class Engine(MiniEngine):
    def __init__(self, config, coordinator, variant, device, network=None):
        if variant not in VARIANTS:
            raise ValueError('Unknown method variant')
        super().__init__(config, coordinator, variant != 'baseline', device, network)
        self.variant = variant

    def step(self, client_batches):
        """Only sampled clients advance their heads/BN; reset shared weights per client."""
        ids = sorted(client_batches)
        if not ids or len(ids) != len(set(ids)) or any(c < 0 or c >= len(self.heads) for c in ids):
            raise ValueError('Invalid sampled client IDs')
        if any(not client_batches[c] for c in ids):
            raise ValueError('Every participating client needs at least one batch')
        shared = list(self.encoder.parameters())
        base = [p.detach().clone() for p in shared]
        rho = self.config['rho'] if self.variant != 'baseline' else 0.
        eta = self.config['local_learning_rate']
        uploads = []
        self.step_index += 1
        try:
            for cid in ids:
                with torch.no_grad():
                    for parameter, saved in zip(shared, base):
                        parameter.copy_(saved)
                gs, hs = None, None
                clean_losses, sharpness, counts = [], [], []
                reconstruction = 0.
                for batch in client_batches[cid]:
                    upload = client_gradients(self.encoder, self.heads[cid], batch,
                                              self.local_buffers[cid], rho, self.config['adaptive_sam'])
                    self.local_buffers[cid] = upload.buffers
                    if gs is None:
                        gs, hs = upload.g.clone(), upload.h.clone()
                    else:
                        gs.add_(upload.g)
                        hs.add_(upload.h)
                    direction = upload.g + upload.h if rho > 0 else upload.g
                    # Plain local SGD makes the two trajectory sums additive.
                    # Its temporary local endpoint is never installed as the server model.
                    with torch.no_grad():
                        offset = 0
                        for parameter in shared:
                            n = parameter.numel()
                            parameter.add_(direction[offset:offset+n].view_as(parameter), alpha=-eta)
                            offset += n
                    optimizer = self.head_optimizers[cid]
                    optimizer.zero_grad(set_to_none=True)
                    for parameter, gradient in zip(self.heads[cid].parameters(), upload.head):
                        parameter.grad = gradient.clone()
                    optimizer.step()
                    counts.append(len(batch[1]))
                    clean_losses.append(upload.loss)
                    sharpness.append(upload.sharpness)
                    reconstruction = max(reconstruction, upload.reconstruction_error)
                actual_delta = flatten([b - p.detach() for b, p in zip(base, shared)])
                expected_delta = eta * (gs + hs)
                relative_error = float((actual_delta - expected_delta).norm() /
                                       expected_delta.norm().clamp_min(1e-12))
                finite(actual_delta, 'Local shared endpoint')
                k = len(client_batches[cid])
                uploads.append(dict(cid=cid, g=gs / k, h=hs / k, steps=k, samples=sum(counts),
                                    loss=float(np.average(clean_losses, weights=counts)),
                                    sharpness=float(np.average(sharpness, weights=counts)),
                                    reconstruction_error=reconstruction,
                                    parameter_reconstruction_relative_error=relative_error))
        finally:
            with torch.no_grad():
                for parameter, saved in zip(shared, base):
                    parameter.copy_(saved)

        gs, hs = [u['g'] for u in uploads], [u['h'] for u in uploads]
        losses = [u['loss'] for u in uploads]
        main = [g + h for g, h in zip(gs, hs)] if self.variant == 'sam' else gs
        dg, qp_g = self.g_coordinator(main, losses, self.step_index, ids=ids)
        dh = torch.zeros_like(dg)
        qp_h = {'solver': 'inactive'}
        if self.variant == 'split' and rho > 0 and self.config['flat_lambda'] != 0:
            dh, qp_h = self.h_coordinator(hs, [u['sharpness'] for u in uploads], self.step_index, ids=ids)
        direction = dg + self.config['flat_lambda'] * dh
        finite(direction, 'Server fused direction')
        self.shared_optimizer.zero_grad(set_to_none=True)
        assign_gradients(shared, direction)
        self.shared_optimizer.step()
        for module in [self.encoder, self.heads]:
            if not all(bool(torch.isfinite(p).all()) for p in module.parameters()):
                raise FloatingPointError('Nonfinite model after server/head optimizer')
        actual_server_delta = flatten([b - p.detach() for b, p in zip(base, shared)])
        gnorm, hnorm = float(dg.norm()), float(dh.norm())
        vector_bytes = dg.numel() * dg.element_size()
        local_steps = sum(u['steps'] for u in uploads)
        return dict(round=self.step_index, client_ids=ids,
                    client_samples={str(u['cid']): u['samples'] for u in uploads},
                    client_steps={str(u['cid']): u['steps'] for u in uploads},
                    samples=sum(u['samples'] for u in uploads), local_steps=local_steps,
                    losses=losses, sharpness=[u['sharpness'] for u in uploads],
                    backward_passes=local_steps * (2 if rho > 0 else 1),
                    uplink_vector_bytes=vector_bytes * len(ids) * (2 if self.variant == 'split' else 1),
                    downlink_parameter_bytes=vector_bytes * len(ids),
                    g_norm=gnorm, h_norm=hnorm, fused_norm=float(direction.norm()),
                    branch_cosine=float(dg @ dh) / (gnorm * hnorm) if gnorm * hnorm else None,
                    client_g_geometry=pair_geometry(gs), client_h_geometry=pair_geometry(hs),
                    actual_server_step_norm=float(actual_server_delta.norm()),
                    client_g_dot_actual_server_step=[float(g @ actual_server_delta) for g in gs],
                    reconstruction_error=max(u['reconstruction_error'] for u in uploads),
                    parameter_reconstruction_relative_error=max(u['parameter_reconstruction_relative_error'] for u in uploads),
                    qp_g=qp_g, qp_h=qp_h)
