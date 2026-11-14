"""Five unchanged first-order coordinators, two independent branch states."""
import numpy as np
import torch

from fl_coordination._impl.feda3.engine import Engine as BaseEngine, NAMES as FIRST_ORDER
from fl_coordination._impl.fedmini.model import assign_gradients, flatten
from fl_coordination._impl.fedpairs.common import finite
from fl_coordination._impl.fedseries.engine import pair_geometry
from .gradients import paired_gradients
from .mechanisms import Mechanism, NAMES


def method_label(coordinator, mechanism):
    return FIRST_ORDER[coordinator] + ' & ' + NAMES[mechanism]


class Engine(BaseEngine):
    def __init__(self, config, coordinator, variant, device, network=None):
        super().__init__(config, coordinator, 'split', device, network)
        self.mechanism = Mechanism(variant, config, device)

    def state_dict(self):
        return dict(super().state_dict(), flatness=self.mechanism.state_dict())

    def load_state_dict(self, value):
        super().load_state_dict(value)
        self.mechanism.load_state_dict(value['flatness'])

    def step(self, client_batches):
        ids = sorted(client_batches)
        lengths = {len(v) for v in client_batches.values()}
        if ids != list(range(self.config['clients'])) or len(lengths) != 1 or 0 in lengths:
            raise ValueError('This controlled comparison requires all five clients and equal local steps')
        shared = list(self.encoder.parameters())
        base = [p.detach().clone() for p in shared]
        base_vector = flatten(base)
        self.mechanism.begin(base_vector)
        eta = self.config['local_learning_rate']
        if hasattr(self.g_coordinator, 'bind_context'):
            scale = eta * next(iter(lengths))
            self.g_coordinator.bind_context(base_vector, scale)
            self.h_coordinator.bind_context(base_vector, scale)
        self.step_index += 1
        uploads = []
        try:
            for cid in ids:
                with torch.no_grad():
                    for parameter, saved in zip(shared, base):
                        parameter.copy_(saved)
                gs, hs = torch.zeros_like(base_vector), torch.zeros_like(base_vector)
                clean_losses, sharpness, counts, perturbations = [], [], [], []
                backwards, reconstruction = 0, 0.
                for batch in client_batches[cid]:
                    upload, calls, perturb_norm = paired_gradients(
                        self.encoder, self.heads[cid], batch, self.local_buffers[cid], self.mechanism)
                    self.local_buffers[cid] = upload.buffers
                    gs.add_(upload.g)
                    hs.add_(upload.h)
                    direction = upload.g + upload.h
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
                    perturbations.append(perturb_norm)
                    backwards += calls
                    reconstruction = max(reconstruction, upload.reconstruction_error)
                delta = flatten([b-p.detach() for b, p in zip(base, shared)])
                expected = eta*(gs+hs)
                finite(delta, 'Local shared endpoint')
                relative_error = float((delta-expected).norm()/expected.norm().clamp_min(1e-12))
                k = len(client_batches[cid])
                uploads.append(dict(cid=cid, g=gs/k, h=hs/k, steps=k, samples=sum(counts),
                                    loss=float(np.average(clean_losses, weights=counts)),
                                    sharpness=float(np.average(sharpness, weights=counts)),
                                    perturbation_norm=float(np.mean(perturbations)),
                                    backward_passes=backwards, endpoint_norm=float(expected.norm()),
                                    reconstruction_error=reconstruction,
                                    parameter_reconstruction_relative_error=relative_error))
        finally:
            with torch.no_grad():
                for parameter, saved in zip(shared, base):
                    parameter.copy_(saved)
        gs, hs = [u['g'] for u in uploads], [u['h'] for u in uploads]
        losses = [u['loss'] for u in uploads]
        dg, qp_g = self.g_coordinator(gs, losses, self.step_index, ids=ids)
        dh, qp_h = torch.zeros_like(dg), {'solver': 'inactive_zero_flatness'}
        if self.config['flat_lambda'] != 0 and any(bool(torch.count_nonzero(h)) for h in hs):
            dh, qp_h = self.h_coordinator(hs, [u['sharpness'] for u in uploads], self.step_index, ids=ids)
        direction = dg+self.config['flat_lambda']*dh
        finite(direction, 'Fused server direction')
        self.shared_optimizer.zero_grad(set_to_none=True)
        assign_gradients(shared, direction)
        self.shared_optimizer.step()
        for module in (self.encoder, self.heads):
            if not all(bool(torch.isfinite(p).all()) for p in module.parameters()):
                raise FloatingPointError('Nonfinite parameters after optimizer step')
        # Raw endpoint mean is kept separately from both coordinated directions
        # and the Adam-preconditioned actual update. No extra coordinator state.
        mean_p = torch.zeros_like(dg)
        for g, h in zip(gs, hs):
            mean_p.add_(g+h, alpha=1/len(gs))
        mechanism_info = self.mechanism.finish(mean_p, [u['endpoint_norm'] for u in uploads])
        server_delta = flatten([b-p.detach() for b, p in zip(base, shared)])
        gn, hn = float(dg.norm()), float(dh.norm())
        vector_bytes = dg.numel()*dg.element_size()
        # GloSS split needs both the clean reference and perturbed broadcast;
        down_vectors = 2
        return dict(round=self.step_index, client_ids=ids,
                    client_samples={str(u['cid']):u['samples'] for u in uploads},
                    client_steps={str(u['cid']):u['steps'] for u in uploads},
                    samples=sum(u['samples'] for u in uploads), local_steps=sum(u['steps'] for u in uploads),
                    losses=losses, sharpness=[u['sharpness'] for u in uploads],
                    perturbation_norms=[u['perturbation_norm'] for u in uploads],
                    backward_passes=sum(u['backward_passes'] for u in uploads),
                    uplink_vector_bytes=vector_bytes*len(ids)*2,
                    downlink_parameter_bytes=vector_bytes*len(ids)*down_vectors,
                    g_norm=gn, h_norm=hn, fused_norm=float(direction.norm()),
                    branch_cosine=float(dg@dh)/(gn*hn) if gn*hn else None,
                    client_g_geometry=pair_geometry(gs), client_h_geometry=pair_geometry(hs),
                    actual_server_step_norm=float(server_delta.norm()),
                    reconstruction_error=max(u['reconstruction_error'] for u in uploads),
                    parameter_reconstruction_relative_error=max(u['parameter_reconstruction_relative_error'] for u in uploads),
                    qp_g=qp_g, qp_h=qp_h, flatness=mechanism_info)
