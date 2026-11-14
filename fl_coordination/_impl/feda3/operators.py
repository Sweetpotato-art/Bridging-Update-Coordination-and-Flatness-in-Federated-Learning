"""Shared-coordinate aggregation adapters for Fashion-MNIST."""
import contextlib
import copy
import io
from types import SimpleNamespace

import numpy as np
import torch

from fl_coordination._impl.fedmini.coordinators import make_coordinator as original_coordinator
from fl_coordination._impl.fedpairs.common import finite, moved
from .upstream import awa_functions, mdfg_functions
from .mdfg_qp import solve as solve_mdfg_qp


class FedMDFGAgg:
    """Fairness-guided direction with independent expected-objective history per branch.
    """
    def __init__(self, config, device, layout=None):
        self.device, self.n = device, config['clients']
        self.theta = config['operators']['fedmdfg']['theta']
        self.native = mdfg_functions()
        solve = self.native.setup_qp_and_solve
        self.qp_diagnostics = {}
        def checked_solve(vectors):
            coefficients, self.qp_diagnostics = solve_mdfg_qp(solve,vectors)
            return coefficients, 1
        self.native.setup_qp_and_solve = checked_solve
        self.expected = [None]*self.n
        self.join_count = [0]*self.n

    def __call__(self, vectors, losses, step, ids=None):
        if ids != list(range(self.n)):
            raise ValueError('FedMDFG-Agg requires all client vectors in client order')
        grads = torch.stack(vectors)
        value = torch.as_tensor(losses, dtype=grads.dtype, device=self.device)
        finite(grads, 'FedMDFG inputs'); finite(value, 'FedMDFG objective values')
        increased = 0
        for i in ids:
            if self.join_count[i] == 0:
                self.expected[i] = value[i].detach().clone()
            elif bool(value[i] <= self.expected[i]):
                self.expected[i] = (self.expected[i]*self.join_count[i]+value[i])/(self.join_count[i]+1)
            else:
                increased += 1
            self.join_count[i] += 1
        force = 0 < increased < self.n
        norms = grads.norm(dim=1)
        live = torch.where(norms > 1e-6)[0]  # Threshold for including directions in the quadratic program.
        diagnostics = dict(solver='author_fedmdfg_qp', fairness_forced=bool(force),
                           live_clients=live.cpu().tolist(), line_search=False)
        zero = torch.zeros_like(grads[0])
        if not len(live):
            return zero, dict(diagnostics, solver='author_no_live_gradients')
        grads, value, norms = grads[live], value[live], norms[live]
        grads = grads / norms[:,None] * norms.mean()
        if float(value.norm()) == 0:
            raise FloatingPointError('Nonzero FedMDFG gradients with zero objective vector: fairness undefined')
        guidance = torch.ones(len(live), dtype=grads.dtype, device=self.device)
        normalized_value, normalized_guidance = value/value.norm(), guidance/guidance.norm()
        cosine = float(normalized_value @ normalized_guidance)
        bias = float(np.arccos(np.clip(cosine,-1,1))/np.pi*180)
        active = bias > self.theta or force
        if active:
            fairness_axis = (normalized_guidance @ normalized_value)*normalized_value-normalized_guidance
            if float(fairness_axis.norm()) == 0:
                raise FloatingPointError('Active FedMDFG fairness derivative is zero: author normalization undefined')
        d, vec, active, _ = self.native.get_fedmdfg_d(grads, value, None, self.theta,
                                                    guidance, force, self.device)
        diagnostics['qp'] = dict(self.qp_diagnostics)
        finite(d, 'FedMDFG QP solution')
        dnorm = d.norm()
        if float(dnorm) == 0:
            return zero, dict(diagnostics, solver='zero_qp_direction', fairness_active=bool(active))
        weights = torch.full((len(live),), 1/len(live), dtype=grads.dtype, device=self.device)
        target_norm = (weights @ grads).norm()
        old = d.clone()
        d = d/dnorm*target_norm
        # Check the descent constraints after restoring the direction norm.
        while bool(torch.max(-(vec @ d)) > 1e-6):
            if bool(d.norm() > dnorm*2):
                d /= 2
            else:
                d = old
                break
        finite(d, 'FedMDFG rescaled direction')
        return d, dict(diagnostics, fairness_active=bool(active), fairness_angle_degrees=bias,
                       direction_norm=float(d.norm()))

    def state_dict(self):
        return dict(expected=copy.deepcopy(self.expected), join_count=list(self.join_count))

    def load_state_dict(self, value):
        if len(value['expected']) != self.n or len(value['join_count']) != self.n:
            raise ValueError('FedMDFG client-state shape mismatch')
        self.expected = moved(value['expected'], self.device)
        self.join_count = list(value['join_count'])


class FedAWAAgg:
    """Fit branch weights to shared-parameter pseudo-endpoints w - eta*K*v.

    The weights available at the start of the round combine the branch vectors.
    """
    def __init__(self, config, device, layout=None):
        self.device, self.n = device, config['clients']
        self.args = SimpleNamespace(**config['operators']['fedawa'])
        if (self.args.server_epochs, self.args.server_optimizer, self.args.gamma, self.args.reg_distance) != (1,'adam',1.,'cos'):
            raise ValueError('FedAWA-Agg pins the released default weight-optimizer settings')
        self.native = awa_functions()
        # Equal fitting-set sizes give each client the same initial logit.
        self.logits = torch.full((self.n,), 1/self.n, device=device)
        self.context = None

    def bind_context(self, base, scale):
        if scale <= 0:
            raise ValueError('Pseudo-endpoint scale must be positive')
        self.context = (base.detach(), float(scale))

    def __call__(self, vectors, losses, step, ids=None):
        if ids != list(range(self.n)) or self.context is None:
            raise ValueError('FedAWA-Agg requires full participation and current shared model context')
        matrix = torch.stack(vectors)
        finite(matrix, 'FedAWA inputs')
        base, scale = self.context
        if matrix.shape[1:] != base.shape:
            raise ValueError('FedAWA shared-coordinate mismatch')
        used_weights = torch.softmax(self.logits, dim=0).detach().clone()
        parameters = [{'flat_w': base - scale*v} for v in vectors]
        central = SimpleNamespace(model=SimpleNamespace(get_param=lambda: {'flat_w': base}))
        with contextlib.redirect_stdout(io.StringIO()):
            merged, new_logits = self.native.fedawa(self.args, parameters, [1/self.n]*self.n,
                                                    central, step, self.logits.detach().clone())
        finite(new_logits, 'FedAWA persistent logits')
        finite(merged['flat_w'], 'FedAWA author merged endpoint')
        self.logits = new_logits.detach().clone()
        # In the released server_epochs=1 code, probability_train is evaluated
        # BEFORE the sole optimizer step; the updated logits apply next round.
        # Sum directions directly, avoiding cancellation in (w-merged_w)/scale.
        direction = used_weights @ matrix
        finite(direction, 'FedAWA direction')
        error = float((merged['flat_w']-(base-scale*direction)).norm()/base.norm().clamp_min(1e-12))
        return direction, dict(solver='author_fedawa_release', weights=used_weights.cpu().tolist(),
                               next_weights=torch.softmax(self.logits,0).cpu().tolist(),
                               pseudo_endpoint_scale=scale, endpoint_relative_error=error,
                               weight_application='release_pre_step', regularizer='weighted_client_model_cosine')

    def state_dict(self):
        return dict(logits=self.logits.detach().clone())

    def load_state_dict(self, value):
        self.logits = moved(value['logits'], self.device)
        if self.logits.shape != (self.n,):
            raise ValueError('FedAWA state shape mismatch')
        self.context = None


class FedLDAgg:
    """Published per-parameter pcgrad_svd, restricted to shared trainable parameters.

    Preserve first-use mean, history gate/EMA, 80% axes, spectrum normalization,
    and scalar-parameter averaging. Zero projections contribute zero; this defines
    the release's 0/0 cases without adding epsilon mass or changing finite cases.
    """
    def __init__(self, config, device, layout):
        self.device, self.n = device, config['clients']
        options = config['operators']['fedld']
        self.proportion, self.decay = options['k_proportion'], options['history_decay']
        self.layout = list(layout)
        self.history = {name: None for name, _ in self.layout}
        if not self.layout or not 1 <= int(self.n*self.proportion) <= self.n or self.decay != .99:
            raise ValueError('Invalid FedLD author settings/layout')

    def __call__(self, vectors, losses, step, ids=None):
        if ids != list(range(self.n)):
            raise ValueError('FedLD-Agg requires all client vectors')
        matrix = torch.stack(vectors)
        finite(matrix, 'FedLD inputs')
        if matrix.shape[1] != sum(size for _,size in self.layout):
            raise ValueError('FedLD parameter slices do not match uploaded coordinates')
        start, result, svd_layers, guarded = 0, [], 0, 0
        for name, size in self.layout:
            layer = matrix[:,start:start+size]
            start += size
            history = self.history[name]
            mean = layer.mean(0)
            if history is None or not bool(torch.count_nonzero(history)) or size == 1:
                new = mean
            elif not bool(torch.count_nonzero(layer)):
                new = torch.zeros_like(mean)
                guarded += 1
            else:
                gram = (1/len(layer)) * layer @ layer.T
                _, spectrum, eigenvectors = torch.svd(gram)
                axes = (layer.T @ eigenvectors).T
                k = int(len(spectrum)*self.proportion)
                new = torch.zeros_like(mean)
                lengths = layer.norm(dim=1)
                if float(spectrum.sum()) == 0:
                    guarded += 1
                else:
                    for j in range(k):
                        axis = axes[j]
                        products = layer @ axis
                        if int((products >= 0).sum()) < int((products < 0).sum()):
                            axis, products = -axis, -products
                        axis_norm = axis.norm()
                        if float(axis_norm) == 0:
                            guarded += 1
                            continue
                        guarded += int((products == 0).sum())
                        coefficient = (products.sign()*lengths).mean() * spectrum[j]/spectrum.sum()
                        new.add_(axis/axis_norm*coefficient)
                svd_layers += 1
            finite(new, 'FedLD revised parameter direction')
            self.history[name] = new.detach().clone() if history is None else self.decay*history+(1-self.decay)*new
            result.append(new)
        direction = torch.cat(result)
        return direction, dict(solver='author_fedld_parameter_svd', svd_parameters=svd_layers,
                               selected_axes=int(self.n*self.proportion), zero_projection_guards=guarded)

    def state_dict(self):
        return dict(history=copy.deepcopy(self.history), layout=self.layout)

    def load_state_dict(self, value):
        if value['layout'] != self.layout or set(value['history']) != set(self.history):
            raise ValueError('FedLD shared-parameter schema changed')
        self.history = moved(value['history'], self.device)


def make_coordinator(name, config, device, layout):
    if name in ('fedfv','fedmgda'):
        return original_coordinator(name, config, device)
    return {'fedmdfg':FedMDFGAgg,'fedawa':FedAWAAgg,'fedld':FedLDAgg}[name](config,device,layout)
