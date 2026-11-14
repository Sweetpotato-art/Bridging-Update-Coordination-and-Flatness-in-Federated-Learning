"""Shared-coordinate perturbations, following the cited papers' equations.

These are flatness-mechanism adaptations, not complete original FL algorithms.
Private heads are never perturbed or included in the global direction norm.
"""
import copy

import torch

from fl_coordination._impl.fedpairs.common import finite, moved


NAMES = {'fedgf': 'FedGF', 'fedgloss': 'FedGloSS'}


def radius_vector(vector, rho):
    finite(vector, 'Perturbation reference')
    if rho < 0:
        raise ValueError('Negative perturbation radius')
    length = vector.norm()
    if float(length) == 0 or rho == 0:
        return torch.zeros_like(vector)
    return vector * (rho / length)


class Mechanism:
    def __init__(self, name, config, device):
        if name not in NAMES:
            raise ValueError('Unknown flatness mechanism')
        self.name, self.device = name, device
        self.rho = config['rho']
        self.threshold = config['fedgf_threshold']
        self.window_size = config['fedgf_window']
        self.previous_global = None
        self.previous_p = None
        self.window = []
        self.base = self.epsilon = None
        self.c = 0.

    def begin(self, base):
        self.base = base.detach().clone()
        # Previous mean client pseudo-gradient, before server Adam.
        reference = self.previous_p
        self.epsilon = torch.zeros_like(base) if reference is None else radius_vector(reference, self.rho)
        self.c = sum(self.window) / len(self.window) if self.window else 0.

    def perturbation(self, current, clean_gradient):
        if self.rho == 0:
            return torch.zeros_like(current)
        if self.name == 'fedgf':
            local = radius_vector(clean_gradient, self.rho)
            # Interpolate perturbed MODELS, not merely normalized directions.
            # w_eval = (1-c)*(w_local + eps_local) + c*(w_global + eps_global).
            return (1-self.c)*local + self.c*(self.base + self.epsilon - current)
        # GloSS keeps a fixed shift within the communication round.
        # The paired probe also evaluates the virtual unshifted coordinates.
        return self.epsilon

    def finish(self, mean_p, endpoint_norms):
        divergence = sum(endpoint_norms) / len(endpoint_norms)
        if self.name == 'fedgf':
            self.window.append(int(divergence > self.threshold))
            self.window = self.window[-self.window_size:]
        self.previous_global = self.base.detach().clone()
        self.previous_p = mean_p.detach().clone()
        return dict(mechanism=self.name, global_perturbation_norm=float(self.epsilon.norm()),
                    fedgf_c=self.c if self.name == 'fedgf' else None,
                    client_endpoint_norm_mean=divergence,
                    fedgf_threshold=self.threshold if self.name == 'fedgf' else None)

    def state_dict(self):
        return dict(name=self.name, previous_global=copy.deepcopy(self.previous_global),
                    previous_p=copy.deepcopy(self.previous_p), window=list(self.window))

    def load_state_dict(self, state):
        if state['name'] != self.name or len(state['window']) > self.window_size:
            raise ValueError('Flatness state identity/shape mismatch')
        self.previous_global = moved(state['previous_global'], self.device)
        self.previous_p = moved(state['previous_p'], self.device)
        self.window = list(state['window'])
        self.base = self.epsilon = None
