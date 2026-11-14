import copy
import random
import types
import numpy as np
import torch
from torch.utils.data import Subset
from . import upstream
from .common import rng, set_rng, gradients, perturb, flat_dict, diagnostics, finite


def seed(value):
    random.seed(1 + value)
    np.random.seed(21 + value)
    torch.manual_seed(12 + value)
    torch.cuda.manual_seed_all(123 + value)


def options():
    return dict(eval_interval=1, num_rounds=2000, proportion=.1, sample='uniform', aggregate='uniform',
                learning_rate=.1, alpha=.1, tau=10, num_epochs=1, batch_size=-1, momentum=0)


class ZeroModel:
    def __init__(self, template):
        self.state = {k: torch.zeros_like(v) for k, v in template.items()}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.state = state


class Coordinator:
    """Transport shim; all projection/history/norm math executes in author code."""
    def __init__(self, native, n, alpha=.1, tau=10):
        self.server = object.__new__(native.Server)
        s = self.server
        s.num_clients, s.alpha, s.tau = n, alpha, tau
        s.client_last_sample_round = [-1] * n
        s.client_grads_history = [0] * n
        self.native = native

    def __call__(self, updates, losses, ids, round_index):
        s = self.server
        s.model = ZeroModel(updates[0])
        s.sample = lambda: ids
        by_id = {cid: i for i, cid in enumerate(ids)}
        s.communicate = lambda cid: ({k: -v for k, v in updates[by_id[cid]].items()}, losses[by_id[cid]])
        if all(float(flat_dict(g).norm()) <= 1e-12 for g in updates):
            for cid, g in zip(ids, updates):
                s.client_last_sample_round[cid] = round_index
                s.client_grads_history[cid] = copy.deepcopy(g)
            return s.model.state_dict()
        s.iterate(round_index)
        d = {k: -v for k, v in s.model.state_dict().items()}
        finite(flat_dict(d), 'FedFV direction')
        return d

    def state_dict(self):
        return {k: getattr(self.server, k) for k in ['client_last_sample_round', 'client_grads_history']}

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self.server, k, v)
