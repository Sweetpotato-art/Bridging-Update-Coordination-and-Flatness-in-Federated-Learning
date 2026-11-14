"""Use pinned author operators on shared parameter coordinates only."""
import torch

from fl_coordination._impl.fedpairs import upstream
from fl_coordination._impl.fedpairs.fv import Coordinator as AuthorFV
from fl_coordination._impl.fedpairs.mgda import coefficients
from fl_coordination._impl.fedpairs.common import finite


class FedFV:
    def __init__(self, config, device):
        self.native = upstream.fv(device)
        self.operator = AuthorFV(self.native, config['clients'],
                                 config['fedfv_alpha'], config['fedfv_tau'])

    def __call__(self, vectors, losses, step, ids=None):
        ids = list(range(len(vectors))) if ids is None else ids
        for v in vectors:
            finite(v, 'FedFV input')
        updates = [{'shared': v} for v in vectors]
        raw = sum(v / len(vectors) for v in vectors)
        if float(raw.norm()) == 0.:
            # The original norm restoration prescribes a zero output norm here.
            # Avoid 0/0 for exactly opposite/all-zero vectors; keep history semantics.
            s = self.operator.server
            for cid, update in zip(ids, updates):
                s.client_last_sample_round[cid] = step
                s.client_grads_history[cid] = {k: v.clone() for k, v in update.items()}
            return torch.zeros_like(raw), {'solver': 'zero_raw_average'}
        result = self.operator(updates, losses, ids, step)['shared']
        finite(result, 'FedFV output')
        return result, {'solver': 'author_fedfv'}

    def state_dict(self):
        return self.operator.state_dict()

    def load_state_dict(self, value):
        self.operator.load_state_dict(value)


class FedMGDAPlus:
    def __init__(self, config, device):
        if config['mgda_epsilon'] != 1.:
            raise ValueError('This recipe pins author epsilon=1; do not silently change the QP')
        self.native = upstream.mgda()

    def __call__(self, vectors, losses, step, ids=None):
        normalized = []
        for vector in vectors:
            finite(vector, 'FedMGDA+ input')
            norm = vector.norm()
            finite(norm, 'FedMGDA+ input norm')
            # Preserve an exact zero vector in the same convex problem. No random
            # direction, magnitude floor, diagonal shift or branch rescaling.
            normalized.append(vector / norm if float(norm) > 0 else torch.zeros_like(vector))
        if all(float(v.norm()) == 0 for v in normalized):
            return torch.zeros_like(vectors[0]), {'solver': 'all_zero',
                                                  'weights': [1. / len(vectors)] * len(vectors)}
        info = {}
        weights = coefficients(self.native, [{'shared': v} for v in normalized],
                               robust=True, diagnostics=info)
        result = torch.zeros_like(normalized[0])
        for weight, vector in zip(weights, normalized):
            result.add_(vector * float(weight))
        finite(result, 'FedMGDA+ output')
        info['weights'] = [float(w) for w in weights]
        return result, info

    def state_dict(self):
        return {}

    def load_state_dict(self, value):
        if value:
            raise ValueError('FedMGDA+ has no history state')


def make_coordinator(name, config, device):
    return {'fedfv': FedFV, 'fedmgda': FedMGDAPlus}[name](config, device)
