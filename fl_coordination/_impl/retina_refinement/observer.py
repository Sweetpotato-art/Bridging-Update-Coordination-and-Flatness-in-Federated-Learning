"""Five-client Retina weight refinement observer: synchronous round sampling, isolated evaluation."""
import copy
import math
import time
import torch
from fl_coordination._impl.fedmini.engine import Engine as EvaluationEngine
from fl_coordination._impl.fedretina.engine import Evaluation

class RetinaScratch(Evaluation, EvaluationEngine):
    pass
from fl_coordination._impl.fedmini.model import buffers, restore_buffers
from fl_coordination._impl.fedpairs.common import moved, preserve_rng
from ._averaging.core import FrozenSegment, _merge_state
from ._averaging.detectors import AveragingGrid


def flatten_state(value):
    out = {f'encoder::{k}': v for k, v in value['encoder'].items()}
    out.update({f'heads::{k}': v for k, v in value['heads'].items()})
    for cid, local in enumerate(value['local_buffers']):
        out.update({f'bn{cid}::{k}': v for k, v in local.items()})
    return out


def unflatten_state(value):
    out = dict(encoder={}, heads={}, local_buffers=[{} for _ in range(5)])
    for key, tensor in value.items():
        owner, name = key.split('::', 1)
        target = out['local_buffers'][int(owner[2:])] if owner.startswith('bn') else out[owner]
        target[name] = tensor
    return out


def current_tensors(engine):
    # State-dict tensors are only read; never modify a live training tensor.
    return flatten_state(dict(encoder=engine.encoder.state_dict(), heads=engine.heads.state_dict(),
                              local_buffers=engine.local_buffers))


def retention(validation, reference):
    values = [r['accuracy'] for r in validation['clients']]
    refs = [r['accuracy'] for r in reference['clients']]
    if len(values) != 5 or len(refs) != 5 or not all(math.isfinite(v) for v in values+refs) or min(refs) <= 0:
        raise ValueError('Five positive finite baseline validation references required')
    drops = [(b-a)/b for a, b in zip(values, refs)]
    positive = [max(0., d) for d in drops]
    return dict(signed_relative_drops=drops, worst_positive_drop=max(positive),
                mean_positive_drop=sum(positive)/5, mean_signed_drop=sum(drops)/5)


def rank(record, rule='retention'):
    s = record['scores']
    if rule == 'retention':
        key = (s['worst_positive_drop'], s['mean_positive_drop'], s['mean_signed_drop'])
    elif rule == 'delta':
        key = (s['mean_signed_drop'],)
    else:
        raise ValueError(rule)
    return key+(record['id'],)


class Pool:
    """Keep the union of top-two per original ranking; score before pruning."""
    def __init__(self):
        self.items, self.history = {}, []

    def add(self, record, state_factory):
        if any(r['id'] == record['id'] for r in self.history):
            raise ValueError('Duplicate candidate')
        self.history.append(copy.deepcopy(record))
        combined = dict(self.items, **{record['id']: record})
        keep = {r['id'] for rule in ('retention', 'delta')
                for r in sorted(combined.values(), key=lambda r: rank(r, rule))[:2]}
        if record['id'] in keep:
            combined[record['id']] = dict(record, state=state_factory())
        self.items = {k: combined[k] for k in sorted(keep)}


class RoundSampler:
    """One vote per completed server update, including matching private states."""
    def __init__(self, global_step=0):
        self.global_step = global_step
        self.begin_epoch()

    def begin_epoch(self):
        self.state, self.n = None, 0
        self.start_step = self.global_step+1

    @torch.no_grad()
    def update(self, engine):
        if engine.step_index != self.global_step+1:
            raise ValueError('Only consecutive completed communication rounds may be sampled')
        source = current_tensors(engine)
        if self.n == 0:
            self.state = moved(source)
        else:
            if self.state.keys() != source.keys():
                raise ValueError('Changing private/shared state schema')
            _merge_state(self.state, source, self.n, 1)
        self.n += 1
        self.global_step += 1

    def freeze(self, epoch, validation, initial_losses):
        if self.n <= 0:
            raise ValueError('Empty epoch sampler')
        normalized = [c['loss']/max(v, 1e-12) for c, v in zip(validation['clients'], initial_losses)]
        if len(normalized) != 5:
            raise ValueError('Exactly five client signals')
        return FrozenSegment(epoch, self.state, self.n, self.start_step, self.global_step,
                             sum(normalized)/5,
                             {f'normalized_task_loss_{i}': v for i, v in enumerate(normalized)})


def scratch_engine(engine):
    scratch = RetinaScratch.__new__(RetinaScratch)
    scratch.encoder, scratch.heads = copy.deepcopy(engine.encoder), copy.deepcopy(engine.heads)
    scratch.local_buffers = moved(engine.local_buffers, engine.device)
    scratch.config, scratch.device = engine.config, engine.device
    return scratch


@torch.no_grad()
def calibrate_bn(scratch, data):
    """Recompute each client's BN statistics using its fitting images."""
    modules = [m for m in scratch.encoder.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    modes = {m: m.training for m in scratch.encoder.modules()}
    momenta = {m: m.momentum for m in modules}
    counts = []
    try:
        for cid in range(5):
            restore_buffers(scratch.encoder, scratch.local_buffers[cid])
            scratch.encoder.eval()
            for m in modules:
                m.reset_running_stats()
                m.train()
            count = 0
            ids = data.indices['fit'][cid]
            bs = scratch.config['eval_batch_size']
            for offset in range(0, len(ids), bs):
                # Training IDs; deterministic center crop, not validation/test membership.
                x, _ = data.batch('bn_calibration', ids[offset:offset+bs], scratch.device, cid=cid)
                size = len(x)
                for m in modules:
                    m.momentum = size/(count+size)
                scratch.encoder(x)
                count += size
            if count != len(ids) or count == 0:
                raise ValueError('Incomplete BN fitting-data pass')
            scratch.local_buffers[cid] = buffers(scratch.encoder)
            counts.append(count)
    finally:
        for m, value in momenta.items():
            m.momentum = value
        for m, value in modes.items():
            m.training = value
    return counts


class Observer:
    def __init__(self, engine, data, reference, settings, initial_losses=None):
        self.scratch = scratch_engine(engine)
        self.data, self.reference = data, reference
        self.grid = AveragingGrid(None, settings)
        self.sampler = RoundSampler()
        self.pools = {s['id']: Pool() for s in settings}
        self.timing = dict(segment_validation_seconds=0., candidate_seconds=0., candidates=0)
        if initial_losses is None:
            with preserve_rng():
                initial_losses = [c['loss'] for c in self.scratch.evaluate(data, 'validation')['clients']]
        if len(initial_losses) != 5 or any(not math.isfinite(v) or v < 0 for v in initial_losses):
            raise ValueError('Invalid initialization validation losses')
        self.initial_losses = initial_losses

    def begin_epoch(self):
        self.sampler.begin_epoch()

    def on_round(self, engine):
        self.sampler.update(engine)

    def end_epoch(self, epoch):
        begin = time.monotonic()
        with preserve_rng():
            self.scratch.load_model(unflatten_state(self.sampler.state))
            val = self.scratch.evaluate(self.data, 'validation')
        self.timing['segment_validation_seconds'] += time.monotonic()-begin
        self.grid.observe(self.sampler.freeze(epoch, val, self.initial_losses))
        self.grid.drain(self.consume)
        return val

    def consume(self, variant, candidate):
        if candidate.segment_count < 3:
            raise ValueError('Ineligible valley')
        begin = time.monotonic()
        with preserve_rng():
            self.scratch.load_model(unflatten_state(candidate.state))
            counts = calibrate_bn(self.scratch, self.data)
            val = self.scratch.evaluate(self.data, 'validation')
            meta = candidate.metadata()
            meta['weighting'] = 'one_vote_per_completed_communication_round'
            record = dict(id=f'{candidate.grid_id}/{candidate.valley_id}', metadata=meta,
                          validation=val, scores=retention(val, self.reference),
                          bn_calibrated_on='domain_owned_fit_only_center_crop', calibration_samples=counts)
            self.pools[candidate.grid_id].add(record, lambda: moved(self.scratch.model_state()))
        self.timing['candidate_seconds'] += time.monotonic()-begin
        self.timing['candidates'] += 1

    def finalize(self):
        self.grid.finalize()
        self.grid.drain(self.consume)
        candidates = [r for pool in self.pools.values() for r in pool.items.values()]
        return min(candidates, key=rank) if candidates else None

    def audit(self):
        return dict(initial_validation_losses=self.initial_losses, timing=self.timing,
                    candidates=[r for pool in self.pools.values() for r in pool.history],
                    detection=self.grid.diagnostics())

    def state_dict(self):
        # Preserve shared tensor bank references; torch.save performs one memoized serialization.
        return dict(grid=self.grid.state_dict(), pools=self.pools,
                    initial_losses=self.initial_losses, timing=self.timing)

    def load_state_dict(self, state):
        if self.initial_losses != state['initial_losses']:
            raise ValueError('Normalization reference changed')
        self.grid.load_state_dict(state['grid'])
        self.pools, self.timing = state['pools'], state['timing']
        self.sampler = RoundSampler(self.grid.last_step)
