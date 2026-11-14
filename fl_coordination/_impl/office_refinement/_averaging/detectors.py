"""weight refinement: task-stability entry and sustained-descent segmentation.

Shared bounded snapshots are merged in update order; rollback never subtracts
averaged tensors. Entry observes three epochs. Descent uses six epochs and three
consecutive confirmations. Candidates are evaluated before bounded-pool pruning.
"""
from dataclasses import replace
import math

from .core import (DenseEpochSampler, FrozenSegment, LossValleyDetector,
                        ValleyCandidate, WeightedStateAverage)

VARIANTS = ('weight_refinement',)
REVISION = 'weight_refinement_officehome_four_domains_v1'
EPSILON = 1e-12


def line(values):
    """Small O(W) scalar OLS; returns slope and residual standard deviation."""
    n = len(values)
    if n < 3 or not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Trend calculation needs at least three finite values')
    center, mean = (n - 1) / 2., sum(values) / n
    xx = sum((j - center) ** 2 for j in range(n))
    slope = sum((j - center) * (v - mean) for j, v in enumerate(values)) / xx
    residual = sum((v - mean - slope * (j - center)) ** 2 for j, v in enumerate(values))
    return slope, math.sqrt(max(0., residual) / (n - 2)), xx


def stability(segments):
    series = [[s.metrics[f'normalized_task_loss_{task}'] for s in segments]
              for task in range(4)]
    fits = [line(values) for values in series]
    squared_change = sum(b * b for b, _, _ in fits)
    noise = sum(sigma * sigma / xx for _, sigma, xx in fits)
    return dict(task_slopes=[f[0] for f in fits], squared_task_change=squared_change,
                slope_noise=noise, stable=squared_change <= noise + EPSILON ** 2)


class StabilityDetector(LossValleyDetector):
    """Experiment 3 replaces only entry, preserving loss-based exit and flush."""
    def observe(self, segment):
        if self.finalized:
            raise RuntimeError('Cannot observe a finalized detector')
        if self.dead_valley:
            return
        # Once entered, the pinned implementation handles all exit/averaging.
        if self.converge_step is not None:
            return super().observe(segment)
        WeightedStateAverage._validate_segment(segment)
        if (segment.start_step <= self._last_observed_step
                or segment.epoch <= self._last_observed_epoch
                or not math.isfinite(segment.end_loss)):
            raise ValueError('Invalid or out-of-order segment')
        self._last_observed_step, self._last_observed_epoch = segment.end_step, segment.epoch
        self.converge_queue.append(segment)
        self.smooth_queue.append(segment)
        if len(self.converge_queue) < self.n_converge:
            return
        if not stability(list(self.converge_queue))['stable']:
            return
        first = self.converge_queue[0]
        self.converge_step = first.end_step
        self.final_average = WeightedStateAverage(first)
        self.threshold = (sum(s.end_loss for s in self.converge_queue) / self.n_converge
                          * (1. + self.tolerance_ratio))
        if self.n_tolerance < self.n_converge:
            for index in range(self.n_converge - self.n_tolerance):
                self.final_average.update(self.converge_queue[1 + index])


class DeferredController:
    def __init__(self, settings, variant, bank, delay):
        self.settings, self.variant = dict(settings), variant
        self.bank, self.delay = bank, delay
        self.pending, self.events = [], []
        self.valley_index = 1
        self.finalized = False
        self.last_epoch = 0
        self._reset()

    def _reset(self):
        cls = StabilityDetector
        s = self.settings
        self.detector = cls(s['n_converge'], s['n_tolerance'], s['tolerance_ratio'])
        self.records, self.base = [], None
        self.streak, self.descent_start = 0, None

    def _accepted(self, finalize=False):
        avg = self.detector.finalize() if finalize else self.detector.final_average
        return [] if avg is None else [v['epoch'] for v in avg.segment_weights]

    def _append_base(self, epoch):
        segment = self.bank[epoch]
        if self.base is None:
            self.base = WeightedStateAverage(segment)
        else:
            self.base.update(segment)

    def _commit_old(self, epoch):
        for number in self._accepted():
            last = self.base.segment_weights[-1]['epoch'] if self.base else -1
            if last < number <= epoch - self.delay:
                self._append_base(number)

    def _seal(self, reason, cutoff=None):
        epochs = [e for e in self._accepted(finalize=True) if cutoff is None or e <= cutoff]
        event = dict(event='seal', valley_id=f'valley_{self.valley_index:02d}',
                     reason=reason, cutoff_epoch=cutoff, included_epochs=epochs)
        if not epochs:
            event.update(eligible=False, discarded='empty_prefix')
            self.events.append(event)
            return
        if self.base and self.base.segment_weights[-1]['epoch'] > epochs[-1]:
            raise RuntimeError('Rollback crossed committed tensor prefix')
        average = self.base
        for epoch in epochs:
            if average and epoch <= average.segment_weights[-1]['epoch']:
                continue
            if average is None:
                average = WeightedStateAverage(self.bank[epoch])
            else:
                average.update(self.bank[epoch])
        if [v['epoch'] for v in average.segment_weights] != epochs:
            raise RuntimeError('Candidate ledger/tensor mismatch')
        by_epoch = {s.epoch: s for s in self.records}
        if sum(by_epoch[e].n_averaged for e in epochs) != average.n_averaged:
            raise RuntimeError('Incorrect dense-step weight')
        candidate = ValleyCandidate(self.settings['id'], event['valley_id'], dict(self.settings),
            average.state, reason, sum(by_epoch[e].end_loss for e in epochs) / len(epochs),
            average.start_step, average.end_step, average.n_averaged, tuple(epochs),
            tuple(dict(v) for v in average.segment_weights))
        event['eligible'] = len(epochs) >= self.settings['minimum_valley_segments']
        if event['eligible']:
            self.pending.append(candidate)
        self.events.append(event)

    def _process(self, segment, allow_descent=True):
        scalar = replace(segment, state={})
        self.records.append(scalar)
        self.detector.observe(scalar)
        if self.variant == VARIANTS[0] and len(self.detector.converge_queue) >= self.settings['n_converge']:
            self.events.append(dict(event='task_stability', epoch=segment.epoch,
                                    **stability(list(self.detector.converge_queue))))
        # Preserve the original deterioration exit's priority.
        if self.detector.dead_valley:
            self._seal('dead_valley')
            self.valley_index += 1
            self._reset()
            return
        if allow_descent:
            window = self.settings['n_tolerance']
            accepted = self._accepted()
            recent = self.records[-window:]
            ready = (accepted and len(recent) == window and recent[0].epoch >= accepted[0])
            if ready:
                slope, sigma, _ = line([s.end_loss for s in recent])
                drop = -(window - 1) * slope
                qualifies = slope < 0 and drop > max(sigma, EPSILON)
                if qualifies:
                    if self.streak == 0:
                        self.descent_start = recent[0].epoch
                    self.streak += 1
                else:
                    self.streak, self.descent_start = 0, None
                self.events.append(dict(event='descent_window', epoch=segment.epoch,
                    slope=slope, residual_sigma=sigma, fitted_drop=drop,
                    qualifies=qualifies, streak=self.streak, first_window_start=self.descent_start))
                if self.streak >= self.settings['descent_confirmations']:
                    start = self.descent_start
                    replay = [self.bank[s.epoch] for s in self.records if s.epoch >= start]
                    self._seal('confirmed_descent', cutoff=start - 1)
                    self.valley_index += 1
                    self._reset()
                    # Replay entry/ordinary exit exactly once. Do not recursively
                    # trigger the same descent on the same observation window.
                    for old in replay:
                        self._process(old, allow_descent=False)
                    return
        self._commit_old(segment.epoch)

    def observe(self, segment):
        if self.finalized or segment.epoch <= self.last_epoch:
            raise ValueError('Finalized or unordered controller')
        self.last_epoch = segment.epoch
        if segment.epoch > self.settings['warmup_epochs']:
            self._process(segment)

    def finalize(self):
        if not self.finalized:
            if self.records:
                self._seal('training_end')
            self.finalized = True
            self.base, self.records = None, []
        return list(self.pending)


class AveragingGrid:
    def __init__(self, model, settings, variants=VARIANTS):
        self.variants = tuple(variants)
        if not self.variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
            raise ValueError('Invalid variant subset')
        if not settings or len({s['id'] for s in settings}) != len(settings):
            raise ValueError('Unique nonempty grid required')
        expanded = []
        for variant in variants:
            for source in settings:
                s = dict(source)
                s['maximum_candidates'] = 4
                s['n_converge'] = 3
                s['descent_confirmations'] = 3
                if min(s['n_converge'], s['n_tolerance']) < 3:
                    raise ValueError('Exploration trend windows must be at least three')
                expanded.append((variant, s))
        self.delay = max(max(s['n_converge'], s['n_tolerance'],
                             s['n_tolerance'] + s['descent_confirmations'] - 1) for v, s in expanded)
        self.bank = {}
        self.controllers = [DeferredController(s, v, self.bank, self.delay) for v, s in expanded]
        self.sampler = DenseEpochSampler(model) if model is not None else None
        self.last_epoch, self.last_step = 0, 0

    def begin_epoch(self): self.sampler.begin_epoch()
    def on_optimizer_step(self, model): self.sampler.update(model)
    @property
    def segment_model(self): return self.sampler.module

    def observe_epoch(self, epoch, detection_loss, metrics, normalized_task_losses):
        extra = dict(metrics)
        extra.update({f'normalized_task_loss_{i}': float(v) for i, v in enumerate(normalized_task_losses)})
        self.observe(self.sampler.freeze(epoch=epoch, detection_loss=detection_loss, metrics=extra))

    def observe(self, segment):
        WeightedStateAverage._validate_segment(segment)
        if segment.epoch != self.last_epoch + 1 or segment.start_step != self.last_step + 1:
            raise ValueError('Exploration requires consecutive epoch/step segments')
        if not math.isfinite(segment.end_loss): raise ValueError('Nonfinite loss')
        values = [segment.metrics[f'normalized_task_loss_{i}'] for i in range(4)]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Invalid normalized task losses')
        self.bank[segment.epoch] = segment
        for controller in self.controllers:
            controller.observe(segment)
        self.last_epoch, self.last_step = segment.epoch, segment.end_step
        # Evict only after every controller has committed its accepted prefix.
        for epoch in list(self.bank):
            if epoch <= segment.epoch - self.delay:
                del self.bank[epoch]

    def finalize(self):
        for c in self.controllers: c.finalize()

    def drain(self, consume):
        # Score every eligible sealed valley before capacity pruning.
        for c in self.controllers:
            while c.pending:
                consume(c.variant, c.pending[0])
                c.pending.pop(0)

    def diagnostics(self):
        return {v: {c.settings['id']: c.events for c in self.controllers if c.variant == v}
                for v in self.variants}

    def state_dict(self):
        # Pickle preserves shared bank references, storing each snapshot once.
        return dict(revision=REVISION, variants=self.variants, controllers=self.controllers,
                    bank=self.bank, last_epoch=self.last_epoch, last_step=self.last_step,
                    global_step=self.sampler.global_step if self.sampler else self.last_step)

    def load_state_dict(self, state):
        if state['revision'] != REVISION or tuple(state['variants']) != self.variants:
            raise ValueError('Changed weight refinement configuration/revision')
        expected = [(c.variant, c.settings) for c in self.controllers]
        if [(c.variant, c.settings) for c in state['controllers']] != expected:
            raise ValueError('Changed experiment grid')
        self.controllers, self.bank = state['controllers'], state['bank']
        for controller in self.controllers:
            controller.bank = self.bank
        self.last_epoch, self.last_step = state['last_epoch'], state['last_step']
        if state['global_step'] != self.last_step: raise ValueError('Invalid sampler step')
        if self.sampler: self.sampler.global_step = state['global_step']
