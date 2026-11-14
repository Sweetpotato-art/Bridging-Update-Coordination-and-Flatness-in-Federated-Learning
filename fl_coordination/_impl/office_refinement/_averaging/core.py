"""Training-time weight averaging over shared and client model states.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch
from torch import nn


refinement_UPDATE_IMPLEMENTATIONS = ("reference",)
refinement_AVERAGING_REVISION = "unique_segments_v2"


def _validate_update_implementation(value: str) -> str:
    implementation = str(value)
    if implementation not in refinement_UPDATE_IMPLEMENTATIONS:
        raise ValueError(
            "Unknown refinement update implementation "
            f"{implementation!r}; expected one of {refinement_UPDATE_IMPLEMENTATIONS}"
        )
    return implementation


def _copy_state(module: nn.Module, device: str | torch.device = "cpu"):
    return {
        name: value.detach().to(device=device, copy=True)
        for name, value in module.state_dict().items()
    }


def _merge_state(
    target: dict[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    target_weight: int,
    source_weight: int,
) -> None:
    total = int(target_weight) + int(source_weight)
    if total <= 0:
        raise ValueError("Averaging weights must sum to a positive integer")
    fraction = float(source_weight) / float(total)
    with torch.no_grad():
        for name, target_tensor in target.items():
            source_tensor = source[name].to(target_tensor.device)
            if target_tensor.is_floating_point() or target_tensor.is_complex():
                target_tensor.lerp_(source_tensor.to(target_tensor.dtype), fraction)
            else:
                target_tensor.copy_(source_tensor)


class DenseEpochSampler:
    """Average the model state after each update within an epoch."""
    def __init__(self, model, *, update_implementation="reference"):
        if update_implementation != "reference": raise ValueError("Use reference refinement")
        self.module = copy.deepcopy(model).requires_grad_(False).eval()
        self.n_averaged = self.global_step = 0
        self.start_step = 1

    def begin_epoch(self):
        self.n_averaged = 0
        self.start_step = self.global_step + 1

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, target in self.module.state_dict().items():
            current = source[name].detach().to(target.device)
            if self.n_averaged == 0 or not (target.is_floating_point() or target.is_complex()):
                target.copy_(current)
            else:
                target.lerp_(current.to(target.dtype), 1.0 / (self.n_averaged + 1))
        self.n_averaged += 1
        self.global_step += 1

    def freeze(self, *, epoch, detection_loss, metrics):
        if self.n_averaged <= 0: raise RuntimeError("Empty refinement segment")
        return FrozenSegment(epoch=int(epoch),state=_copy_state(self.module),
            n_averaged=self.n_averaged,start_step=self.start_step,end_step=self.global_step,
            end_loss=float(detection_loss),metrics={str(k):float(v) for k,v in metrics.items()})


@dataclass(frozen=True)
class FrozenSegment:
    epoch: int
    state: dict[str, torch.Tensor]
    n_averaged: int
    start_step: int
    end_step: int
    end_loss: float
    metrics: dict[str, float]


class WeightedStateAverage:
    def __init__(self, segment: FrozenSegment) -> None:
        self._validate_segment(segment)
        self.state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in segment.state.items()
        }
        self.n_averaged = int(segment.n_averaged)
        self.start_step = int(segment.start_step)
        self.end_step = int(segment.end_step)
        self.segment_weights = [self._segment_weight(segment)]

    @staticmethod
    def _validate_segment(segment: FrozenSegment) -> None:
        if (
            segment.n_averaged <= 0
            or segment.start_step < 1
            or segment.end_step - segment.start_step + 1 != segment.n_averaged
        ):
            raise ValueError("Dense refinement segment count must match its step interval")

    @staticmethod
    def _segment_weight(segment: FrozenSegment) -> dict[str, int]:
        return {
            "epoch": int(segment.epoch),
            "start_step": int(segment.start_step),
            "end_step": int(segment.end_step),
            "n_averaged": int(segment.n_averaged),
        }

    def update(self, segment: FrozenSegment) -> None:
        self._validate_segment(segment)
        if (
            segment.start_step <= self.end_step
            or segment.epoch <= self.segment_weights[-1]["epoch"]
        ):
            raise ValueError("Duplicate, overlapping, or out-of-order refinement segment")
        _merge_state(
            self.state,
            segment.state,
            self.n_averaged,
            segment.n_averaged,
        )
        self.n_averaged += int(segment.n_averaged)
        self.end_step = int(segment.end_step)
        self.segment_weights.append(self._segment_weight(segment))


class LossValleyDetector:
    """Track an averaging interval and count each eligible epoch segment once.
    """

    def __init__(self, n_converge: int, n_tolerance: int, tolerance_ratio: float):
        if int(n_converge) <= 0 or int(n_tolerance) <= 0:
            raise ValueError("refinement windows must be positive")
        if float(tolerance_ratio) < 0.0:
            raise ValueError("refinement tolerance_ratio must be non-negative")
        self.n_converge = int(n_converge)
        self.n_tolerance = int(n_tolerance)
        self.tolerance_ratio = float(tolerance_ratio)
        self.converge_queue: deque[FrozenSegment] = deque(maxlen=self.n_converge)
        self.smooth_queue: deque[FrozenSegment] = deque(maxlen=self.n_tolerance)
        self.final_average: WeightedStateAverage | None = None
        self.converge_step: int | None = None
        self.threshold: float | None = None
        self.dead_valley = False
        self.finalized = False
        self._last_observed_step = 0
        self._last_observed_epoch = -1

    def _smooth_loss(self, index: int) -> float:
        return min(segment.end_loss for segment in list(self.smooth_queue)[index:])

    def observe(self, segment: FrozenSegment) -> None:
        if self.finalized:
            raise RuntimeError("Cannot observe a finalized refinement detector")
        if self.dead_valley:
            return
        WeightedStateAverage._validate_segment(segment)
        if (
            segment.start_step <= self._last_observed_step
            or segment.epoch <= self._last_observed_epoch
        ):
            raise ValueError("Duplicate or out-of-order refinement observation")
        if not torch.isfinite(torch.tensor(segment.end_loss)):
            raise ValueError("refinement detection loss must be finite")
        self._last_observed_step = segment.end_step
        self._last_observed_epoch = segment.epoch
        self.converge_queue.append(segment)
        self.smooth_queue.append(segment)

        if self.converge_step is None:
            if len(self.converge_queue) < self.n_converge:
                return
            losses = [candidate.end_loss for candidate in self.converge_queue]
            minimum_index = min(range(len(losses)), key=losses.__getitem__)
            if minimum_index != 0:
                return
            first = self.converge_queue[0]
            self.converge_step = first.end_step
            self.final_average = WeightedStateAverage(first)
            self.threshold = (sum(losses) / len(losses)) * (1.0 + self.tolerance_ratio)
            if self.n_tolerance < self.n_converge:
                for index in range(self.n_converge - self.n_tolerance):
                    self.final_average.update(self.converge_queue[1 + index])
            return

        if self.smooth_queue[0].end_step < self.converge_step:
            return
        assert self.threshold is not None and self.final_average is not None
        if self._smooth_loss(0) > self.threshold:
            self.dead_valley = True
            return
        candidate = self.smooth_queue[0]
        if candidate.start_step > self.final_average.end_step:
            self.final_average.update(candidate)

    def finalize(self) -> WeightedStateAverage:
        if not self.finalized:
            if self.converge_step is None:
                if not self.converge_queue:
                    raise RuntimeError("refinement detector received no segments")
                self.final_average = WeightedStateAverage(self.converge_queue[-1])
            elif not self.dead_valley:
                assert self.threshold is not None and self.final_average is not None
                while self.smooth_queue:
                    if self.smooth_queue[0].end_step <= self.final_average.end_step:
                        self.smooth_queue.popleft()
                        continue
                    if self._smooth_loss(0) > self.threshold:
                        break
                    self.final_average.update(self.smooth_queue.popleft())
            self.finalized = True
        assert self.final_average is not None
        return self.final_average


@dataclass
class ValleyCandidate:
    grid_id: str
    valley_id: str
    settings: dict[str, Any]
    state: dict[str, torch.Tensor]
    close_reason: str
    detection_loss: float
    start_step: int
    end_step: int
    averaged_steps: int
    included_epochs: tuple[int, ...]
    segment_weights: tuple[dict[str, int], ...]

    @property
    def segment_count(self) -> int:
        return len(self.included_epochs)

    def metadata(self) -> dict[str, Any]:
        return {
            "averaging_revision": refinement_AVERAGING_REVISION,
            "weighting": "one_vote_per_successful_optimizer_step",
            "segment_weights": [dict(item) for item in self.segment_weights],
            "grid_id": self.grid_id,
            "valley_id": self.valley_id,
            "settings": dict(self.settings),
            "close_reason": self.close_reason,
            "detection_loss": self.detection_loss,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "averaged_steps": self.averaged_steps,
            "included_epochs": list(self.included_epochs),
            "segment_count": self.segment_count,
        }


class MultiValleyController:
    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = {
            "id": str(settings["id"]),
            "warmup_epochs": int(settings["warmup_epochs"]),
            "n_converge": int(settings["n_converge"]),
            "n_tolerance": int(settings["n_tolerance"]),
            "tolerance_ratio": float(settings["tolerance_ratio"]),
            "minimum_valley_segments": int(settings["minimum_valley_segments"]),
            "maximum_candidates": int(settings.get("maximum_candidates", 6)),
        }
        if self.settings["warmup_epochs"] < 0:
            raise ValueError("refinement warmup_epochs cannot be negative")
        if self.settings["minimum_valley_segments"] <= 0:
            raise ValueError("minimum_valley_segments must be positive")
        self.detector = self._new_detector()
        self.records: list[FrozenSegment] = []
        self.candidates: list[ValleyCandidate] = []
        self.valley_index = 1

    def _new_detector(self) -> LossValleyDetector:
        return LossValleyDetector(
            self.settings["n_converge"],
            self.settings["n_tolerance"],
            self.settings["tolerance_ratio"],
        )

    def observe(self, segment: FrozenSegment) -> None:
        if segment.epoch <= self.settings["warmup_epochs"]:
            return
        # This ledger is used only for metadata/weight audits in _seal. Tensor
        # snapshots remain in the detector's bounded queues and averages.
        # Replace rather than mutate: all parameter candidates share segments.
        self.records.append(replace(segment, state={}))
        self.detector.observe(segment)
        if self.detector.dead_valley:
            self._seal("dead_valley")
            self.valley_index += 1
            self.detector = self._new_detector()
            self.records = []

    def _seal(self, reason: str) -> None:
        if not self.records:
            return
        average = self.detector.finalize()
        by_epoch = {segment.epoch: segment for segment in self.records}
        included = [by_epoch[item["epoch"]] for item in average.segment_weights]
        if sum(segment.n_averaged for segment in included) != average.n_averaged:
            raise RuntimeError("refinement weight audit failed: unique segment counts differ")
        candidate = ValleyCandidate(
            grid_id=self.settings["id"],
            valley_id=f"valley_{self.valley_index:02d}",
            settings=dict(self.settings),
            state={name: value.clone() for name, value in average.state.items()},
            close_reason=reason,
            detection_loss=float(
                sum(segment.end_loss for segment in included) / len(included)
            ),
            start_step=int(average.start_step),
            end_step=int(average.end_step),
            averaged_steps=int(average.n_averaged),
            included_epochs=tuple(segment.epoch for segment in included),
            segment_weights=tuple(dict(item) for item in average.segment_weights),
        )
        self.candidates.append(candidate)
        maximum = self.settings["maximum_candidates"]
        if len(self.candidates) > maximum:
            worst = max(
                range(len(self.candidates)),
                key=lambda index: (
                    self.candidates[index].detection_loss,
                    self.candidates[index].valley_id,
                ),
            )
            self.candidates.pop(worst)

    def finalize(self) -> list[ValleyCandidate]:
        if self.records:
            self._seal("training_end")
            self.records = []
        return list(self.candidates)


class refinementParameterGrid:
    def __init__(
        self,
        model: nn.Module,
        settings: Sequence[Mapping[str, Any]],
        *,
        update_implementation: str = "reference",
    ) -> None:
        if not settings:
            raise ValueError("At least one refinement parameter candidate is required")
        ids = [str(item["id"]) for item in settings]
        if len(ids) != len(set(ids)):
            raise ValueError("refinement grid ids must be unique")
        self.sampler = DenseEpochSampler(
            model,
            update_implementation=update_implementation,
        )
        self.controllers = [MultiValleyController(item) for item in settings]

    def begin_epoch(self) -> None:
        self.sampler.begin_epoch()

    def on_optimizer_step(self, model: nn.Module) -> None:
        self.sampler.update(model)

    @property
    def segment_model(self) -> nn.Module:
        if self.sampler.n_averaged <= 0:
            raise RuntimeError("No optimizer update has been sampled this epoch")
        return self.sampler.module

    def observe_epoch(
        self,
        epoch: int,
        detection_loss: float,
        metrics: Mapping[str, float],
    ) -> None:
        segment = self.sampler.freeze(
            epoch=epoch,
            detection_loss=detection_loss,
            metrics=metrics,
        )
        for controller in self.controllers:
            controller.observe(segment)

    def finalize(self) -> list[ValleyCandidate]:
        result: list[ValleyCandidate] = []
        for controller in self.controllers:
            result.extend(controller.finalize())
        return result


def retention(metrics, references):
    import math
    values = [metrics[k] for k in ("miou", "depth_abs", "normal_mean")]
    if any(not math.isfinite(v) for v in values+list(references)) or min(references)<=0:
        raise ValueError("Invalid validation metrics or STL references")
    signed = [(references[0]-values[0])/references[0],
              (values[1]-references[1])/references[1],
              (values[2]-references[2])/references[2]]
    positive = [max(0.,v) for v in signed]
    return dict(signed_relative_drops=signed,positive_relative_drops=positive,
                worst_task_relative_drop=max(positive),mean_positive_drop=sum(positive)/3)


def rank_candidates(evaluations, references):
    scored=[]
    for source in evaluations:
        item=copy.deepcopy(source)
        if item['metadata']['segment_count'] < item['metadata']['settings']['minimum_valley_segments']:
            continue
        item['st_retention']=retention(item['validation'],references)
        scored.append(item)
    if not scored: raise RuntimeError('No eligible refinement valley; no test evaluation or silently substituted endpoint')
    scored.sort(key=lambda x:(x['st_retention']['worst_task_relative_drop'],
        x['st_retention']['mean_positive_drop'],x['validation_score'],
        -x['metadata']['segment_count'],x['metadata']['detection_loss'],
        x['metadata']['grid_id'],x['metadata']['valley_id']))
    return dict(selection_rule='worst_positive_relative_STL_drop_then_mean_then_normalized_validation_loss',
        references=list(references),selected=scored[0],valid_candidates=scored)
