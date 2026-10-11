"""Request triggers: fixed-period (B1/B2), every-step (R0), and feedback (M).

All triggers see only :class:`~sim_vla.latency_eval.contracts.PolicyView`
(arrived observations, fixed instruction, sent commands). Evaluator ground
truth is structurally invisible to them. Trigger state mutates only when a
request is actually submitted (``on_request_submitted``); a suppressed
evaluation leaves the state untouched so busy suppression cannot change
future behavior.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .contracts import ObservationPacket, PolicyView, TriggerDecision


class Trigger(Protocol):
    def evaluate(self, tick: int, view: PolicyView) -> TriggerDecision: ...

    def on_request_submitted(self, tick: int, packet: ObservationPacket) -> None: ...

    def reset(self) -> None: ...


class EveryStepTrigger:
    """R0 reference: request at every tick (official replan-every-step cadence)."""

    def evaluate(self, tick: int, view: PolicyView) -> TriggerDecision:
        if view.latest_packet is None:
            return TriggerDecision(fire=False, reason="no_observation")
        return TriggerDecision(fire=True, reason="every_step")

    def on_request_submitted(self, tick: int, packet: ObservationPacket) -> None:
        return

    def reset(self) -> None:
        return


class FixedPeriodTrigger:
    """B1/B2: fire every K ticks, measured on request time, no backlog.

    ``K`` is the request-schedule period; it is independent of the model chunk
    horizon H. While a request is in flight the runner suppresses evaluation,
    so missed periods are simply skipped (no accumulation).
    """

    def __init__(self, period_ticks: int) -> None:
        if period_ticks < 1:
            raise ValueError(f"period must be >= 1 ticks, got {period_ticks}")
        self.period_ticks = int(period_ticks)
        self._last_request_tick: int | None = None

    def evaluate(self, tick: int, view: PolicyView) -> TriggerDecision:
        if view.latest_packet is None:
            return TriggerDecision(fire=False, reason="no_observation")
        if self._last_request_tick is None or tick - self._last_request_tick >= self.period_ticks:
            return TriggerDecision(fire=True, reason="period")
        return TriggerDecision(fire=False, reason="cooldown", detail={"remaining": float(self.period_ticks - (tick - self._last_request_tick))})

    def on_request_submitted(self, tick: int, packet: ObservationPacket) -> None:
        self._last_request_tick = tick

    def reset(self) -> None:
        self._last_request_tick = None


def to_lowres_gray(image: np.ndarray, factor: int = 4) -> np.ndarray:
    """Deterministic low-resolution grayscale of an RGB uint8 image.

    Block-mean downsampling by an integer factor; no external cv2 dependency so
    the transformation is identical on every machine.
    """

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"expected (H, W, >=3) image, got {array.shape}")
    height, width = array.shape[0] // factor * factor, array.shape[1] // factor * factor
    blocks = array[:height, :width, :3].astype(np.float64).reshape(
        height // factor, factor, width // factor, factor, 3
    )
    return blocks.mean(axis=(1, 3, 4))


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of two same-shape grayscale frames."""

    if a.shape != b.shape:
        raise ValueError(f"frame shapes differ: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).mean())


class FeedbackTrigger:
    """M: image-difference trigger with cooldown, max interval, and hysteresis.

    Fire rule (all must hold; the in-flight check is the runner's job): a
    packet has arrived that the trigger has not evaluated yet, the cooldown
    since the last request has elapsed, and either the low-resolution gray
    difference against the frame underlying the last request exceeds
    ``diff_threshold_high`` (with the trigger armed, i.e. the difference has
    previously fallen below ``diff_threshold_low``), or the max interval since
    the last request has elapsed. Long observation outages therefore cannot
    re-trigger on a stale frame: without a *new* packet nothing fires, and the
    executor's exhaustion rules take over.
    """

    def __init__(
        self,
        *,
        diff_threshold_high: float,
        diff_threshold_low: float | None = None,
        min_interval_ticks: int,
        max_interval_ticks: int,
        image_key: str = "image",
        downsample_factor: int = 4,
    ) -> None:
        if diff_threshold_high <= 0:
            raise ValueError("diff_threshold_high must be positive")
        low = diff_threshold_low if diff_threshold_low is not None else diff_threshold_high / 2.0
        if not 0 <= low <= diff_threshold_high:
            raise ValueError("diff_threshold_low must lie in [0, diff_threshold_high]")
        if min_interval_ticks < 1 or max_interval_ticks < min_interval_ticks:
            raise ValueError("require 1 <= min_interval_ticks <= max_interval_ticks")
        self.diff_threshold_high = float(diff_threshold_high)
        self.diff_threshold_low = float(low)
        self.min_interval_ticks = int(min_interval_ticks)
        self.max_interval_ticks = int(max_interval_ticks)
        self.image_key = image_key
        self.downsample_factor = int(downsample_factor)
        self._last_request_tick: int | None = None
        self._last_request_frame: np.ndarray | None = None
        self._last_seen_packet_id: int | None = None
        self._armed = True
        self.last_diff: float | None = None

    def _frame_of(self, packet: ObservationPacket) -> np.ndarray | None:
        image = packet.images.get(self.image_key)
        if image is None:
            return None
        return to_lowres_gray(image, self.downsample_factor)

    def evaluate(self, tick: int, view: PolicyView) -> TriggerDecision:
        packet = view.latest_packet
        if packet is None:
            return TriggerDecision(fire=False, reason="no_observation")
        if packet.packet_id == self._last_seen_packet_id:
            return TriggerDecision(fire=False, reason="no_new_observation")
        # Each packet is examined at most once; cooldown-blocked packets are
        # consumed too (a stale frame must never re-arm a later request).
        self._last_seen_packet_id = packet.packet_id
        frame = self._frame_of(packet)
        if frame is None:
            return TriggerDecision(fire=False, reason=f"missing_image_{self.image_key}")
        since_last = None if self._last_request_tick is None else tick - self._last_request_tick
        detail: dict[str, float] = {}
        if since_last is not None and since_last < self.min_interval_ticks:
            detail["remaining"] = float(self.min_interval_ticks - since_last)
            return TriggerDecision(fire=False, reason="cooldown", detail=detail)
        if self._last_request_frame is None:
            return TriggerDecision(fire=True, reason="first_request", detail=detail)
        if frame.shape == self._last_request_frame.shape:
            diff = frame_difference(frame, self._last_request_frame)
            self.last_diff = diff
            detail["image_diff"] = diff
            if diff < self.diff_threshold_low:
                self._armed = True
        max_due = since_last is not None and since_last >= self.max_interval_ticks
        if max_due:
            return TriggerDecision(fire=True, reason="max_interval", detail=detail)
        if self._armed and self.last_diff is not None and self.last_diff >= self.diff_threshold_high:
            return TriggerDecision(fire=True, reason="image_diff", detail=detail)
        return TriggerDecision(fire=False, reason="below_threshold", detail=detail)

    def on_request_submitted(self, tick: int, packet: ObservationPacket) -> None:
        self._last_request_tick = tick
        frame = self._frame_of(packet)
        if frame is not None:
            self._last_request_frame = frame
        self._armed = False
        self._last_seen_packet_id = packet.packet_id

    def reset(self) -> None:
        self._last_request_tick = None
        self._last_request_frame = None
        self._last_seen_packet_id = None
        self._armed = True
        self.last_diff = None
