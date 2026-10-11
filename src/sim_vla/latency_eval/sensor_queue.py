"""Observation arrival queue implementing fixed sensor latency."""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import ObservationPacket


def _deep_copy_observation(
    raw_obs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    """Snapshot an environment observation into immutable-by-convention copies."""

    images: dict[str, np.ndarray] = {}
    pixels = raw_obs.get("pixels", raw_obs)
    if isinstance(pixels, dict):
        for name, image in pixels.items():
            images[str(name)] = np.array(image, copy=True)
    robot_state = raw_obs.get("robot_state")
    if robot_state is not None:
        robot_state = {
            group: {
                key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
                for key, value in fields.items()
            }
            for group, fields in robot_state.items()
            if isinstance(fields, dict)
        }
    return images, robot_state


class SensorQueue:
    """Captures observations at each tick and releases them after the sensor delay.

    A packet captured at tick ``t`` with delay ``d`` is releasable exactly at
    tick ``t + d`` and only once. Composite packets (image + state) share one
    arrival time by construction; if modalities ever split, they must be given
    explicit separate queues rather than silently reusing this class.
    """

    def __init__(self, delay_ticks: int, instruction: str) -> None:
        if delay_ticks < 0:
            raise ValueError(f"sensor delay must be >= 0 ticks, got {delay_ticks}")
        self.delay_ticks = int(delay_ticks)
        self.instruction = instruction
        self._next_packet_id = 0
        self._pending: list[ObservationPacket] = []
        self._released_ids: set[int] = set()

    def capture(self, tick: int, raw_obs: dict[str, Any]) -> ObservationPacket:
        images, robot_state = _deep_copy_observation(raw_obs)
        packet = ObservationPacket(
            packet_id=self._next_packet_id,
            capture_tick=tick,
            arrive_tick=tick + self.delay_ticks,
            images=images,
            robot_state=robot_state,
            instruction=self.instruction,
            modality_capture_ticks={
                "images": tick,
                "state": tick,
            },
        )
        self._next_packet_id += 1
        self._pending.append(packet)
        return packet

    def release_due(self, tick: int) -> list[ObservationPacket]:
        due = [packet for packet in self._pending if packet.arrive_tick <= tick]
        if due:
            self._pending = [packet for packet in self._pending if packet.arrive_tick > tick]
            for packet in due:
                if packet.packet_id in self._released_ids:
                    raise RuntimeError(f"packet {packet.packet_id} released twice")
                self._released_ids.add(packet.packet_id)
        return due

    @property
    def pending_count(self) -> int:
        return len(self._pending)
