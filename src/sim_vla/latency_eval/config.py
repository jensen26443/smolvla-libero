"""Experiment configuration dataclasses and YAML loading.

Configurations are explicit and small; the frozen formal matrix is a config
plus recorded hashes, written verbatim into the experiment directory as
``experiment_spec.yaml``. Latency levels chosen during the pilot are frozen
here *before* the formal runs and never re-tuned on test results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

METHOD_KINDS = ("r0_every_step", "b1_fixed_period", "b2_fixed_period_conservative", "m_feedback")


@dataclass
class JitterConfig:
    """One pre-registered inference-delay jitter distribution.

    The multiplier trajectory is generated once from ``seed`` with length
    ``length_ticks`` (>= max_ticks) and saved next to the experiment spec; all
    methods read the same tick-indexed curve.
    """

    distribution: str = "uniform"  # only "uniform" is implemented
    low: float = 0.8
    high: float = 1.2
    seed: int = 1235
    length_ticks: int = 512


@dataclass
class LatencyCondition:
    name: str
    obs_delay_ticks: int
    inference_delay_ticks: int
    jitter: JitterConfig | None = None


@dataclass
class FeedbackConfig:
    diff_threshold_high: float
    diff_threshold_low: float | None = None
    min_interval_ticks: int = 2
    max_interval_ticks: int = 25
    image_key: str = "image"
    downsample_factor: int = 4


@dataclass
class MethodConfig:
    name: str
    kind: str
    period_ticks: int | None = None  # K for b1/b2 (request schedule)
    segment_ticks: int | None = None  # S for b2 (switch boundary)
    feedback: FeedbackConfig | None = None  # for m_feedback
    # Restrict this method to a subset of latency conditions / tasks (e.g. the
    # R0 reference only runs at near-zero latency). None means all.
    latency_names: list[str] | None = None
    task_keys: list[str] | None = None


@dataclass
class InitRange:
    start_index: int
    count: int
    start_seed: int

    def pairs(self) -> list[tuple[int, int]]:
        return [
            (self.start_index + i, self.start_seed + i) for i in range(self.count)
        ]


@dataclass
class VideoSampleConfig:
    record: bool = True
    trials_per_condition: int = 1  # record the first N inits of each condition
    fps: int = 20


@dataclass
class ExperimentConfig:
    experiment_id: str
    checkpoint: str
    tasks: list[str]
    methods: list[MethodConfig]
    latency_conditions: list[LatencyCondition]
    inits: InitRange
    max_ticks: int = 300
    device: str = "cuda"
    max_attempts: int = 3
    env_rebuild_every: int = 50
    video: VideoSampleConfig = field(default_factory=VideoSampleConfig)
    notes: str = ""

    def validate(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must be set")
        if not (Path(self.checkpoint).is_dir() if ":" not in self.checkpoint else True):
            raise FileNotFoundError(f"checkpoint dir not found: {self.checkpoint}")
        seen_methods = set()
        for method in self.methods:
            if method.kind not in METHOD_KINDS:
                raise ValueError(f"unknown method kind {method.kind}")
            if method.name in seen_methods:
                raise ValueError(f"duplicate method name {method.name}")
            seen_methods.add(method.name)
            if method.kind in ("b1_fixed_period", "b2_fixed_period_conservative") and not method.period_ticks:
                raise ValueError(f"{method.name}: period_ticks (K) is required")
            if method.kind == "b2_fixed_period_conservative" and not method.segment_ticks:
                raise ValueError(f"{method.name}: segment_ticks (S) is required")
            if method.kind == "m_feedback" and method.feedback is None:
                raise ValueError(f"{method.name}: feedback config is required")
        names = set()
        for condition in self.latency_conditions:
            if condition.name in names:
                raise ValueError(f"duplicate latency condition name {condition.name}")
            names.add(condition.name)
            if condition.jitter is not None and condition.inference_delay_ticks <= 0:
                raise ValueError(f"{condition.name}: jitter needs a positive base delay")
        if self.inits.count <= 0:
            raise ValueError("init range must select at least one initialization")


def _build_feedback(data: dict[str, Any]) -> FeedbackConfig:
    return FeedbackConfig(
        diff_threshold_high=float(data["diff_threshold_high"]),
        diff_threshold_low=(
            float(data["diff_threshold_low"]) if data.get("diff_threshold_low") is not None else None
        ),
        min_interval_ticks=int(data.get("min_interval_ticks", 2)),
        max_interval_ticks=int(data.get("max_interval_ticks", 25)),
        image_key=str(data.get("image_key", "image")),
        downsample_factor=int(data.get("downsample_factor", 4)),
    )


def _build_method(data: dict[str, Any]) -> MethodConfig:
    feedback = data.get("feedback")
    return MethodConfig(
        name=str(data["name"]),
        kind=str(data["kind"]),
        period_ticks=int(data["period_ticks"]) if data.get("period_ticks") is not None else None,
        segment_ticks=int(data["segment_ticks"]) if data.get("segment_ticks") is not None else None,
        feedback=_build_feedback(feedback) if feedback else None,
        latency_names=[str(v) for v in data["latency_names"]] if data.get("latency_names") else None,
        task_keys=[str(v) for v in data["task_keys"]] if data.get("task_keys") else None,
    )


def _build_condition(data: dict[str, Any]) -> LatencyCondition:
    jitter = data.get("jitter")
    return LatencyCondition(
        name=str(data["name"]),
        obs_delay_ticks=int(data["obs_delay_ticks"]),
        inference_delay_ticks=int(data["inference_delay_ticks"]),
        jitter=JitterConfig(**jitter) if jitter else None,
    )


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    inits = data["inits"]
    video = data.get("video", {})
    config = ExperimentConfig(
        experiment_id=str(data["experiment_id"]),
        checkpoint=str(data["checkpoint"]),
        tasks=[str(t) for t in data["tasks"]],
        methods=[_build_method(m) for m in data["methods"]],
        latency_conditions=[_build_condition(c) for c in data["latency_conditions"]],
        inits=InitRange(
            start_index=int(inits["start_index"]),
            count=int(inits["count"]),
            start_seed=int(inits["start_seed"]),
        ),
        max_ticks=int(data.get("max_ticks", 300)),
        device=str(data.get("device", "cuda")),
        max_attempts=int(data.get("max_attempts", 3)),
        env_rebuild_every=int(data.get("env_rebuild_every", 50)),
        video=VideoSampleConfig(
            record=bool(video.get("record", True)),
            trials_per_condition=int(video.get("trials_per_condition", 1)),
            fps=int(video.get("fps", 20)),
        ),
        notes=str(data.get("notes", "")),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"config root must be a mapping: {source}")
    return config_from_dict(data)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    data = asdict(config)
    return data


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(config_to_dict(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_frozen_spec(config: ExperimentConfig, path: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
    """Persist the frozen experiment spec (config + hashes) atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config_to_dict(config),
        "config_hash": config_hash(config),
        "extra": extra or {},
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    tmp.replace(target)
    return target
