"""Pinned Hub assets used by the LIBERO/SmolVLA smoke and runbook.

The LeRobot 0.5.1 command line does not expose a ``--policy.revision`` field.
The reproducible workaround is to materialize the policy at its pinned commit
and pass that local directory to the unchanged official ``lerobot-*`` entry
point.  This module keeps that preparation explicit; importing it never
downloads anything.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .protocol import (
    LIBERO_ASSETS_REPO,
    LIBERO_ASSETS_REVISION,
    MODEL_REPO,
    MODEL_REVISION,
)


def project_root() -> Path:
    """Return the checked-out ``sim_vla`` directory."""

    return Path(__file__).resolve().parents[2]


def model_snapshot_dir(root: str | Path | None = None) -> Path:
    """Canonical local path for the pinned SmolVLA policy snapshot."""

    base = Path(root) if root is not None else project_root()
    return base / ".cache" / "models" / "smolvla_libero" / MODEL_REVISION


def libero_assets_dir() -> Path:
    """Path searched by the installed ``hf-libero`` package at runtime."""

    # hf-libero 0.1.4 hard-codes this cache location in get_assets_path().
    return Path.home() / ".cache" / "libero" / "assets"


def _hub_snapshot_download(
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    target: Path,
    allow_patterns: Iterable[str] | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional simulation extra
        raise RuntimeError(
            "pinned asset preparation requires huggingface_hub; install `.[simulation,smolvla]`"
        ) from exc

    target.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "local_dir": str(target),
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    snapshot_download(**kwargs)
    return target


def prepare_model_snapshot(root: str | Path | None = None) -> Path:
    """Download/verify the policy files at the frozen model revision."""

    target = model_snapshot_dir(root)
    _hub_snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        repo_type="model",
        target=target,
    )
    required = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        raise RuntimeError(f"model snapshot is incomplete at {target}: missing {missing}")
    return target


def prepare_libero_assets(target: str | Path | None = None) -> Path:
    """Download/verify the pinned LIBERO asset repository."""

    destination = Path(target) if target is not None else libero_assets_dir()
    _hub_snapshot_download(
        repo_id=LIBERO_ASSETS_REPO,
        revision=LIBERO_ASSETS_REVISION,
        repo_type="dataset",
        target=destination,
    )
    required_dirs = ("articulated_objects", "stable_scanned_objects", "stable_hope_objects", "scenes")
    missing = [name for name in required_dirs if not (destination / name).is_dir()]
    if missing:
        raise RuntimeError(f"LIBERO assets are incomplete at {destination}: missing {missing}")
    return destination
