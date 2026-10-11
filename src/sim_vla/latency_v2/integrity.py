"""Content fingerprints without truncating model weights or altering assets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, files: Iterable[Path]) -> dict:
    entries = []
    for path in sorted(files):
        if path.is_file():
            entries.append({
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            })
    return {
        "sha256": hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": entries,
    }


def source_inventory(root: Path, *, legacy_only: bool = False) -> dict:
    files = []
    for directory in ("src/sim_vla", "configs", "tests", "scripts", "docs", "manifests"):
        for path in (root / directory).rglob("*"):
            if path.suffix not in {".py", ".json", ".yaml", ".sh", ".md"}:
                continue
            if "__pycache__" in path.parts:
                continue
            if legacy_only and ("latency_v2" in path.parts or "latency_v2" in path.name):
                continue
            files.append(path)
    files.extend(root / name for name in ("README.md", "pyproject.toml", "uv.lock"))
    return inventory(root, files)
