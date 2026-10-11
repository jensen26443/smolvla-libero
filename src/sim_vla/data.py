"""Selective LIBERO data materialization and schema checks.

The public ``HuggingFaceVLA/libero`` snapshot is a LeRobot v3 dataset whose
metadata currently reports a stale ``data/file_index`` grouping.  The parquet
files themselves are authoritative: their row-group statistics identify the
episode range stored in each file.  This module builds a small, revision-pinned
file manifest from those statistics and downloads only files intersecting the
selected episodes.

No data is downloaded when this module is imported.  The network-dependent
functions are called by ``sim-vla prepare-data`` and are intentionally kept
separate from the lightweight protocol/manifests code.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .protocol import DATASET_REPO, DATASET_REVISION, dump_json, load_json

DATA_INDEX_VERSION = "sim-vla-libero-data-v1"
METADATA_FILES = (
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
)
DATA_FILE_RE = re.compile(r"data/chunk-(?P<chunk>\d+)/file-(?P<file>\d+)\.parquet$")


def _as_unique_ids(values: Iterable[int], label: str) -> list[int]:
    ids = sorted({int(value) for value in values})
    if not ids:
        raise ValueError(f"{label} contains no episode IDs")
    if ids[0] < 0:
        raise ValueError(f"{label} contains a negative episode ID")
    return ids


def load_selected_episode_ids(path: str | Path, split: str = "all") -> list[int]:
    """Read episode IDs from an episode index or nested train manifest.

    ``split`` may be ``all``, ``train`` or ``validation``.  The accepted
    shapes intentionally match the checked-in files in ``manifests/`` and are
    kept independent of the optional LeRobot/PyArrow stack.
    """

    if split not in {"all", "train", "validation"}:
        raise ValueError("split must be all, train, or validation")
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"episode source must be a JSON object: {path}")

    if "episodes_by_task" in payload:
        groups = payload["episodes_by_task"]
    elif split != "all" and f"{split}_episodes" in payload:
        groups = payload[f"{split}_episodes"]
    elif split == "all" and {"train_episodes", "validation_episodes"}.issubset(payload):
        groups = {key: list(values) for key, values in payload["train_episodes"].items()}
        for key, values in payload["validation_episodes"].items():
            groups.setdefault(key, []).extend(values)
    elif "episodes" in payload:
        rows = payload["episodes"]
        if not isinstance(rows, list):
            raise ValueError("episodes must be a list")
        ids = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("episode rows must be objects")
            value = row.get("episode_index", row.get("episode_id"))
            if value is None:
                raise ValueError("episode row is missing episode_index")
            ids.append(int(value))
        return _as_unique_ids(ids, str(path))
    else:
        raise ValueError(f"unsupported episode source shape: {path}")

    if not isinstance(groups, Mapping):
        raise TypeError("episode groups must be a mapping from task to IDs")
    ids = [int(value) for values in groups.values() for value in values]
    return _as_unique_ids(ids, f"{path}:{split}")


def _require_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    except ImportError as exc:  # pragma: no cover - optional simulation extra
        raise RuntimeError(
            "selective data preparation requires huggingface_hub; install `.[simulation,smolvla]`"
        ) from exc
    return HfApi, hf_hub_download, hf_hub_url


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional simulation extra
        raise RuntimeError("data checks require pyarrow; install `.[simulation,smolvla]`") from exc
    return pq


class _RemoteRangeReader:
    """Small seekable HTTP range reader used for Parquet footers.

    A full data file is not downloaded while discovering the file map.  The
    LeRobot files contain one row group, and the footer statistics are enough
    to identify the episode range.
    """

    def __init__(self, url: str, size: int, block_size: int = 128 * 1024):
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - huggingface dependency
            raise RuntimeError("remote index discovery requires requests") from exc
        self._requests = requests
        self._session = requests.Session()
        self._url = url
        self._size = int(size)
        self._block_size = int(block_size)
        self._position = 0
        self._cache: dict[int, bytes] = {}
        self.closed = False

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            offset += self._position
        elif whence == io.SEEK_END:
            offset += self._size
        if whence not in {io.SEEK_SET, io.SEEK_CUR, io.SEEK_END}:
            raise ValueError(f"unsupported seek mode: {whence}")
        self._position = max(0, min(self._size, int(offset)))
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed remote parquet")
        if size < 0:
            size = self._size - self._position
        size = min(int(size), self._size - self._position)
        remaining = size
        result = bytearray()
        while remaining:
            start = (self._position // self._block_size) * self._block_size
            end = min(self._size, start + self._block_size)
            if start not in self._cache:
                response = self._session.get(
                    self._url,
                    headers={"Range": f"bytes={start}-{end - 1}"},
                    timeout=60,
                    allow_redirects=True,
                )
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Hub did not honor a range request for {self._url}: status={response.status_code}"
                    )
                self._cache[start] = response.content
            block = self._cache[start]
            offset = self._position - start
            take = min(remaining, len(block) - offset)
            if take <= 0:
                raise RuntimeError(f"empty range response while reading {self._url}")
            result.extend(block[offset : offset + take])
            self._position += take
            remaining -= take
        return bytes(result)

    def close(self) -> None:
        self.closed = True
        self._session.close()


def _remote_file_summary(
    *, repo_id: str, revision: str, filename: str, size_bytes: int, hf_hub_url: Any
) -> dict[str, Any]:
    pq = _require_pyarrow()
    url = hf_hub_url(repo_id, filename=filename, revision=revision, repo_type="dataset")
    reader = _RemoteRangeReader(url, size_bytes)
    try:
        parquet = pq.ParquetFile(reader)
        row_group = parquet.metadata.row_group(0)
        names = parquet.schema.names

        def stat(column: str) -> tuple[int | float, int | float]:
            try:
                position = names.index(column)
            except ValueError as exc:
                raise RuntimeError(f"{filename} has no {column} column") from exc
            statistics = row_group.column(position).statistics
            if statistics is None or not statistics.has_min_max:
                raise RuntimeError(f"{filename} has no min/max statistics for {column}")
            return statistics.min, statistics.max

        episode_min, episode_max = stat("episode_index")
        index_min, index_max = stat("index")
        task_min, task_max = stat("task_index")
        return {
            "path": filename,
            "size_bytes": int(size_bytes),
            "rows": int(row_group.num_rows),
            "episode_min": int(episode_min),
            "episode_max": int(episode_max),
            "index_min": int(index_min),
            "index_max": int(index_max),
            "task_min": int(task_min),
            "task_max": int(task_max),
        }
    finally:
        reader.close()


def build_data_file_manifest(
    *,
    episode_ids: Iterable[int],
    output: str | Path | None = None,
    repo_id: str = DATASET_REPO,
    revision: str = DATASET_REVISION,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Discover data parquet files overlapping ``episode_ids``.

    Discovery reads only Parquet footers through HTTP ranges.  The returned
    manifest is deterministic and can be checked into a run directory or
    reused offline for subsequent downloads.
    """

    selected = _as_unique_ids(episode_ids, "episode_ids")
    HfApi, _hf_hub_download, hf_hub_url = _require_hub()
    info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
    siblings = []
    for sibling in info.siblings:
        match = DATA_FILE_RE.fullmatch(sibling.rfilename)
        if match is None:
            continue
        size = sibling.size
        if size is None:
            raise RuntimeError(f"Hub did not report a size for {sibling.rfilename}")
        siblings.append(
            {
                "filename": sibling.rfilename,
                "size_bytes": int(size),
                "sha256": getattr(getattr(sibling, "lfs", None), "sha256", None),
                "file_index": int(match.group("file")),
            }
        )
    if not siblings:
        raise RuntimeError(f"no data parquet files found in {repo_id}@{revision}")

    summaries: list[dict[str, Any]] = []
    workers = max(1, min(int(max_workers), 32))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _remote_file_summary,
                repo_id=repo_id,
                revision=revision,
                filename=item["filename"],
                size_bytes=item["size_bytes"],
                hf_hub_url=hf_hub_url,
            ): item
            for item in siblings
        }
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            summary = future.result()
            summary["file_index"] = item["file_index"]
            summary["sha256"] = item["sha256"]
            if summary["episode_max"] >= selected[0] and summary["episode_min"] <= selected[-1]:
                summary["selected_episode_ids"] = [
                    episode for episode in selected if summary["episode_min"] <= episode <= summary["episode_max"]
                ]
                if summary["selected_episode_ids"]:
                    summaries.append(summary)
    summaries.sort(key=lambda item: item["path"])
    covered = {episode for item in summaries for episode in item["selected_episode_ids"]}
    missing = sorted(set(selected) - covered)
    if missing:
        raise RuntimeError(f"remote data index did not locate selected episode(s): {missing[:10]}")

    result = {
        "index_version": DATA_INDEX_VERSION,
        "source_dataset": repo_id,
        "dataset_revision": revision,
        "episode_ids": selected,
        "metadata_files": list(METADATA_FILES),
        "files": summaries,
        "selected_file_count": len(summaries),
        "selected_data_bytes": sum(item["size_bytes"] for item in summaries),
        "notes": [
            "File selection uses Parquet row-group statistics, not meta/episodes data/file_index.",
            "The pinned Hub snapshot has 377 data shards; selected episodes for this protocol occupy a subset.",
        ],
    }
    if output is not None:
        dump_json(result, output)
    return result


def _load_or_build_file_manifest(
    *,
    path: str | Path,
    episode_ids: list[int],
    repo_id: str,
    revision: str,
) -> dict[str, Any]:
    target = Path(path)
    if target.is_file():
        manifest = load_json(target)
        if not isinstance(manifest, Mapping):
            raise ValueError(f"data file manifest must be an object: {target}")
        if manifest.get("source_dataset") != repo_id or manifest.get("dataset_revision") != revision:
            raise ValueError(f"data file manifest source does not match {repo_id}@{revision}: {target}")
        known = {int(value) for value in manifest.get("episode_ids", [])}
        if set(episode_ids).issubset(known):
            return dict(manifest)
    return build_data_file_manifest(episode_ids=episode_ids, output=target, repo_id=repo_id, revision=revision)


def _download_one(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    root: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    _HfApi, hf_hub_download, _hf_hub_url = _require_hub()
    destination = root / filename
    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        if expected_sha256 is not None:
            digest = _sha256_file(destination)
            if digest != expected_sha256:
                raise RuntimeError(
                    f"existing file checksum mismatch for {filename}: {digest} != {expected_sha256}"
                )
        return {
            "path": filename,
            "bytes": destination.stat().st_size,
            "downloaded": False,
            **({"sha256": expected_sha256, "sha256_verified": True} if expected_sha256 else {}),
        }
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            local_dir=str(root),
        )
    )
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(f"downloaded size mismatch for {filename}: {path.stat().st_size} != {expected_size}")
    digest = _sha256_file(path) if expected_sha256 is not None else None
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"downloaded checksum mismatch for {filename}: {digest} != {expected_sha256}")
    return {
        "path": filename,
        "bytes": path.stat().st_size,
        "downloaded": True,
        **({"sha256": expected_sha256, "sha256_verified": True} if expected_sha256 else {}),
    }


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_local_dataset(root: Path, episode_ids: list[int], repo_id: str, revision: str) -> dict[str, Any]:
    # Keep the Arrow/Datasets fingerprint cache alongside the selected data.
    # This makes an offline verification work on hosts where the global
    # ``~/.cache/huggingface`` is read-only (common in managed runners).
    os.environ.setdefault("HF_DATASETS_CACHE", str(root.parent / "datasets-cache"))
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - optional simulation extra
        raise RuntimeError("dataset verification requires lerobot; install `.[simulation,smolvla]`") from exc
    import torch

    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episode_ids,
        revision=revision,
        download_videos=False,
    )
    if dataset.meta.video_keys:
        raise RuntimeError(f"unexpected video-backed keys in local LIBERO data: {dataset.meta.video_keys}")
    loaded_episodes = {
        int(value.item() if isinstance(value, torch.Tensor) else value)
        for value in dataset.reader.hf_dataset.unique("episode_index")
    }
    missing = sorted(set(episode_ids) - loaded_episodes)
    if missing:
        raise RuntimeError(f"local LeRobotDataset did not load episode(s): {missing[:10]}")

    samples: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        task = str(item["task"])
        if task in samples:
            continue
        image_shapes = {}
        image_ranges = {}
        for key in dataset.meta.camera_keys:
            image = item[key]
            image_shapes[key] = list(image.shape)
            image_ranges[key] = [float(image.min().item()), float(image.max().item())]
            if tuple(image.shape) != (3, 256, 256) or not torch.isfinite(image).all():
                raise RuntimeError(f"invalid image tensor for {key}: {tuple(image.shape)}")
        state = item["observation.state"]
        action = item["action"]
        if tuple(state.shape) != (8,) or tuple(action.shape) != (7,):
            raise RuntimeError(f"invalid state/action shapes: {tuple(state.shape)}, {tuple(action.shape)}")
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise RuntimeError("state/action contains non-finite values")
        if float(action.abs().max()) > 1.0001:
            raise RuntimeError("dataset action exceeds the expected [-1, 1] range")
        samples[task] = {
            "episode_index": int(item["episode_index"].item()),
            "task_index": int(item["task_index"].item()),
            "image_shapes": image_shapes,
            "image_ranges": image_ranges,
            "state_shape": list(state.shape),
            "action_shape": list(action.shape),
            "state_finite": bool(torch.isfinite(state).all()),
            "action_finite": bool(torch.isfinite(action).all()),
        }
        if len(samples) >= min(3, dataset.meta.total_tasks):
            break
    if not samples:
        raise RuntimeError("local dataset contains no readable samples")
    return {
        "num_frames": len(dataset),
        "num_loaded_episodes": len(loaded_episodes),
        "loaded_episode_min": min(loaded_episodes),
        "loaded_episode_max": max(loaded_episodes),
        "video_keys": list(dataset.meta.video_keys),
        "features": dataset.meta.features,
        "task_samples": samples,
    }


def prepare_data(
    *,
    episode_source: str | Path = "manifests/libero_episode_index.json",
    split: str = "all",
    file_manifest: str | Path = "manifests/libero_data_files.json",
    root: str | Path = ".cache/lerobot/libero",
    output: str | Path = "runs/data_prepare.json",
    repo_id: str = DATASET_REPO,
    revision: str = DATASET_REVISION,
    download: bool = True,
    verify: bool = True,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Materialize and validate only files needed by an episode selection."""

    started = time.time()
    episode_ids = load_selected_episode_ids(episode_source, split=split)
    manifest = _load_or_build_file_manifest(
        path=file_manifest,
        episode_ids=episode_ids,
        repo_id=repo_id,
        revision=revision,
    )
    selected_files = [
        item
        for item in manifest["files"]
        if set(episode_ids) & {int(value) for value in item["selected_episode_ids"]}
    ]
    if not selected_files:
        raise RuntimeError("file manifest has no files for the requested episode selection")
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    download_records: list[dict[str, Any]] = []
    if download:
        metadata = [
            _download_one(repo_id=repo_id, revision=revision, filename=name, root=target)
            for name in METADATA_FILES
        ]
        download_records.extend(metadata)
        workers = max(1, min(int(max_workers), 8))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _download_one,
                    repo_id=repo_id,
                    revision=revision,
                    filename=item["path"],
                    root=target,
                    expected_size=int(item["size_bytes"]),
                    expected_sha256=item.get("sha256"),
                )
                for item in selected_files
            ]
            download_records.extend(future.result() for future in futures)
    verification = _verify_local_dataset(target, episode_ids, repo_id, revision) if verify else None
    result: dict[str, Any] = {
        "status": "passed" if (not verify or verification is not None) else "unknown",
        "source_dataset": repo_id,
        "dataset_revision": revision,
        "episode_source": str(episode_source),
        "split": split,
        "episode_count": len(episode_ids),
        "episode_ids": episode_ids,
        "file_manifest": str(file_manifest),
        "data_root": str(target),
        "selected_file_count": len(selected_files),
        "selected_data_bytes": sum(int(item["size_bytes"]) for item in selected_files),
        "metadata_files": list(METADATA_FILES),
        "download": download_records,
        "verification": verification,
        "elapsed_seconds": time.time() - started,
    }
    dump_json(result, output)
    return result


def build_data_manifest_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="discover revision-pinned LIBERO data shards via Parquet footers")
    parser.add_argument("--episode-source", default="manifests/libero_episode_index.json")
    parser.add_argument("--output", default="manifests/libero_data_files.json")
    parser.add_argument("--repo-id", default=DATASET_REPO)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args(argv)
    ids = load_selected_episode_ids(args.episode_source)
    result = build_data_file_manifest(
        episode_ids=ids,
        output=args.output,
        repo_id=args.repo_id,
        revision=args.revision,
        max_workers=args.max_workers,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "episode_count": len(ids),
                "selected_file_count": result["selected_file_count"],
                "selected_data_bytes": result["selected_data_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def prepare_data_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="download and verify selected LIBERO LeRobot data")
    parser.add_argument("--episode-source", default="manifests/libero_episode_index.json")
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all")
    parser.add_argument("--file-manifest", default="manifests/libero_data_files.json")
    parser.add_argument("--root", default=".cache/lerobot/libero")
    parser.add_argument("--output", default="runs/data_prepare.json")
    parser.add_argument("--repo-id", default=DATASET_REPO)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_data(
        episode_source=args.episode_source,
        split=args.split,
        file_manifest=args.file_manifest,
        root=args.root,
        output=args.output,
        repo_id=args.repo_id,
        revision=args.revision,
        download=not args.no_download,
        verify=not args.no_verify,
        max_workers=args.max_workers,
    )
    print(
        f"LIBERO data preparation passed; {result['episode_count']} episodes, "
        f"{result['selected_file_count']} shards, record written to {args.output}"
    )
    return 0
