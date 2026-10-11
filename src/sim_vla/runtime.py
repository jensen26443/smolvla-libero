"""Runtime setup that keeps LIBERO configuration inside this project."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from .assets import libero_assets_dir, project_root


def _libero_package_root() -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("libero is not importable; install the simulation extras first")
    return Path(next(iter(spec.submodule_search_locations))) / "libero"


def configure_libero(root: str | Path | None = None, assets: str | Path | None = None) -> Path:
    """Set a non-interactive, project-local LIBERO config before importing it.

    ``hf-libero`` asks an interactive question the first time it is imported.
    Creating this small config first makes CI/smoke commands deterministic and
    avoids writing an untracked config into ``~/.libero``.  The asset cache is
    still the location expected by hf-libero 0.1.4 unless an explicit
    ``SIM_VLA_LIBERO_ASSETS`` override is supplied.
    """

    root_path = Path(root) if root is not None else project_root()
    config_dir = root_path / ".cache" / "libero-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    package_root = _libero_package_root()
    asset_path = Path(
        assets
        if assets is not None
        else os.environ.get("SIM_VLA_LIBERO_ASSETS", str(libero_assets_dir()))
    )
    config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(root_path / "data" / "libero"),
        "assets": str(asset_path),
    }
    config_file = config_dir / "config.yaml"
    # YAML is already an hf-libero dependency. Keep this write idempotent so a
    # repeated smoke run does not alter the working tree beyond the ignored cache.
    import yaml

    rendered = yaml.safe_dump(config, sort_keys=False)
    if not config_file.exists() or config_file.read_text(encoding="utf-8") != rendered:
        config_file.write_text(rendered, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return config_file


def import_libero(root: str | Path | None = None, assets: str | Path | None = None) -> Any:
    """Configure and import ``libero.libero`` without its first-run prompt."""

    configure_libero(root=root, assets=assets)
    module = __import__("libero.libero", fromlist=["*"])
    asset_path = Path(
        assets
        if assets is not None
        else os.environ.get("SIM_VLA_LIBERO_ASSETS", str(libero_assets_dir()))
    )
    if asset_path.is_dir():
        # get_assets_path() consults this cache before attempting an unpinned
        # download from Hub. This makes the selected asset revision explicit.
        module._assets_path_cache = str(asset_path)
    return module

