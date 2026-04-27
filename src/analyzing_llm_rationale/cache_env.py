from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_cache_paths(root: Path | None = None) -> dict[str, Path]:
    cache_root = (root or repo_root()) / ".cache"
    hf_home = cache_root / "huggingface"
    hf_hub_cache = hf_home / "hub"
    return {
        "cache_root": cache_root,
        "xdg_cache_home": cache_root / "xdg",
        "hf_home": hf_home,
        "hf_hub_cache": hf_hub_cache,
        "transformers_cache": hf_home / "transformers",
        "hf_datasets_cache": hf_home / "datasets",
        "hf_assets_cache": hf_home / "assets",
        "hf_modules_cache": hf_home / "modules",
        "torch_home": cache_root / "torch",
        "torchinductor_cache_dir": cache_root / "torchinductor",
        "triton_cache_dir": cache_root / "triton",
        "pip_cache_dir": cache_root / "pip",
    }


def configure_workspace_cache_env(
    root: Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Path]:
    env = os.environ if environ is None else environ
    paths = workspace_cache_paths(root)
    assignments = {
        "XDG_CACHE_HOME": paths["xdg_cache_home"],
        "HF_HOME": paths["hf_home"],
        "HF_HUB_CACHE": paths["hf_hub_cache"],
        "HUGGINGFACE_HUB_CACHE": paths["hf_hub_cache"],
        "TRANSFORMERS_CACHE": paths["transformers_cache"],
        "HF_DATASETS_CACHE": paths["hf_datasets_cache"],
        "HF_ASSETS_CACHE": paths["hf_assets_cache"],
        "HF_MODULES_CACHE": paths["hf_modules_cache"],
        "TORCH_HOME": paths["torch_home"],
        "TORCHINDUCTOR_CACHE_DIR": paths["torchinductor_cache_dir"],
        "TRITON_CACHE_DIR": paths["triton_cache_dir"],
        "PIP_CACHE_DIR": paths["pip_cache_dir"],
    }

    resolved = {"cache_root": paths["cache_root"]}
    for env_var, default_path in assignments.items():
        if not env.get(env_var):
            env[env_var] = str(default_path)
        resolved[env_var] = Path(env[env_var])
        resolved[env_var].mkdir(parents=True, exist_ok=True)

    return resolved
