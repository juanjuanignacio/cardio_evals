"""Configuration loading and global seed management.

Secrets are NEVER read from YAML. Pass them via environment variables:
    HF_TOKEN   — HuggingFace API token (loaded by python-dotenv from .env)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class ModelConfig:
    name: str
    key: str
    dtype: str
    device: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    evaluator_label: str = ""      # label used in dataset 'evaluator' column
    use_hf_backend: bool = False   # use HF multi-GPU (device_map="auto") instead of TransformerLens


@dataclass
class ExtractionConfig:
    batch_size: int
    max_seq_len: int
    positions: List[str]
    dtype: str
    checkpoint_every: int


@dataclass
class ProbingConfig:
    probe_types: List[str]
    score_low_threshold: int
    score_high_threshold: int
    sklearn_params: Dict[str, Any]
    n_bootstrap: int


@dataclass
class ProjectConfig:
    seed: int
    paths: Dict[str, str]
    dataset: Dict[str, Any]
    primary_model: ModelConfig
    comparison_models: List[ModelConfig]
    extraction: ExtractionConfig
    probing: ProbingConfig
    patching: Dict[str, Any]
    steering: Dict[str, Any]
    ablation: Dict[str, Any]
    sycophancy: Dict[str, Any]
    additional: Dict[str, Any]
    format_instructions: str


def load_config(config_path: str = "config/config.yaml") -> ProjectConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    primary = ModelConfig(**raw["primary_model"])
    comparisons = [ModelConfig(**m) for m in raw.get("comparison_models", [])]

    # Secrets come from the environment, not from the YAML. Env var wins always.
    dataset = dict(raw["dataset"])
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if env_token:
        dataset["hf_token"] = env_token
    elif dataset.get("hf_token"):
        # YAML still has a value — refuse it. Force env var usage.
        raise RuntimeError(
            "hf_token must not be set in config.yaml. "
            "Remove it and export HF_TOKEN as an environment variable "
            "(see .env.example)."
        )
    else:
        dataset["hf_token"] = None

    return ProjectConfig(
        seed=raw["seed"],
        paths=raw["paths"],
        dataset=dataset,
        primary_model=primary,
        comparison_models=comparisons,
        extraction=ExtractionConfig(**raw["extraction"]),
        probing=ProbingConfig(**raw["probing"]),
        patching=raw["patching"],
        steering=raw["steering"],
        ablation=raw["ablation"],
        sycophancy=raw["sycophancy"],
        additional=raw["additional"],
        format_instructions=raw.get("format_instructions", ""),
    )


def get_model_config(cfg: ProjectConfig, model_key: str) -> ModelConfig:
    if cfg.primary_model.key == model_key:
        return cfg.primary_model
    for m in cfg.comparison_models:
        if m.key == model_key:
            return m
    raise KeyError(f"Model key '{model_key}' not found in config")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        # is_initialized() does NOT open the CUDA context — is_available() does.
        # Opening CUDA here causes kernel panics on exit when CUDA_VISIBLE_DEVICES
        # points to a non-existent GPU. CUDA seed is set later when the model loads.
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import transformers
        transformers.set_seed(seed)
    except (ImportError, AttributeError):
        pass


def resolve_path(cfg: ProjectConfig, key: str, root: Optional[str] = None) -> Path:
    """Return absolute Path for a config path key, creating the dir if needed."""
    rel = cfg.paths[key]
    base = Path(root) if root else Path.cwd()
    p = base / rel
    p.mkdir(parents=True, exist_ok=True)
    return p
