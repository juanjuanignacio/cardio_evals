"""Hidden state extraction to HDF5 with checkpointing."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# HDF5 Schema
# ---------------------------------------------------------------------------
# /activations/{position}/  shape=(n_samples, n_layers+1, d_model) float32
#   layer 0 = embedding (hook_embed + hook_pos_embed)
#   layer i = blocks.{i-1}.hook_resid_post  (1-indexed)
# /sample_metadata/         parallel arrays of length n_samples
# /extraction_progress      scalar int32: last completed sample index


def _create_h5_file(
    path: str,
    n_samples: int,
    n_layers: int,
    d_model: int,
    positions: List[str],
    model_name: str,
) -> h5py.File:
    """Create (or reopen) HDF5 file with pre-allocated datasets."""
    f = h5py.File(path, "a")
    f.attrs["model_name"] = model_name
    f.attrs["n_samples"] = n_samples
    f.attrs["n_layers"] = n_layers
    f.attrs["d_model"] = d_model
    f.attrs.setdefault("created_at", datetime.utcnow().isoformat())

    act = f.require_group("activations")
    for pos in positions:
        if pos not in act:
            act.create_dataset(
                pos,
                shape=(n_samples, n_layers + 1, d_model),
                dtype=np.float32,
                chunks=(min(64, n_samples), 1, d_model),
                compression="gzip",
                compression_opts=4,
            )

    meta = f.require_group("sample_metadata")
    str_dt = h5py.special_dtype(vlen=str)
    for col in ["response_source", "source_predicted", "medical_specialty", "split"]:
        if col not in meta:
            meta.create_dataset(col, shape=(n_samples,), dtype=str_dt)
    for col in ["accuracy_score", "clarity_score", "completeness_score"]:
        if col not in meta:
            meta.create_dataset(col, shape=(n_samples,), dtype=np.int8)
    for col in ["sample_idx", "score_token_pos", "response_start_pos", "inst_end_pos", "question_end_pos"]:
        if col not in meta:
            meta.create_dataset(col, shape=(n_samples,), dtype=np.int32)

    if "extraction_progress" not in f:
        f.create_dataset("extraction_progress", data=np.int32(-1))

    return f


def extract_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    sample_metadata: List[Dict[str, Any]],
    output_path: str,
    positions: List[str],
    batch_size: int = 8,
    device: str = "cuda:0",
    checkpoint_every: int = 100,
    max_seq_len: int = 1024,
) -> None:
    """
    Main extraction loop. Extracts residual stream at specified token positions
    for all samples, saves incrementally to HDF5.

    Uses TransformerLens run_with_cache() with names_filter to extract only
    hook_resid_post activations (memory efficient).

    Checkpointing: skips samples already extracted (progress stored in HDF5).
    """
    n_samples = len(prompts)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    f = _create_h5_file(output_path, n_samples, n_layers, d_model, positions, model.cfg.model_name)

    start_idx = int(f["extraction_progress"][()]) + 1
    if start_idx >= n_samples:
        print(f"Extraction already complete ({n_samples} samples).")
        f.close()
        return

    print(f"Extracting hidden states: samples {start_idx} to {n_samples - 1}")
    print(f"Output: {output_path}")

    # Hook names for residual stream at each layer
    # Layer 0 = embedding; layer i = hook_resid_post of block i-1
    def resid_names_filter(name: str) -> bool:
        return "hook_resid_post" in name or name == "hook_embed" or name == "hook_pos_embed"

    model.eval()
    batch_buffer: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        pbar = tqdm(range(start_idx, n_samples, batch_size), desc="Extracting")
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, n_samples)
            batch_prompts = prompts[batch_start:batch_end]
            batch_meta = sample_metadata[batch_start:batch_end]

            # Tokenize
            tokenizer.padding_side = "left"
            tokens = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            actual_batch = input_ids.shape[0]

            # Run with cache — only residual stream hooks
            _, cache = model.run_with_cache(
                input_ids,
                attention_mask=attention_mask,
                names_filter=resid_names_filter,
                return_type=None,
            )

            # Build layer-indexed residual stream tensor
            # Shape: (batch, n_layers+1, seq_len, d_model)
            layer_acts = []
            # Layer 0: embedding
            embed = cache["hook_embed"]  # (batch, seq, d_model)
            pos_embed = cache["hook_pos_embed"] if "hook_pos_embed" in cache else torch.zeros_like(embed)
            layer_acts.append((embed + pos_embed).cpu().float().numpy())
            # Layers 1..n_layers: hook_resid_post
            for layer in range(n_layers):
                key = f"blocks.{layer}.hook_resid_post"
                layer_acts.append(cache[key].cpu().float().numpy())
            # Stack: (n_layers+1, batch, seq, d_model) → (batch, n_layers+1, seq, d_model)
            all_acts = np.stack(layer_acts, axis=1)  # (batch, n_layers+1, seq, d_model)

            # Find token positions for each sample
            pos_indices = _compute_position_indices(
                input_ids, tokenizer, batch_meta, positions
            )

            # Write to HDF5
            for b in range(actual_batch):
                global_idx = batch_start + b
                for pos_name in positions:
                    tok_idx = pos_indices[pos_name][b]
                    # Clamp to valid seq range
                    tok_idx = min(tok_idx, all_acts.shape[3] - 1)
                    f["activations"][pos_name][global_idx] = all_acts[b, :, tok_idx, :]

                # Write metadata
                meta = batch_meta[b]
                f["sample_metadata"]["response_source"][global_idx] = meta.get("response_source", "")
                f["sample_metadata"]["source_predicted"][global_idx] = meta.get("source_predicted", "")
                f["sample_metadata"]["medical_specialty"][global_idx] = meta.get("medical_specialty", "")
                f["sample_metadata"]["split"][global_idx] = meta.get("split", "")
                f["sample_metadata"]["accuracy_score"][global_idx] = meta.get("accuracy_score", -1)
                f["sample_metadata"]["clarity_score"][global_idx] = meta.get("clarity_score", -1)
                f["sample_metadata"]["completeness_score"][global_idx] = meta.get("completeness_score", -1)
                f["sample_metadata"]["sample_idx"][global_idx] = global_idx
                for pos_name in positions:
                    col = f"{pos_name}_pos"
                    if col in f["sample_metadata"]:
                        f["sample_metadata"][col][global_idx] = pos_indices[pos_name][b]

            # Checkpoint
            f["extraction_progress"][()] = batch_end - 1
            if (batch_end - start_idx) % checkpoint_every == 0:
                f.flush()

            del cache, all_acts, input_ids, attention_mask
            pbar.set_postfix({"last": batch_end - 1})

    f["extraction_progress"][()] = n_samples - 1
    f.flush()
    f.close()
    print(f"\nExtraction complete. File: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1e9:.2f} GB")


def extract_hidden_states_hf(
    model,
    tokenizer,
    prompts: List[str],
    sample_metadata: List[Dict[str, Any]],
    output_path: str,
    positions: List[str],
    n_layers: int,
    d_model: int,
    model_name: str = "",
    batch_size: int = 4,
    checkpoint_every: int = 50,
    max_seq_len: int = 1024,
) -> None:
    """
    Extraction for HuggingFace multi-GPU models (no TransformerLens).

    Uses PyTorch forward hooks on model.model.layers[i] to capture the residual
    stream at each decoder layer. Works with device_map="auto" across multiple GPUs
    because hooks fire on whatever device the layer runs on, and we immediately
    move activations to CPU.

    Layer mapping (matches TL convention):
      layer 0 → embedding (model.model.embed_tokens output)
      layer i → model.model.layers[i-1] output[0]  (residual post block i-1)
    """
    n_samples = len(prompts)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    f = _create_h5_file(output_path, n_samples, n_layers, d_model, positions, model_name)

    start_idx = int(f["extraction_progress"][()]) + 1
    if start_idx >= n_samples:
        print(f"Extraction already complete ({n_samples} samples).")
        f.close()
        return

    print(f"Extracting (HF multi-GPU): samples {start_idx} to {n_samples - 1}")

    model.eval()
    with torch.no_grad():
        pbar = tqdm(range(start_idx, n_samples, batch_size), desc="Extracting HF")
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, n_samples)
            batch_prompts = prompts[batch_start:batch_end]
            batch_meta = sample_metadata[batch_start:batch_end]
            actual_batch = len(batch_prompts)

            tokenizer.padding_side = "left"
            tokens = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            # Keep input_ids on CPU for position finding; move to first GPU for model
            input_ids_cpu = tokens["input_ids"]
            input_ids = input_ids_cpu.to("cuda:0")
            attention_mask = tokens["attention_mask"].to("cuda:0")

            # Register hooks to capture each layer's output (residual stream)
            captured: Dict[int, np.ndarray] = {}  # layer_idx → (batch, seq, d_model) float32

            def make_embed_hook():
                def hook(module, input, output):
                    captured[0] = output.detach().cpu().float().numpy()
                return hook

            def make_layer_hook(layer_idx: int):
                def hook(module, input, output):
                    # output is a tuple; first element is the hidden state
                    h = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx + 1] = h.detach().cpu().float().numpy()
                return hook

            handles = []
            handles.append(model.model.embed_tokens.register_forward_hook(make_embed_hook()))
            for i in range(n_layers):
                handles.append(model.model.layers[i].register_forward_hook(make_layer_hook(i)))

            try:
                model(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                for h in handles:
                    h.remove()

            # Stack into (batch, n_layers+1, seq, d_model)
            all_acts = np.stack([captured[i] for i in range(n_layers + 1)], axis=1)

            pos_indices = _compute_position_indices(input_ids_cpu, tokenizer, batch_meta, positions)

            for b in range(actual_batch):
                global_idx = batch_start + b
                for pos_name in positions:
                    tok_idx = min(pos_indices[pos_name][b], all_acts.shape[3] - 1)
                    f["activations"][pos_name][global_idx] = all_acts[b, :, tok_idx, :]
                meta = batch_meta[b]
                f["sample_metadata"]["response_source"][global_idx] = meta.get("response_source", "")
                f["sample_metadata"]["source_predicted"][global_idx] = meta.get("source_predicted", "")
                f["sample_metadata"]["medical_specialty"][global_idx] = meta.get("medical_specialty", "")
                f["sample_metadata"]["split"][global_idx] = meta.get("split", "")
                f["sample_metadata"]["accuracy_score"][global_idx] = meta.get("accuracy_score", -1)
                f["sample_metadata"]["clarity_score"][global_idx] = meta.get("clarity_score", -1)
                f["sample_metadata"]["completeness_score"][global_idx] = meta.get("completeness_score", -1)
                f["sample_metadata"]["sample_idx"][global_idx] = global_idx
                for pos_name in positions:
                    col = f"{pos_name}_pos"
                    if col in f["sample_metadata"]:
                        f["sample_metadata"][col][global_idx] = pos_indices[pos_name][b]

            f["extraction_progress"][()] = batch_end - 1
            if (batch_end - start_idx) % checkpoint_every == 0:
                f.flush()

            del all_acts, input_ids, attention_mask, captured
            pbar.set_postfix({"last": batch_end - 1})

    f["extraction_progress"][()] = n_samples - 1
    f.flush()
    f.close()
    print(f"\nExtraction complete. File: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1e9:.2f} GB")


def _compute_position_indices(
    input_ids: torch.Tensor,
    tokenizer,
    batch_meta: List[Dict],
    positions: List[str],
) -> Dict[str, List[int]]:
    """Compute token position indices for each position type."""
    from src.data import (
        find_inst_end_position,
        find_question_end_position,
        find_response_start_position,
        find_score_token_position,
    )
    result = {}
    seq_len = input_ids.shape[1]
    if "score_token" in positions:
        result["score_token"] = find_score_token_position(input_ids, tokenizer)
    if "response_start" in positions:
        # Use response text from metadata
        positions_list = []
        for b, meta in enumerate(batch_meta):
            resp = meta.get("response", "")
            single = input_ids[b:b+1]
            positions_list.append(
                find_response_start_position(single, tokenizer, resp)[0]
            )
        result["response_start"] = positions_list
    if "inst_end" in positions:
        result["inst_end"] = find_inst_end_position(input_ids, tokenizer)
    if "question_end" in positions:
        positions_list = []
        for b, meta in enumerate(batch_meta):
            q = meta.get("question", "")
            single = input_ids[b:b+1]
            positions_list.append(
                find_question_end_position(single, tokenizer, q)[0]
            )
        result["question_end"] = positions_list
    return result


# ---------------------------------------------------------------------------
# Loading from HDF5
# ---------------------------------------------------------------------------

def load_hidden_states(
    h5_path: str,
    position: str,
    layer: Optional[int] = None,
    split: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load hidden states and metadata from HDF5.

    Returns:
        activations: shape (n, n_layers+1, d_model) or (n, d_model) if layer given
        metadata: dict of parallel arrays (response_source, accuracy_score, ...)
    """
    with h5py.File(h5_path, "r") as f:
        acts = f["activations"][position][:]  # (n_samples, n_layers+1, d_model)
        meta = {}
        for col in f["sample_metadata"]:
            raw = f["sample_metadata"][col][:]
            # Decode byte strings
            if raw.dtype.kind in ("O", "S"):
                raw = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in raw])
            meta[col] = raw

    # Filter by split
    if split is not None:
        mask = meta["split"] == split
        acts = acts[mask]
        meta = {k: v[mask] for k, v in meta.items()}

    if layer is not None:
        acts = acts[:, layer, :]  # (n, d_model)

    return acts, meta


def compute_mean_diff_vector(
    h5_path: str,
    position: str,
    layer: int,
    split: str = "train",
    ai_sources: Tuple[str, ...] = ("AI", "CoT AI"),
    human_sources: Tuple[str, ...] = ("Human",),
) -> np.ndarray:
    """
    Compute mean(h_AI) - mean(h_human) at (position, layer).
    Returns unit-normalized direction vector, shape (d_model,).
    """
    acts, meta = load_hidden_states(h5_path, position, layer=layer, split=split)
    sources = meta["response_source"]

    ai_mask = np.isin(sources, list(ai_sources))
    human_mask = np.isin(sources, list(human_sources))

    mean_ai = acts[ai_mask].mean(axis=0)
    mean_human = acts[human_mask].mean(axis=0)

    direction = mean_ai - mean_human
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction = direction / norm
    return direction.astype(np.float32)


def compute_mean_diff_vector_indexed(
    h5_path: str,
    position: str,
    layer: int,
    train_indices: np.ndarray,
    ai_sources: Tuple[str, ...] = ("AI", "CoT AI"),
    human_sources: Tuple[str, ...] = ("Human",),
) -> np.ndarray:
    """
    Compute mean(h_AI) - mean(h_human) using only specific sample indices.
    Used for per-fold steering vector computation (avoids data leakage).
    Returns unit-normalized direction, shape (d_model,).
    """
    acts, meta = load_hidden_states(h5_path, position, layer=layer)
    sources = meta["response_source"]

    ai_mask = np.zeros(len(sources), dtype=bool)
    human_mask = np.zeros(len(sources), dtype=bool)
    ai_mask[train_indices] = np.isin(sources[train_indices], list(ai_sources))
    human_mask[train_indices] = np.isin(sources[train_indices], list(human_sources))

    mean_ai = acts[ai_mask].mean(axis=0)
    mean_human = acts[human_mask].mean(axis=0)

    direction = mean_ai - mean_human
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction = direction / norm
    return direction.astype(np.float32)


def extract_attention_patterns_for_heads(
    model,
    tokenizer,
    prompts: List[str],
    head_list: List[Tuple[int, int]],
    output_path: str,
    batch_size: int = 4,
    device: str = "cuda:0",
    max_seq_len: int = 1024,
) -> None:
    """
    Extract attention patterns for specific (layer, head) pairs.
    Saves to HDF5 at /attention_patterns/L{layer}_H{head}/.
    """
    if not head_list:
        return

    n_samples = len(prompts)
    hook_names = set()
    for layer, _ in head_list:
        hook_names.add(f"blocks.{layer}.attn.hook_pattern")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "a") as f:
        model.eval()
        with torch.no_grad():
            for batch_start in tqdm(range(0, n_samples, batch_size), desc="Attn patterns"):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_prompts = prompts[batch_start:batch_end]
                tokens = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                ).to(device)

                _, cache = model.run_with_cache(
                    tokens["input_ids"],
                    names_filter=lambda n: "hook_pattern" in n and any(
                        f"blocks.{l}.attn" in n for l, _ in head_list
                    ),
                    return_type=None,
                )
                seq_len = tokens["input_ids"].shape[1]
                for layer, head in head_list:
                    key = f"blocks.{layer}.attn.hook_pattern"
                    pattern = cache[key][:, head, :, :].cpu().float().numpy()
                    ds_name = f"attention_patterns/L{layer}_H{head}"
                    if ds_name not in f:
                        f.create_dataset(
                            ds_name,
                            shape=(n_samples, seq_len, seq_len),
                            dtype=np.float32,
                            chunks=(1, seq_len, seq_len),
                        )
                    f[ds_name][batch_start:batch_end, :, :] = pattern

                del cache
