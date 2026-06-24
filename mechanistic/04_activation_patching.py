#!/usr/bin/env python3
"""
Block 2 — Step 1: Activation patching to find which (layer, component) pairs carry the bias.

Algorithm:
  - Clean = AI response prompt (model gives high score)
  - Corrupted = Human response prompt (same question, lower score)
  - For each layer: patch clean → corrupted residual stream at last token
  - Measure: does score metric go up? → that layer carries the bias

Backends:
  - TransformerLens (HookedTransformer): small models (≤14B), full mechanistic interp.
  - HuggingFace multi-GPU (device_map="auto"): 70B models — patches via PyTorch hooks.
    attn_out and mlp_out are NOT decomposed in HF backend mode (zeros in output).
    use_hf_backend is set in config.yaml per model.

Outputs:
  results/patching/{model_key}_resid_patching.npy    (n_pairs, n_layers)
  results/patching/{model_key}_attn_patching.npy
  results/patching/{model_key}_mlp_patching.npy
  results/patching/{model_key}_patching_summary.json

Usage:
    python blocks/block2_circuits/01_activation_patching.py
    python blocks/block2_circuits/01_activation_patching.py --model_key llama-3.3-70b --device cuda:0
    python blocks/block2_circuits/01_activation_patching.py --n_pairs 50 --dry_run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, set_global_seed, get_model_config
from src.data import build_judge_prompt, load_cardio_dataset, get_unique_responses, get_matched_pairs
from src.model import free_model_memory, get_dtype, load_hooked_transformer, load_hf_model_large, exit_cleanly
from src.patching import (
    build_score_metric, run_activation_patching,
    build_score_metric_hf, run_activation_patching_hf,
)
from src.visualization import plot_patching_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n_pairs", type=int, default=None)
    parser.add_argument("--model_key", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_global_seed(cfg.seed)

    model_key = args.model_key or cfg.primary_model.key
    model_cfg = get_model_config(cfg, model_key)

    patch_dir = ROOT / cfg.paths["patch_dir"]
    plot_dir  = ROOT / cfg.paths["plot_dir"]
    patch_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_pairs = args.n_pairs or cfg.patching["n_pairs"]
    if args.dry_run:
        n_pairs = 5
        print("[DRY RUN] Using 5 pairs")

    device    = args.device or model_cfg.device
    use_hf_backend  = getattr(model_cfg, "use_hf_backend", False)
    n_layers  = model_cfg.n_layers
    fmt_inst  = cfg.format_instructions

    print(f"=== Block 2.1: Activation Patching ===")
    if not args.dry_run and (patch_dir / f"{model_key}_patching_summary.json").exists():
        print(f"[{model_key}] Patching already complete. Skipping.")
        return
    print(f"Model: {model_key} | n_pairs: {n_pairs} | device: {device} | hf_backend: {use_hf_backend}")

    # Load data (filtered to target evaluator for consistent pair matching)
    raw_ds = load_cardio_dataset(cfg)
    ds = get_unique_responses(
        raw_ds,
        target_evaluator=cfg.dataset.get("target_evaluator", "Llama3.1:8b"),
        filter_uninformative=True,
    )

    print(f"\nMatching {n_pairs} AI-Human pairs...")
    pairs = get_matched_pairs(
        ds,
        n_pairs=n_pairs,
        seed=cfg.seed,
        ai_sources=tuple(cfg.dataset["ai_sources"]),
        human_sources=tuple(cfg.dataset["human_sources"]),
    )
    print(f"Found {len(pairs)} pairs")

    # Load model
    print(f"\nLoading model (use_hf_backend={use_hf_backend})...")
    if use_hf_backend:
        model, tokenizer = load_hf_model_large(
            model_cfg.name, hf_token=cfg.dataset.get("hf_token")
        )
        metric_fn = build_score_metric_hf(
            tokenizer,
            cfg.patching.get("high_score_tokens", ["6", "7"]),
            cfg.patching.get("low_score_tokens", ["1", "2", "3", "4", "5"]),
        )
        print(f"Loaded HF multi-GPU model: {model_cfg.name}")
    else:
        dtype = get_dtype(model_cfg.dtype)
        model, tokenizer = load_hooked_transformer(
            model_cfg.name, device=device, dtype=dtype,
            hf_token=cfg.dataset.get("hf_token"),
        )
        n_layers = model.cfg.n_layers  # authoritative from TL
        metric_fn = build_score_metric(
            tokenizer,
            cfg.patching.get("high_score_tokens", ["6", "7"]),
            cfg.patching.get("low_score_tokens", ["1", "2", "3", "4", "5"]),
        )
        print(f"Loaded TransformerLens model: {n_layers} layers")

    all_resid = np.zeros((len(pairs), n_layers))
    all_attn  = np.zeros((len(pairs), n_layers))
    all_mlp   = np.zeros((len(pairs), n_layers))
    pair_meta = []

    for pair_idx, (ai_idx, human_idx) in enumerate(pairs):
        print(f"\rPair {pair_idx+1}/{len(pairs)}", end="", flush=True)
        ai_row    = ds[ai_idx]
        human_row = ds[human_idx]

        ai_prompt    = build_judge_prompt(ai_row["question"],    ai_row["response"],
                                          tokenizer=tokenizer, format_instructions=fmt_inst)
        human_prompt = build_judge_prompt(human_row["question"], human_row["response"],
                                          tokenizer=tokenizer, format_instructions=fmt_inst)

        ai_ids    = tokenizer(ai_prompt,    return_tensors="pt",
                              max_length=cfg.extraction.max_seq_len, truncation=True
                              ).input_ids.to(device)
        human_ids = tokenizer(human_prompt, return_tensors="pt",
                              max_length=cfg.extraction.max_seq_len, truncation=True
                              ).input_ids.to(device)

        if use_hf_backend:
            result = run_activation_patching_hf(
                model, ai_ids, human_ids, metric_fn, n_layers, device
            )
        else:
            result = run_activation_patching(
                model, ai_ids, human_ids, metric_fn, n_layers=n_layers, device=device
            )

        all_resid[pair_idx] = result["resid_post"]
        all_attn[pair_idx]  = result["attn_out"]
        all_mlp[pair_idx]   = result["mlp_out"]
        pair_meta.append({
            "ai_idx": ai_idx, "human_idx": human_idx,
            "ai_source": ai_row["response_source"],
            "clean_metric": result["clean_metric"],
            "corrupted_metric": result["corrupted_metric"],
            "delta": result["delta"],
        })

    print()

    # Save
    np.save(patch_dir / f"{model_key}_resid_patching.npy", all_resid)
    np.save(patch_dir / f"{model_key}_attn_patching.npy",  all_attn)
    np.save(patch_dir / f"{model_key}_mlp_patching.npy",   all_mlp)

    with open(patch_dir / f"{model_key}_patching_meta.json", "w") as f:
        json.dump({"model_key": model_key, "n_pairs": len(pairs),
                   "use_hf_backend": use_hf_backend, "pairs": pair_meta}, f, indent=2)

    mean_resid = all_resid.mean(axis=0)
    mean_attn  = all_attn.mean(axis=0)
    mean_mlp   = all_mlp.mean(axis=0)

    peak_layer = int(np.argmax(mean_resid))
    print(f"\n=== Results ===")
    print(f"Peak bias layer (resid_post): {peak_layer} (effect={mean_resid[peak_layer]:.3f})")
    if not use_hf_backend:
        print(f"Peak bias layer (attn_out):   {int(np.argmax(mean_attn))} (effect={mean_attn.max():.3f})")
        print(f"Peak bias layer (mlp_out):    {int(np.argmax(mean_mlp))} (effect={mean_mlp.max():.3f})")
    else:
        print("  (attn/mlp decomposition not available in HF backend mode)")

    combined = np.column_stack([mean_resid, mean_attn, mean_mlp])
    plot_patching_heatmap(
        combined,
        title=f"Activation Patching Effects — {model_key}" + (" [hf-backend, resid only]" if use_hf_backend else ""),
        output_path=str(plot_dir / f"{model_key}_patching_heatmap.png"),
        layer_labels=[str(i) for i in range(n_layers)],
    )

    summary = {
        "model_key": model_key, "n_pairs": len(pairs),
        "use_hf_backend": use_hf_backend,
        "peak_layer_resid": peak_layer,
        "peak_effect_resid": float(mean_resid[peak_layer]),
        "mean_resid_by_layer": mean_resid.tolist(),
        "mean_attn_by_layer":  mean_attn.tolist(),
        "mean_mlp_by_layer":   mean_mlp.tolist(),
    }
    with open(patch_dir / f"{model_key}_patching_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPlot and summary saved to {patch_dir}")
    free_model_memory(model)
    print("=== Activation patching complete ===")
    exit_cleanly()


if __name__ == "__main__":
    main()
