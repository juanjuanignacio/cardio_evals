#!/usr/bin/env python3
"""
Block 4 — Step 2: Compute and apply steering vectors.

Computes the mean(h_AI) - mean(h_human) direction at the peak probe layer
and applies it during inference to reduce self-preference bias.

Output:
  results/steering/{model_key}_steering_vector_L{layer}.npy
  results/steering/{model_key}_steering_results_alpha{alpha}.json

Usage:
    python blocks/block4_interventions/02_steering_vectors.py
    python blocks/block4_interventions/02_steering_vectors.py --alpha 2.0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, set_global_seed
from src.data import build_judge_prompt, create_splits, get_unique_responses, load_cardio_dataset
from src.model import free_model_memory, get_dtype, load_hooked_transformer, exit_cleanly
from src.probing import get_peak_probe_layer, load_probe_results
from src.steering import compute_steering_vector, measure_score_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model_key", default=None)
    parser.add_argument("--position", default="score_token")
    parser.add_argument("--layer", type=int, default=None,
                        help="Force a specific layer instead of using probe peak")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Single alpha to test (or tests all from config)")
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--device", default=None,
                        help="Override device (e.g. cuda:1)")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_global_seed(cfg.seed)

    model_key = args.model_key or cfg.primary_model.key
    probe_dir = str(ROOT / cfg.paths["probe_dir"])
    steer_dir = ROOT / cfg.paths["steer_dir"]
    data_dir = str(ROOT / cfg.paths["data_dir"])
    steer_dir.mkdir(parents=True, exist_ok=True)

    h5_path = f"{data_dir}/{model_key}.h5"
    device = args.device or cfg.primary_model.device
    fmt_inst = cfg.format_instructions
    alphas = [args.alpha] if args.alpha else cfg.steering["alpha_values"]

    print(f"=== Block 4.2: Steering Vectors ===")
    if any((steer_dir / f"{model_key}_steering_results_L{l}.json").exists()
           for l in range(100)):
        print(f"[{model_key}] Steering vectors already computed. Skipping.")
        return
    print(f"Model: {model_key} | Position: {args.position}")

    # Find peak probe layer (or use forced layer)
    if args.layer is not None:
        peak_layer = args.layer
        print(f"Peak probe layer: {peak_layer} (forced via --layer)")
    else:
        try:
            probe_results = load_probe_results(probe_dir, model_key, "authorship")
            peak_layer = get_peak_probe_layer(probe_results, args.position, "authorship", "auc")
            print(f"Peak probe layer: {peak_layer}")
        except FileNotFoundError:
            print("[WARNING] Probe results not found. Using layer 16 as default.")
            peak_layer = 16

    # Compute steering vector
    print(f"\nComputing steering vector at layer {peak_layer}...")
    steer_vec = compute_steering_vector(
        h5_path=h5_path,
        position=args.position,
        layer=peak_layer,
        split="train",
        ai_sources=tuple(cfg.dataset["ai_sources"]),
        human_sources=tuple(cfg.dataset["human_sources"]),
    )

    # Save vector
    vec_path = steer_dir / f"{model_key}_steering_vector_L{peak_layer}.npy"
    np.save(str(vec_path), steer_vec)
    print(f"Steering vector saved: {vec_path}")
    print(f"Vector norm: {np.linalg.norm(steer_vec):.4f} (should be ~1.0)")

    # Load dataset and build test prompts — use unique responses (same set as extraction)
    raw_ds = load_cardio_dataset(cfg)
    ds = get_unique_responses(
        raw_ds,
        target_evaluator=cfg.dataset.get("target_evaluator", "Llama3.1:8b"),
        filter_uninformative=True,
    )
    train_ds, val_ds, _ = create_splits(ds, train_ratio=cfg.dataset["train_ratio"],
                                         val_ratio=cfg.dataset["val_ratio"], seed=cfg.seed)
    train_set = set(zip(train_ds["question"], train_ds["response"]))
    val_set = set(zip(val_ds["question"], val_ds["response"]))

    ai_sources = set(cfg.dataset["ai_sources"])
    human_sources = set(cfg.dataset["human_sources"])

    dtype = get_dtype(cfg.primary_model.dtype)
    model, tokenizer = load_hooked_transformer(
        cfg.primary_model.name, device=device, dtype=dtype,
        hf_token=cfg.dataset.get("hf_token"),
    )

    ai_prompts, human_prompts, ai_gt, human_gt = [], [], [], []
    for i in range(len(ds)):
        k = (ds[i]["question"], ds[i]["response"])
        split = "train" if k in train_set else ("val" if k in val_set else "test")
        if split != "test":
            continue
        src = ds[i]["response_source"]
        prompt = build_judge_prompt(ds[i]["question"], ds[i]["response"],
                                    tokenizer=tokenizer, format_instructions=fmt_inst)
        if src in ai_sources and len(ai_prompts) < args.n_test:
            ai_prompts.append(prompt)
            ai_gt.append(int(ds[i]["accuracy_score"]))
        elif src in human_sources and len(human_prompts) < args.n_test:
            human_prompts.append(prompt)
            human_gt.append(int(ds[i]["accuracy_score"]))

    print(f"\nTest prompts: {len(ai_prompts)} AI, {len(human_prompts)} Human")

    high_tokens = cfg.patching.get("high_score_tokens", ["6", "7"])
    low_tokens = cfg.patching.get("low_score_tokens", ["1", "2", "3", "4", "5"])

    all_results = []

    # Baseline (alpha=0)
    print("\nRunning baseline (no steering)...")
    ai_base = measure_score_distribution(model, tokenizer, ai_prompts, None,
                                          peak_layer, 0, device=device,
                                          high_score_tokens=high_tokens,
                                          low_score_tokens=low_tokens)
    human_base = measure_score_distribution(model, tokenizer, human_prompts, None,
                                             peak_layer, 0, device=device,
                                             high_score_tokens=high_tokens,
                                             low_score_tokens=low_tokens)
    baseline_bias = float(np.mean(ai_base["score_metric"]) - np.mean(human_base["score_metric"]))
    print(f"Baseline bias: {baseline_bias:.4f} "
          f"(AI: {np.mean(ai_base['argmax_score']):.2f}, "
          f"Human: {np.mean(human_base['argmax_score']):.2f})")

    all_results.append({
        "alpha": 0.0,
        "layer": peak_layer,
        "bias_metric": baseline_bias,
        "ai_mean_score": float(np.mean(ai_base["argmax_score"])),
        "human_mean_score": float(np.mean(human_base["argmax_score"])),
        "ai_high_lp": float(np.mean(ai_base["high_logprob"])),
        "human_high_lp": float(np.mean(human_base["high_logprob"])),
    })

    # Test each alpha
    for alpha in alphas:
        print(f"\nAlpha = {alpha}...")
        ai_res = measure_score_distribution(
            model, tokenizer, ai_prompts, steer_vec, peak_layer, alpha,
            direction="subtract", device=device,
            high_score_tokens=high_tokens, low_score_tokens=low_tokens,
        )
        human_res = measure_score_distribution(
            model, tokenizer, human_prompts, steer_vec, peak_layer, alpha,
            direction="subtract", device=device,
            high_score_tokens=high_tokens, low_score_tokens=low_tokens,
        )
        bias = float(np.mean(ai_res["score_metric"]) - np.mean(human_res["score_metric"]))
        bias_reduction = (baseline_bias - bias) / abs(baseline_bias) * 100 if baseline_bias != 0 else 0

        print(f"  Bias: {bias:.4f} (reduction: {bias_reduction:.1f}%)")
        print(f"  AI mean score: {np.mean(ai_res['argmax_score']):.2f}, "
              f"Human: {np.mean(human_res['argmax_score']):.2f}")

        all_results.append({
            "alpha": alpha,
            "layer": peak_layer,
            "bias_metric": bias,
            "bias_reduction_pct": bias_reduction,
            "ai_mean_score": float(np.mean(ai_res["argmax_score"])),
            "human_mean_score": float(np.mean(human_res["argmax_score"])),
            "ai_high_lp": float(np.mean(ai_res["high_logprob"])),
            "human_high_lp": float(np.mean(human_res["high_logprob"])),
        })

    with open(steer_dir / f"{model_key}_steering_results_L{peak_layer}.json", "w") as f:
        json.dump({"model_key": model_key, "peak_layer": peak_layer,
                   "position": args.position, "baseline_bias": baseline_bias,
                   "results": all_results}, f, indent=2)

    print(f"\nResults saved to {steer_dir}")
    free_model_memory(model)
    print("=== Steering vectors complete ===")
    exit_cleanly()


if __name__ == "__main__":
    main()
