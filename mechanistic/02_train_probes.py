#!/usr/bin/env python3
"""
Block 1 — Step 2: Train linear probes at every layer for each probe type.

Saves ProbeResult objects to results/probes/
Also saves a summary JSON and the fitted probe objects.

Usage:
    python blocks/block1_representations/02_train_probes.py
    python blocks/block1_representations/02_train_probes.py --probe_type authorship
    python blocks/block1_representations/02_train_probes.py --position score_token
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import load_config, set_global_seed
from src.data import load_cardio_dataset, get_unique_responses, build_judge_prompt
from src.probing import (
    get_peak_probe_layer,
    load_probe_results,
    save_probe_results,
    train_all_probes_kfold,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--probe_type", default="all",
                        help="Probe type to train (or 'all')")
    parser.add_argument("--position", default="all",
                        help="Token position to use (or 'all')")
    parser.add_argument("--model_key", default=None,
                        help="Model key (defaults to primary model)")
    parser.add_argument("--device", default=None, help="Ignored (CPU-only step)")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_global_seed(cfg.seed)

    model_key = args.model_key or cfg.primary_model.key
    h5_path = str(ROOT / cfg.paths["data_dir"] / f"{model_key}.h5")
    probe_dir = str(ROOT / cfg.paths["probe_dir"])
    Path(probe_dir).mkdir(parents=True, exist_ok=True)

    probe_types = cfg.probing.probe_types if args.probe_type == "all" else [args.probe_type]
    positions = cfg.extraction.positions if args.position == "all" else [args.position]

    print(f"=== Block 1.2: Probe Training ===")
    print(f"Model: {model_key}")
    print(f"HDF5: {h5_path}")

    if not Path(h5_path).exists():
        print(f"[SKIP] HDF5 not found — extraction must have failed for {model_key}. Skipping.")
        return

    print(f"Probe types: {probe_types}")
    print(f"Positions: {positions}")

    summary = {}

    for probe_type in probe_types:
        print(f"\n--- Probe type: {probe_type} ---")
        results = train_all_probes_kfold(
            h5_path=h5_path,
            model_key=model_key,
            probe_type=probe_type,
            positions=positions,
            cfg=cfg,
            n_splits=5,
            output_dir=probe_dir,
            suffix="",
        )
        summary[probe_type] = {}

        for position in positions:
            pos_results = [r for r in results if r.position == position]
            if not pos_results:
                continue
            try:
                peak_layer = get_peak_probe_layer(results, position, probe_type, "auc")
                peak_result = next(r for r in results
                                   if r.position == position
                                   and r.probe_type == probe_type
                                   and r.layer == peak_layer)
                print(f"  {position}: peak AUC={peak_result.auc:.3f} "
                      f"(CI: [{peak_result.auc_ci[0]:.3f}, {peak_result.auc_ci[1]:.3f}]) "
                      f"at layer {peak_layer}")
                summary[probe_type][position] = {
                    "peak_layer": peak_layer,
                    "peak_auc": peak_result.auc,
                    "auc_ci": list(peak_result.auc_ci),
                    "peak_accuracy": peak_result.accuracy,
                }
            except (ValueError, StopIteration):
                print(f"  {position}: no results")

    # Save summary JSON
    summary_path = Path(probe_dir) / f"{model_key}_probe_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Verification
    print("\n=== Verification ===")
    try:
        auth_results = [r for r in load_probe_results(probe_dir, model_key, "authorship")
                        if r.position == "score_token"]
        max_auc = max(r.auc for r in auth_results) if auth_results else 0.0
        if max_auc < 0.55:
            print(f"[WARNING] Max authorship AUC={max_auc:.3f} < 0.55 — probe not learning.")
            print("  Check: (1) HDF5 file is complete, (2) prompt reconstruction is correct,")
            print("  (3) dataset has both AI and Human responses in train split.")
        else:
            print(f"[OK] Authorship probe max AUC={max_auc:.3f} (>0.55)")
    except FileNotFoundError:
        print("[INFO] Authorship results not found for verification")

    # Verbosity check — reviewer confound: are AI responses systematically longer?
    _print_verbosity_stats(cfg, ROOT)


def _print_verbosity_stats(cfg, root):
    """Print response length by source group to flag potential verbosity confound."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            cfg.primary_model.name, token=cfg.dataset.get("hf_token"),
        )
        raw_ds = load_cardio_dataset(cfg)
        ds = get_unique_responses(
            raw_ds,
            target_evaluator=cfg.dataset.get("target_evaluator", "Llama3.1:8b"),
            filter_uninformative=True,
        )
        lengths_by_src: dict = {}
        for i in range(len(ds)):
            row = ds[i]
            src = row["response_source"]
            n_toks = len(tok(row["response"], add_special_tokens=False).input_ids)
            lengths_by_src.setdefault(src, []).append(n_toks)

        print("\n=== Verbosity check (response length by source) ===")
        for src in ["AI", "CoT AI", "Human"]:
            if src not in lengths_by_src:
                continue
            l = np.array(lengths_by_src[src])
            print(f"  {src:8s}: n={len(l):3d}  mean={l.mean():.0f}  median={np.median(l):.0f}  "
                  f"max={l.max()}")
        ai = np.array(lengths_by_src.get("AI", []))
        cot = np.array(lengths_by_src.get("CoT AI", []))
        human = np.array(lengths_by_src.get("Human", []))
        if len(ai) and len(human):
            ratio = ai.mean() / human.mean()
            print(f"  AI/Human length ratio: {ratio:.2f}"
                  + (" ← potential verbosity confound" if ratio > 1.2 else " (OK)"))
        if len(cot) and len(human):
            ratio_cot = cot.mean() / human.mean()
            print(f"  CoT/Human length ratio: {ratio_cot:.2f}"
                  + (" ← potential verbosity confound" if ratio_cot > 1.2 else " (OK)"))
    except Exception as e:
        print(f"[INFO] Verbosity check skipped: {e}")


if __name__ == "__main__":
    main()
