#!/usr/bin/env python3
"""
Block 1 — Step 1: Extract hidden states for all cardio_evaluations samples.

Saves to results/hidden_states/{model_key}.h5
Includes checkpointing: safe to interrupt and resume with --resume.

Usage:
    python blocks/block1_representations/01_extract_hidden_states.py
    python blocks/block1_representations/01_extract_hidden_states.py --dry_run
    python blocks/block1_representations/01_extract_hidden_states.py --resume
    python blocks/block1_representations/01_extract_hidden_states.py --calibrate
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.config import load_config, set_global_seed, get_model_config
from src.data import (
    build_judge_prompt,
    create_splits,
    get_binary_labels,
    get_target_evaluator_scores,
    get_unique_responses,
    load_cardio_dataset,
)
from src.extraction import extract_hidden_states, extract_hidden_states_hf
from src.model import free_model_memory, get_dtype, load_hooked_transformer, exit_cleanly


def build_score_lookup(target_ds) -> Dict[Tuple[str, str], Dict]:
    """
    Build a (question, response) → scores dict from the target evaluator's rows.
    Used to label unique responses with scores from the correct evaluator (Llama3.1:8b),
    not an average across all evaluators including humans.
    """
    lookup = {}
    for i in range(len(target_ds)):
        row = target_ds[i]
        key = (row["question"], row["response"])
        lookup[key] = {
            "accuracy_score": int(row["accuracy_score"]),
            "clarity_score": int(row["clarity_score"]),
            "completeness_score": int(row["completeness_score"]),
            "source_predicted": row.get("source_predicted", ""),
        }
    return lookup


def build_metadata(ds, splits_map, score_lookup):
    """
    Build list of metadata dicts parallel to dataset rows.
    Scores come from score_lookup (target evaluator only).
    Rows with no score from the target evaluator get score=-1 (excluded from probes).
    """
    metadata = []
    for i in range(len(ds)):
        row = ds[i]
        key = (row["question"], row["response"])
        scores = score_lookup.get(key, {})
        metadata.append({
            "response_source": row["response_source"],
            "source_predicted": scores.get("source_predicted", row.get("source_predicted", "")),
            "medical_specialty": row.get("medical_specialty", ""),
            "accuracy_score": scores.get("accuracy_score", -1),
            "clarity_score": scores.get("clarity_score", -1),
            "completeness_score": scores.get("completeness_score", -1),
            "has_target_score": int(key in score_lookup),
            "split": splits_map.get(i, "train"),
            "question": row["question"],
            "response": row["response"],
        })
    return metadata


def calibrate(model, tokenizer, ds, cfg, n_samples=50, device="cuda:0",
              own_score_lookup: Dict[Tuple[str, str], int] = None,
              evaluator_label: str = "Llama3.1:8b"):
    """
    Run all samples in ds through the model (teacher-forced) and:
      1. Compare predicted scores against the model's OWN scores from the dataset
         (own_score_lookup, keyed by (question, response)). Falls back to the
         Llama3.1:8b scores already in ds rows when own_score_lookup is None.
      2. Show predicted score distribution by response_source (AI / CoT AI / Human).
      3. Report AI−Human self-preference bias in teacher-forced predictions.

    Uses logits[:, -1, :] argmax over digit token IDs — no generation needed.
    """
    fmt_inst = cfg.format_instructions
    ai_sources = set(cfg.dataset["ai_sources"])

    # Pre-compute digit token IDs for scores 0-7
    digit_ids = []
    for d in range(8):
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        digit_ids.append(toks[0])

    from collections import defaultdict
    pred_by_source = defaultdict(list)   # source → [predicted scores]
    exact_matches = 0
    within_1 = 0
    n_scored = 0
    n_no_ref = 0  # samples with no reference score for this evaluator

    ref_label = evaluator_label if own_score_lookup is not None else "Llama3.1:8b (fallback)"
    print(f"\n=== Calibration: {len(ds)} samples (teacher forcing) ===")
    print(f"  Reference scores from: {ref_label}")

    for i in range(len(ds)):
        row = ds[i]
        src = row["response_source"]

        # Reference score: prefer model's own score, fall back to row score
        if own_score_lookup is not None:
            key = (row["question"], row["response"])
            true_acc = own_score_lookup.get(key, -1)
        else:
            true_acc = int(row["accuracy_score"])

        prompt = build_judge_prompt(row["question"], row["response"],
                                    tokenizer=tokenizer, format_instructions=fmt_inst)
        tokens = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = tokens["input_ids"].to(device)

        with torch.no_grad():
            if hasattr(model, "run_with_cache"):
                # TransformerLens HookedTransformer
                logits = model(input_ids, return_type="logits")
            else:
                # HuggingFace model (device_map="auto" — first device handles embedding)
                logits = model(input_ids).logits

        last_logits = logits[0, -1, :]
        digit_logits = torch.stack([last_logits[tid] for tid in digit_ids])
        pred_acc = int(digit_logits.argmax().item())

        pred_by_source[src].append(pred_acc)

        if true_acc >= 0 and i < n_samples:
            n_scored += 1
            if pred_acc == true_acc:
                exact_matches += 1
            if abs(pred_acc - true_acc) <= 1:
                within_1 += 1
        elif true_acc < 0 and i < n_samples:
            n_no_ref += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}] processed...")

    # --- Match-rate report ---
    print(f"\n--- Score match vs {ref_label} (first {n_samples} samples) ---")
    if n_scored > 0:
        exact_rate   = exact_matches / n_scored * 100
        within_1_rate = within_1    / n_scored * 100
        print(f"  Comparable samples:   {n_scored}  (no ref score: {n_no_ref})")
        print(f"  Exact match:  {exact_matches}/{n_scored} ({exact_rate:.1f}%)")
        print(f"  Within ±1:   {within_1}/{n_scored} ({within_1_rate:.1f}%)")
        if within_1_rate < 40:
            print("  [NOTE] Low match — this model scores differently from the reference "
                  "evaluator (expected if model != reference).")
        elif within_1_rate >= 70:
            print("  [OK] High agreement with reference scores (≥70% within ±1).")
        else:
            print("  [INFO] Moderate agreement.")
    else:
        print(f"  No reference scores available for {ref_label} on these samples.")

    # --- Predicted score distribution by source ---
    print(f"\n--- Predicted accuracy by response_source (all {len(ds)} samples) ---")
    print(f"  {'Source':<12}  {'Mean':>6}  {'Std':>6}  {'Min':>4}  {'Max':>4}  n")
    ai_preds = []
    for src in sorted(pred_by_source):
        preds = np.array(pred_by_source[src])
        print(f"  {src:<12}  {preds.mean():>6.3f}  {preds.std():>6.3f}  "
              f"{preds.min():>4}  {preds.max():>4}  {len(preds)}")
        if src in ai_sources:
            ai_preds.extend(preds.tolist())
    human_preds = pred_by_source.get("Human", [])
    if ai_preds and human_preds:
        bias = float(np.mean(ai_preds)) - float(np.mean(human_preds))
        print(f"\n  AI (all) mean:   {np.mean(ai_preds):.3f}")
        print(f"  Human mean:      {np.mean(human_preds):.3f}")
        print(f"  Bias (AI−Human): {bias:+.3f}")
        if bias > 0.3:
            print("  [OK] Self-preference bias confirmed in teacher-forced predictions.")
        elif bias > 0:
            print("  [INFO] Small positive bias.")
        else:
            print("  [NOTE] No AI self-preference in teacher-forced scores.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model_key", default=None,
                        help="Model key to extract (default: primary model from config)")
    parser.add_argument("--device", default=None,
                        help="Override device, e.g. cuda:0 or cuda:1")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted extraction")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration only")
    parser.add_argument("--dry_run", action="store_true", help="Process only first 20 samples")
    parser.add_argument("--n_calibrate", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    model_key = args.model_key or cfg.primary_model.key
    model_cfg = get_model_config(cfg, model_key)

    output_path = ROOT / cfg.paths["data_dir"] / f"{model_key}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Early-exit BEFORE set_global_seed — set_global_seed calls torch.cuda.is_available()
    # which opens the CUDA context. If we exit after that, PyTorch destroys the context
    # on shutdown. With CUDA_VISIBLE_DEVICES pointing to a non-existent GPU this causes
    # a kernel panic and node reboot.
    if not args.calibrate and not args.dry_run and output_path.exists():
        import h5py
        with h5py.File(str(output_path), "r") as f:
            progress = int(f.get("extraction_progress", [-1])[()] ) + 1
            n_samples = int(f.attrs.get("n_samples", 0))
        if progress >= n_samples and n_samples > 0:
            print(f"=== Block 1.1: Hidden State Extraction ===")
            print(f"Extraction already complete ({n_samples} samples). Skipping.")
            return   # exits without ever touching CUDA

    # CUDA context opened here for the first time — only reached if work is needed
    set_global_seed(cfg.seed)

    device = args.device or model_cfg.device

    print(f"=== Block 1.1: Hidden State Extraction ===")
    print(f"Model: {model_cfg.name}  [{model_key}]")
    print(f"Output: {output_path}")
    print(f"Device: {device}")

    # Load dataset: 490 responses evaluated by Llama3.1:8b, minus IDK responses.
    # This is both the extraction set AND the score label source — same dataset.
    print("\nLoading dataset...")
    raw_ds = load_cardio_dataset(cfg)
    target_evaluator = cfg.dataset.get("target_evaluator", "Llama3.1:8b")
    ds = get_unique_responses(raw_ds, target_evaluator=target_evaluator, filter_uninformative=True)

    from collections import Counter
    src_counts = Counter(ds["response_source"])
    print(f"Responses for extraction ({target_evaluator}, IDK filtered): {len(ds)} — " +
          ", ".join(f"{k}={v}" for k, v in sorted(src_counts.items())))

    # Scores come directly from ds (already from Llama3.1:8b, no mixing needed)
    score_lookup = build_score_lookup(ds)

    # Create splits on unique responses
    train_ds, val_ds, test_ds = create_splits(
        ds,
        train_ratio=cfg.dataset["train_ratio"],
        val_ratio=cfg.dataset["val_ratio"],
        seed=cfg.seed,
    )

    train_set = set(zip(train_ds["question"], train_ds["response"]))
    val_set = set(zip(val_ds["question"], val_ds["response"]))
    splits_map = {}
    for i in range(len(ds)):
        key = (ds[i]["question"], ds[i]["response"])
        if key in train_set:
            splits_map[i] = "train"
        elif key in val_set:
            splits_map[i] = "val"
        else:
            splits_map[i] = "test"

    print(f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Load model — TransformerLens for small models, HF device_map="auto" for 70B+
    print(f"\nLoading model: {model_cfg.name}")
    dtype = get_dtype(model_cfg.dtype)
    use_hf = getattr(model_cfg, "use_hf_backend", False)

    if use_hf:
        from src.model import load_hf_model_large
        model, tokenizer = load_hf_model_large(
            model_cfg.name,
            dtype=dtype,
            hf_token=cfg.dataset.get("hf_token"),
        )
        n_layers = model_cfg.n_layers
        d_model  = model_cfg.d_model
        print(f"Model loaded (HF backend): {n_layers} layers, d_model={d_model}")
    else:
        model, tokenizer = load_hooked_transformer(
            model_cfg.name,
            device=device,
            dtype=dtype,
            hf_token=cfg.dataset.get("hf_token"),
        )
        n_layers = model.cfg.n_layers
        d_model  = model.cfg.d_model
        print(f"Model loaded (TL backend): {n_layers} layers, d_model={d_model}")

    # Build own-evaluator score lookup for calibration.
    # For the primary model (Llama3.1:8b) this matches the ds scores exactly.
    # For comparison models the lookup maps (question, response) → their own accuracy score.
    evaluator_label = getattr(model_cfg, "evaluator_label", None)
    own_score_lookup = None
    if evaluator_label:
        from src.data import filter_by_evaluator
        own_rows = filter_by_evaluator(raw_ds, evaluator_label)
        own_score_lookup = {}
        for i in range(len(own_rows)):
            r = own_rows[i]
            key = (r["question"], r["response"])
            own_score_lookup[key] = int(r["accuracy_score"])
        print(f"  Own-evaluator scores loaded: {len(own_score_lookup)} rows ({evaluator_label})")

    # Calibration: runs for both TL and HF backends.
    # NOTE: R1 models prepend <think>...</think> before the JSON, so teacher-forced
    # logits at the GENERATION_PREFIX position don't land on the score digit.
    # The calibration still runs to show predicted score distribution, but the
    # match rate will be low for R1 models — that is expected, not a bug.
    # For HF multi-GPU models, use the first parameter's device as inference device.
    calib_device = device
    if use_hf:
        try:
            calib_device = str(next(iter(model.parameters())).device)
        except Exception:
            calib_device = "cuda:0"
    calibrate(model, tokenizer, ds, cfg, n_samples=args.n_calibrate, device=calib_device,
              own_score_lookup=own_score_lookup,
              evaluator_label=evaluator_label or "Llama3.1:8b")
    if args.calibrate:
        return

    # Build prompts and metadata (unique responses; scores from target evaluator)
    print("\nBuilding prompts...")
    fmt_inst = cfg.format_instructions
    if args.dry_run:
        print("[DRY RUN] Processing first 20 samples only")
        n = 20
    else:
        n = len(ds)

    prompts = [
        build_judge_prompt(ds[i]["question"], ds[i]["response"],
                           tokenizer=tokenizer, format_instructions=fmt_inst)
        for i in range(n)
    ]
    metadata = build_metadata(ds, splits_map, score_lookup)[:n]

    # Extract
    print(f"\nExtracting hidden states for {n} samples...")
    if use_hf:
        extract_hidden_states_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            sample_metadata=metadata,
            output_path=str(output_path),
            positions=cfg.extraction.positions,
            n_layers=n_layers,
            d_model=d_model,
            model_name=model_cfg.name,
            batch_size=max(1, cfg.extraction.batch_size // 2),  # 70B needs smaller batches
            checkpoint_every=cfg.extraction.checkpoint_every,
            max_seq_len=cfg.extraction.max_seq_len,
        )
    else:
        extract_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            sample_metadata=metadata,
            output_path=str(output_path),
            positions=cfg.extraction.positions,
            batch_size=cfg.extraction.batch_size,
            device=device,
            checkpoint_every=cfg.extraction.checkpoint_every,
            max_seq_len=cfg.extraction.max_seq_len,
        )

    print("\n=== Extraction complete ===")
    print(f"File: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1e9:.2f} GB")

    # Synchronize BEFORE cleanup: drain all pending CUDA ops so no async work fires
    # during or after model teardown.  empty_cache() is intentionally omitted —
    # the process exits immediately after; the OS reclaims all GPU memory on exit,
    # and empty_cache() causes driver-level instability on shared H100s after long jobs.
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    del tokenizer
    free_model_memory(model)
    exit_cleanly()


if __name__ == "__main__":
    main()
