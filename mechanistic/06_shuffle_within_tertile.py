"""
Block 8.8 — Shuffle within tertile experiment (Llama 3.1 8B).

Same idea as the original shuffle experiment in 03_geometry_all_probes.py, but
responses are shuffled **within each length tertile** (T1/T2/T3).  This ensures
the shuffled pair always has the same length tier as the original response, so
any residual score difference between AI and Human cannot be explained by length.

Tertile boundaries: computed from word counts on the filtered unique-response set
(same as the notebook uses).

Output:
  results/tables/{model_key}_shuffled_within_tertile.json
  results/plots/{model_key}_shuffled_within_tertile.png

Usage:
  python blocks/block8_probes/08_shuffle_within_tertile.py
  python blocks/block8_probes/08_shuffle_within_tertile.py --force
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import build_judge_prompt, get_unique_responses, load_cardio_dataset
from src.model import free_model_memory, get_dtype, load_hooked_transformer


# ── Helpers (copied from 03_geometry_all_probes.py) ──────────────────────────

def _score_prompts_forward(model, tokenizer, prompts, digit_ids,
                            device, batch_size=8, max_seq_len=2048,
                            digit_vals=None):
    if digit_vals is None:
        digit_vals = list(range(len(digit_ids)))
    digit_vals_arr = np.array(digit_vals, dtype=float)
    digit_ids_t    = torch.tensor(digit_ids, device=device)
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_seq_len).to(device)
        with torch.no_grad():
            out = model(enc["input_ids"])
        last_logits = out[:, -1, :]
        digit_p = torch.log_softmax(last_logits, dim=-1)[:, digit_ids_t].exp()
        for b in range(digit_p.shape[0]):
            p = digit_p[b].cpu().float().numpy()
            p = p / p.sum()
            results.append({
                "expected_score": round(float(np.dot(digit_vals_arr, p)), 4),
                "most_likely":    int(digit_vals_arr[int(np.argmax(p))]),
            })
        print(f"\r  scored {min(i + batch_size, len(prompts))}/{len(prompts)}",
              end="", flush=True)
    print()
    return results


def _bias_stats(per_sample, sources):
    groups = {}
    for rec, src in zip(per_sample, sources):
        groups.setdefault(src, []).append(rec["expected_score"])
    stats = {}
    for src, vals in groups.items():
        stats[src] = {"mean": round(float(np.mean(vals)), 4),
                      "std":  round(float(np.std(vals)),  4),
                      "n":    len(vals)}
    ai_vals    = groups.get("AI", []) + groups.get("CoT AI", [])
    human_vals = groups.get("Human", [])
    if ai_vals and human_vals:
        stats["ai_minus_human"] = round(
            float(np.mean(ai_vals) - np.mean(human_vals)), 4)
    return stats


# ── Stratified (within-tertile) derangement ──────────────────────────────────

def _derangement(idxs: list, rng: np.random.Generator) -> list:
    """Permute list so no element stays in its original position."""
    arr      = np.array(idxs)
    shuffled = rng.permutation(arr)
    for k in range(len(arr)):
        if shuffled[k] == arr[k]:
            swap = (k + 1) % len(arr)
            shuffled[k], shuffled[swap] = shuffled[swap], shuffled[k]
    return shuffled.tolist()


def _build_trials(sources, questions, responses, rng, ai_sources, human_sources,
                  tertile_labels):
    """
    For each group × tertile cell, shuffle response indices within the cell
    (derangement). Returns list of (question_idx, response_idx, group_label).
    """
    group_map = {}
    for i, src in enumerate(sources):
        if src in ai_sources:
            g = "AI"
        elif src in human_sources:
            g = "Human"
        else:
            continue
        t = tertile_labels[i]
        group_map.setdefault((g, t), []).append(i)

    trials = []
    for (grp, tert), idxs in sorted(group_map.items()):
        if len(idxs) < 2:
            # Can't derange a single element — skip (shouldn't happen)
            continue
        resp_idxs = _derangement(idxs, rng)
        for q_i, r_i in zip(idxs, resp_idxs):
            trials.append((q_i, r_i, grp, tert))

    return trials


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--model_key", default=None,
                        help="Override model key (default: primary model from config)")
    parser.add_argument("--device",    default=None)
    parser.add_argument("--force",     action="store_true",
                        help="Re-run even if output already exists")
    args = parser.parse_args()

    cfg       = load_config(ROOT / args.config)
    model_cfg = cfg.primary_model
    if args.model_key and args.model_key != model_cfg.key:
        # Allow overriding to a comparison model
        for m in cfg.comparison_models:
            if m.key == args.model_key:
                model_cfg = m
                break
        else:
            raise ValueError(f"Unknown model_key: {args.model_key}")

    model_key = model_cfg.key
    table_dir = ROOT / cfg.paths["table_dir"]
    plot_dir  = ROOT / cfg.paths["plot_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    out_json = table_dir / f"{model_key}_shuffled_within_tertile.json"

    if out_json.exists() and not args.force:
        with open(out_json) as f:
            saved = json.load(f)
        if saved.get("complete"):
            print(f"[{model_key}] Already complete: {out_json.name}  (use --force to rerun)")
            return

    # ── Dataset ──────────────────────────────────────────────────────────────
    raw_ds = load_cardio_dataset(cfg)
    ds = get_unique_responses(
        raw_ds,
        target_evaluator=cfg.dataset.get("target_evaluator", "Llama3.1:8b"),
        filter_uninformative=True,
    )
    n         = len(ds)
    sources   = [ds[i]["response_source"] for i in range(n)]
    questions = [ds[i]["question"]         for i in range(n)]
    responses = [ds[i]["response"]         for i in range(n)]

    # ── Tertile boundaries (word count) ──────────────────────────────────────
    word_counts = np.array([len(r.split()) for r in responses])
    q33 = float(np.percentile(word_counts, 33.33))
    q66 = float(np.percentile(word_counts, 66.67))

    def _assign(wc):
        if wc <= q33: return "T1"
        if wc <= q66: return "T2"
        return "T3"

    tertile_labels = [_assign(wc) for wc in word_counts]

    tert_counts = {}
    for t in tertile_labels:
        tert_counts[t] = tert_counts.get(t, 0) + 1
    print(f"[{model_key}] Tertile boundaries: T1≤{q33:.0f}w, T2≤{q66:.0f}w, T3>{q66:.0f}w")
    print(f"  Counts: {tert_counts}")

    # ── Build trials (stratified derangement) ────────────────────────────────
    rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else 42)

    ai_sources_set    = set(cfg.dataset["ai_sources"])
    human_sources_set = set(cfg.dataset["human_sources"])

    trials = _build_trials(sources, questions, responses, rng,
                           ai_sources_set, human_sources_set, tertile_labels)

    print(f"[{model_key}] Trials: {len(trials)}")
    for grp in ["AI", "Human"]:
        for tert in ["T1", "T2", "T3"]:
            cnt = sum(1 for t in trials if t[2] == grp and t[3] == tert)
            print(f"  {grp} × {tert}: {cnt}")

    # ── Load model ────────────────────────────────────────────────────────────
    device = args.device or (model_cfg.device if hasattr(model_cfg, "device") else "cuda:0")
    dtype  = get_dtype(model_cfg.dtype if hasattr(model_cfg, "dtype") else "bfloat16")
    use_hf = getattr(model_cfg, "use_hf_backend", False)

    if use_hf:
        from src.model import load_hf_model_large
        model, tokenizer = load_hf_model_large(
            model_cfg.name, device=device, dtype=dtype,
            hf_token=cfg.dataset.get("hf_token"),
        )
    else:
        model, tokenizer = load_hooked_transformer(
            model_cfg.name, device=device, dtype=dtype,
            hf_token=cfg.dataset.get("hf_token"),
        )

    fmt_inst  = getattr(cfg, "format_instructions", "")
    _DIGITS   = list(range(1, 8))
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in _DIGITS]

    prompts = [
        build_judge_prompt(questions[q_i], responses[r_i],
                           tokenizer=tokenizer, format_instructions=fmt_inst)
        for q_i, r_i, _, _ in trials
    ]
    trial_sources = [g for _, _, g, _ in trials]
    trial_terts   = [t for _, _, _, t in trials]

    # ── Build matched (original) prompts for the same questions ──────────────
    matched_prompts = [
        build_judge_prompt(questions[q_i], responses[q_i],
                           tokenizer=tokenizer, format_instructions=fmt_inst)
        for q_i, _, _, _ in trials
    ]

    # ── Score shuffled ────────────────────────────────────────────────────────
    print(f"[{model_key}] Scoring {len(prompts)} shuffled-within-tertile pairs...")
    per_sample_shuf = _score_prompts_forward(
        model, tokenizer, prompts, digit_ids,
        device=device, batch_size=cfg.extraction.batch_size,
        max_seq_len=cfg.extraction.max_seq_len,
        digit_vals=_DIGITS,
    )

    # ── Score matched (original) ──────────────────────────────────────────────
    print(f"[{model_key}] Scoring {len(matched_prompts)} matched (original) pairs...")
    per_sample_orig = _score_prompts_forward(
        model, tokenizer, matched_prompts, digit_ids,
        device=device, batch_size=cfg.extraction.batch_size,
        max_seq_len=cfg.extraction.max_seq_len,
        digit_vals=_DIGITS,
    )
    free_model_memory(model)

    # ── Assemble records ──────────────────────────────────────────────────────
    records = []
    for (q_i, r_i, grp, tert), shuf_rec, orig_rec in zip(trials, per_sample_shuf, per_sample_orig):
        records.append({
            "group":                  grp,
            "tertile":                tert,
            "question_idx":           int(q_i),
            "response_idx":           int(r_i),
            "response_source":        sources[r_i],
            "resp_word_count":        int(word_counts[r_i]),
            # shuffled pair scores
            "expected_score":         shuf_rec["expected_score"],
            "most_likely":            shuf_rec["most_likely"],
            # original matched pair scores
            "orig_expected_score":    orig_rec["expected_score"],
            "orig_most_likely":       orig_rec["most_likely"],
            # per-sample delta
            "delta_expected_score":   round(orig_rec["expected_score"] - shuf_rec["expected_score"], 4),
            "delta_most_likely":      orig_rec["most_likely"] - shuf_rec["most_likely"],
        })

    bias = _bias_stats(per_sample_shuf, trial_sources)
    print(f"[{model_key}] AI − Human (within-tertile shuffle): "
          f"{bias.get('ai_minus_human', 'N/A')}")

    output = {
        "model_key": model_key,
        "complete":  True,
        "tertile_boundaries": {"q33": q33, "q66": q66},
        "description": (
            "Responses shuffled within each length tertile × group cell. "
            "Length confound is removed: shuffled pair always has same length tier. "
            "Each record contains both the shuffled score and the original matched score "
            "so that per-sample delta = orig - shuf can be computed directly."
        ),
        "bias_stats": bias,
        "records":    records,
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[{model_key}] Saved: {out_json}")


if __name__ == "__main__":
    main()
