"""
Block 8.9 — Cross-tertile shuffle experiment (Llama 3.1 8B).

Pairs questions from one length tertile with responses from a DIFFERENT tertile
to isolate the effect of response length on the judge score, independent of
question-response content match.

Conditions (q_tertile → r_tertile):
  T1 → T3  : short questions  paired with long responses
  T3 → T1  : long questions   paired with short responses
  T2 → T1  : medium questions paired with short responses
  T2 → T3  : medium questions paired with long responses

For each cross-tertile pair, the original matched pair (question_idx == response_idx)
is also scored so that per-sample delta = orig - cross can be computed and
a paired Wilcoxon test can be run per condition.

question_idx is preserved in every record to enable paired tests.

Output:
  results/tables/{model_key}_cross_tertile_shuffle.json

Usage:
  python blocks/block8_probes/09_cross_tertile_shuffle.py
  python blocks/block8_probes/09_cross_tertile_shuffle.py --force
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


# ── Scoring helper ────────────────────────────────────────────────────────────

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


# ── Cross-tertile trial builder ───────────────────────────────────────────────

CROSS_CONDITIONS = [
    ("T1", "T3"),   # short questions  × long responses
    ("T3", "T1"),   # long questions   × short responses
    ("T2", "T1"),   # medium questions × short responses
    ("T2", "T3"),   # medium questions × long responses
]


def _build_cross_trials(sources, rng, ai_sources, human_sources,
                        tertile_labels):
    """
    Group ALL AI+Human samples by their RESPONSE tertile (not by group).
    For each (q_tert, r_tert) condition, randomly pair questions from q_tert
    with responses from r_tert. n ≈ min(|q_tert|, |r_tert|) ≈ 147.
    Tertile assignment is always based on response word count.
    Returns list of (question_idx, response_idx, q_group, r_group, q_tert, r_tert).
    """
    # Build tertile → [sample indices] across ALL AI+Human samples
    tert_map = {}
    for i, src in enumerate(sources):
        if src not in ai_sources and src not in human_sources:
            continue  # skip CoT AI
        t = tertile_labels[i]   # always response-based tertile
        tert_map.setdefault(t, []).append(i)

    trials = []
    for q_tert, r_tert in CROSS_CONDITIONS:
        q_idxs = tert_map.get(q_tert, [])
        r_idxs = tert_map.get(r_tert, [])
        if not q_idxs or not r_idxs:
            continue
        n = min(len(q_idxs), len(r_idxs))
        q_sample = rng.permutation(q_idxs)[:n].tolist()
        r_sample = rng.permutation(r_idxs)[:n].tolist()
        for q_i, r_i in zip(q_sample, r_sample):
            q_grp = "AI" if sources[q_i] in ai_sources else "Human"
            r_grp = "AI" if sources[r_i] in ai_sources else "Human"
            trials.append((q_i, r_i, q_grp, r_grp, q_tert, r_tert))

    return trials


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--model_key", default=None)
    parser.add_argument("--device",    default=None)
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args()

    cfg       = load_config(ROOT / args.config)
    model_cfg = cfg.primary_model
    if args.model_key and args.model_key != model_cfg.key:
        for m in cfg.comparison_models:
            if m.key == args.model_key:
                model_cfg = m
                break
        else:
            raise ValueError(f"Unknown model_key: {args.model_key}")

    model_key = model_cfg.key
    table_dir = ROOT / cfg.paths["table_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    out_json  = table_dir / f"{model_key}_cross_tertile_shuffle.json"

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

    # ── Tertile boundaries ────────────────────────────────────────────────────
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

    # ── Build cross-tertile trials ────────────────────────────────────────────
    rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else 42)
    ai_sources_set    = set(cfg.dataset["ai_sources"])
    human_sources_set = set(cfg.dataset["human_sources"])

    trials = _build_cross_trials(sources, rng, ai_sources_set,
                                  human_sources_set, tertile_labels)

    print(f"[{model_key}] Total trials: {len(trials)}")
    for q_t, r_t in CROSS_CONDITIONS:
        cnt = sum(1 for t in trials if t[4] == q_t and t[5] == r_t)
        print(f"  Q={q_t} × R={r_t}: {cnt}")

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

    # ── Build prompts ─────────────────────────────────────────────────────────
    cross_prompts = [
        build_judge_prompt(questions[q_i], responses[r_i],
                           tokenizer=tokenizer, format_instructions=fmt_inst)
        for q_i, r_i, _, _, _, _ in trials
    ]
    matched_prompts = [
        build_judge_prompt(questions[q_i], responses[q_i],
                           tokenizer=tokenizer, format_instructions=fmt_inst)
        for q_i, _, _, _, _, _ in trials
    ]

    # ── Score cross-tertile pairs ─────────────────────────────────────────────
    print(f"[{model_key}] Scoring {len(cross_prompts)} cross-tertile pairs...")
    per_cross = _score_prompts_forward(
        model, tokenizer, cross_prompts, digit_ids,
        device=device, batch_size=cfg.extraction.batch_size,
        max_seq_len=cfg.extraction.max_seq_len,
        digit_vals=_DIGITS,
    )

    # ── Score original matched pairs ──────────────────────────────────────────
    print(f"[{model_key}] Scoring {len(matched_prompts)} original matched pairs...")
    per_orig = _score_prompts_forward(
        model, tokenizer, matched_prompts, digit_ids,
        device=device, batch_size=cfg.extraction.batch_size,
        max_seq_len=cfg.extraction.max_seq_len,
        digit_vals=_DIGITS,
    )
    free_model_memory(model)

    # ── Assemble records ──────────────────────────────────────────────────────
    records = []
    for (q_i, r_i, q_grp, r_grp, q_tert, r_tert), cross_rec, orig_rec in \
            zip(trials, per_cross, per_orig):
        records.append({
            "q_group":              q_grp,
            "r_group":              r_grp,
            "q_tertile":            q_tert,
            "r_tertile":            r_tert,
            "condition":            f"{q_tert}→{r_tert}",
            "question_idx":         int(q_i),
            "response_idx":         int(r_i),
            "q_word_count":         int(word_counts[q_i]),
            "r_word_count":         int(word_counts[r_i]),
            "cross_expected_score": cross_rec["expected_score"],
            "cross_most_likely":    cross_rec["most_likely"],
            "orig_expected_score":  orig_rec["expected_score"],
            "orig_most_likely":     orig_rec["most_likely"],
            "delta_expected_score": round(orig_rec["expected_score"] - cross_rec["expected_score"], 4),
            "delta_most_likely":    orig_rec["most_likely"] - cross_rec["most_likely"],
        })

    # ── Summary stats per condition ───────────────────────────────────────────
    cond_stats = {}
    for cond in [f"{q}→{r}" for q, r in CROSS_CONDITIONS]:
        sub = [rec for rec in records if rec["condition"] == cond]
        if sub:
            deltas = [rec["delta_expected_score"] for rec in sub]
            cond_stats[cond] = {
                "n":           len(sub),
                "mean_delta":  round(float(np.mean(deltas)), 4),
                "std_delta":   round(float(np.std(deltas)),  4),
            }
            print(f"  {cond}: n={len(sub)}, mean_delta={cond_stats[cond]['mean_delta']:.3f}")

    output = {
        "model_key":          model_key,
        "complete":           True,
        "tertile_boundaries": {"q33": q33, "q66": q66},
        "conditions":         [f"{q}→{r}" for q, r in CROSS_CONDITIONS],
        "description": (
            "Cross-tertile shuffle: questions from one tertile paired with "
            "responses from a different tertile. question_idx is preserved "
            "for paired Wilcoxon tests (orig vs cross-tertile score)."
        ),
        "condition_stats":    cond_stats,
        "records":            records,
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[{model_key}] Saved: {out_json}")


if __name__ == "__main__":
    main()
