"""Bias metrics, statistical tests, and evaluation utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def compute_bias_score(
    ai_scores: np.ndarray,
    human_scores: np.ndarray,
) -> Dict:
    """
    Primary bias metric: Cohen's d between AI and Human score distributions.
    Also returns mean difference, t-statistic, p-value.
    """
    ai_scores = np.asarray(ai_scores, dtype=np.float64)
    human_scores = np.asarray(human_scores, dtype=np.float64)

    mean_diff = float(np.mean(ai_scores) - np.mean(human_scores))

    pooled_std = np.sqrt(
        (np.var(ai_scores, ddof=1) * (len(ai_scores) - 1) +
         np.var(human_scores, ddof=1) * (len(human_scores) - 1)) /
        (len(ai_scores) + len(human_scores) - 2)
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 1e-8 else 0.0

    t_stat, p_val = stats.ttest_ind(ai_scores, human_scores, equal_var=False)

    if abs(cohens_d) < 0.2:
        effect_label = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_label = "small"
    elif abs(cohens_d) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"

    return {
        "mean_ai": float(np.mean(ai_scores)),
        "mean_human": float(np.mean(human_scores)),
        "mean_diff": mean_diff,
        "std_ai": float(np.std(ai_scores, ddof=1)),
        "std_human": float(np.std(human_scores, ddof=1)),
        "cohens_d": float(cohens_d),
        "effect_label": effect_label,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "n_ai": len(ai_scores),
        "n_human": len(human_scores),
    }


def compute_accuracy_bias(
    ds,
    ai_sources: Tuple[str, ...] = ("AI", "CoT AI"),
    human_sources: Tuple[str, ...] = ("Human",),
    evaluator_filter: Optional[str] = None,
) -> Dict:
    """
    Primary bias metric: Cohen's d on accuracy_score between AI and Human groups.
    Pass evaluator_filter (e.g. "Llama3.1:8b") to restrict to a single evaluator's rows.
    Always filter to the target evaluator before calling this — mixing evaluators
    dilutes the signal (human evaluators show the opposite bias direction).
    """
    _cols = ds.column_names if hasattr(ds, "column_names") else list(ds.keys())
    evaluators = np.array(ds["evaluator"] if "evaluator" in _cols else [""] * len(ds["response_source"]))
    sources = np.array(ds["response_source"])
    scores = np.array(ds["accuracy_score"], dtype=np.float64)

    eval_mask = (evaluators == evaluator_filter) if evaluator_filter else np.ones(len(sources), bool)
    ai_mask = np.isin(sources, list(ai_sources)) & eval_mask
    human_mask = np.isin(sources, list(human_sources)) & eval_mask
    return compute_bias_score(scores[ai_mask], scores[human_mask])


def compute_three_way_bias(ds, evaluator_filter: Optional[str] = None) -> Dict[str, Dict]:
    """
    Compute accuracy_score bias for each pairwise comparison across the three
    respondent groups: AI, CoT AI, Human.

    Pass evaluator_filter (e.g. "Llama3.1:8b") to restrict to one evaluator's rows.
    Human evaluators show reverse bias — always filter unless comparing evaluators.

    Returns keys: "AI_vs_Human", "CoT_AI_vs_Human", "AI_vs_CoT_AI".
    """
    _cols = ds.column_names if hasattr(ds, "column_names") else list(ds.keys())
    evaluators = np.array(ds["evaluator"] if "evaluator" in _cols else [""] * len(ds["response_source"]))
    sources = np.array(ds["response_source"])
    scores = np.array(ds["accuracy_score"], dtype=np.float64)

    eval_mask = (evaluators == evaluator_filter) if evaluator_filter else np.ones(len(sources), bool)

    groups = {
        "AI":     scores[(sources == "AI")     & eval_mask],
        "CoT_AI": scores[(sources == "CoT AI") & eval_mask],
        "Human":  scores[(sources == "Human")  & eval_mask],
    }

    results = {}
    for a, b in [("AI", "Human"), ("CoT_AI", "Human"), ("AI", "CoT_AI")]:
        key = f"{a}_vs_{b}"
        results[key] = (
            compute_bias_score(groups[a], groups[b])
            if len(groups[a]) > 1 and len(groups[b]) > 1
            else {"error": "insufficient samples"}
        )
    return results


def compute_evaluator_comparison(ds) -> Dict[str, Dict]:
    """
    Compute three-way accuracy bias for every evaluator in the dataset.
    Useful for showing that Llama3.1:8b has the highest self-preference bias
    while Human evaluators show the opposite direction.
    """
    evaluators = np.unique(ds["evaluator"])
    return {ev: compute_three_way_bias(ds, evaluator_filter=ev) for ev in evaluators}


def compute_bias_by_dimension(
    ds,
    ai_sources: Tuple[str, ...] = ("AI", "CoT AI"),
    human_sources: Tuple[str, ...] = ("Human",),
) -> Dict[str, Dict]:
    """
    Compute bias score for each score dimension (accuracy, clarity, completeness).
    accuracy_score is the primary dimension; clarity and completeness are secondary.
    """
    sources = np.array(ds["response_source"])
    ai_mask = np.isin(sources, list(ai_sources))
    human_mask = np.isin(sources, list(human_sources))

    results = {}
    for dim in ["accuracy_score", "clarity_score", "completeness_score"]:
        scores = np.array(ds[dim], dtype=np.float64)
        results[dim] = compute_bias_score(scores[ai_mask], scores[human_mask])
    results["_primary"] = "accuracy_score"
    return results


def compute_source_prediction_metrics(
    predicted: List[str],
    actual: List[str],
    ai_label: str = "ai",
    human_label: str = "human",
) -> Dict:
    """
    Accuracy, precision, recall, F1 for source prediction.
    Labels are normalized to lowercase.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    pred_arr = np.array([str(p).lower().strip() for p in predicted])
    actual_arr = np.array([str(a).lower().strip() for a in actual])

    # Binarize: ai=1, human=0
    def binarize(arr):
        out = np.full(len(arr), -1, dtype=np.int8)
        out[np.isin(arr, [ai_label, "ai-generated", "1"])] = 1
        out[np.isin(arr, [human_label, "human-written", "0"])] = 0
        return out

    pred_b = binarize(pred_arr)
    actual_b = binarize(actual_arr)

    valid = (pred_b >= 0) & (actual_b >= 0)
    pred_b = pred_b[valid]
    actual_b = actual_b[valid]

    if len(pred_b) == 0:
        return {"error": "No valid predictions"}

    acc = float(np.mean(pred_b == actual_b))
    tp = float(np.sum((pred_b == 1) & (actual_b == 1)))
    fp = float(np.sum((pred_b == 1) & (actual_b == 0)))
    fn = float(np.sum((pred_b == 0) & (actual_b == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_valid": len(pred_b),
        "n_ai_predicted": int(np.sum(pred_b == 1)),
        "n_human_predicted": int(np.sum(pred_b == 0)),
    }


def compute_score_agreement(
    predicted_scores: np.ndarray,
    actual_scores: np.ndarray,
) -> Dict:
    """
    Agreement metrics between predicted and actual scores.
    Returns Pearson r, Spearman rho, exact match rate, within-1 match rate.
    """
    from scipy.stats import pearsonr, spearmanr

    predicted = np.asarray(predicted_scores, dtype=np.float64)
    actual = np.asarray(actual_scores, dtype=np.float64)

    valid = ~(np.isnan(predicted) | np.isnan(actual))
    predicted = predicted[valid]
    actual = actual[valid]

    if len(predicted) < 2:
        return {"error": "Insufficient data"}

    pearson_r, pearson_p = pearsonr(predicted, actual)
    spearman_r, spearman_p = spearmanr(predicted, actual)
    exact_match = float(np.mean(predicted == actual))
    within_1 = float(np.mean(np.abs(predicted - actual) <= 1))
    mae = float(np.mean(np.abs(predicted - actual)))

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "exact_match": exact_match,
        "within_1_match": within_1,
        "mae": mae,
        "n": len(predicted),
    }


def score_token_logprobs(
    logits: "torch.Tensor",
    tokenizer,
    position: int = -1,
) -> Dict[str, float]:
    """
    Given model logits, return log-probabilities for each score digit (0-7).
    logits shape: (batch, seq, vocab). Uses position for extraction.
    Returns dict {"0": logprob, ..., "7": logprob}.
    """
    import torch

    last_logits = logits[:, position, :]
    log_probs = torch.log_softmax(last_logits, dim=-1)

    result = {}
    for digit in range(8):
        tok_ids = tokenizer.encode(str(digit), add_special_tokens=False)
        if tok_ids:
            result[str(digit)] = float(log_probs[0, tok_ids[0]].cpu())
    return result


def compute_nondeterminism_stats(
    scores_list: List[List[int]],
) -> Dict:
    """
    Given N runs of scores for the same prompts, compute variance statistics.
    scores_list: list of N arrays, each shape (n_samples,).
    Returns mean variance, CV, and flip rate.
    """
    arr = np.array(scores_list, dtype=np.float64)  # (N_runs, n_samples)
    variances = np.var(arr, axis=0, ddof=1)  # (n_samples,)
    means = np.mean(arr, axis=0)
    cv = np.where(means > 0, np.std(arr, axis=0, ddof=1) / means, 0.0)

    # Flip rate: fraction of samples where score changes ≥2 between any two runs
    flipped = np.any(np.abs(arr - arr[0:1, :]) >= 2, axis=0)

    return {
        "mean_variance": float(np.mean(variances)),
        "median_variance": float(np.median(variances)),
        "mean_cv": float(np.mean(cv)),
        "flip_rate": float(np.mean(flipped)),
        "n_runs": len(scores_list),
        "n_samples": arr.shape[1],
    }
