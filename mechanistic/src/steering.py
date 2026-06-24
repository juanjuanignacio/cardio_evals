"""Steering vectors, head ablation, and bias-quality tradeoff experiments."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.extraction import compute_mean_diff_vector


# ---------------------------------------------------------------------------
# Steering vector computation
# ---------------------------------------------------------------------------

def compute_steering_vector(
    h5_path: str,
    position: str,
    layer: int,
    split: str = "train",
    ai_sources: Tuple[str, ...] = ("AI", "CoT AI"),
    human_sources: Tuple[str, ...] = ("Human",),
) -> np.ndarray:
    """
    Compute mean(h_AI) - mean(h_human) at (position, layer).
    Returns unit-normalized direction, shape (d_model,).
    """
    return compute_mean_diff_vector(h5_path, position, layer, split, ai_sources, human_sources)


# ---------------------------------------------------------------------------
# Steering application
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_steering(
    model,
    tokenizer,
    prompts: List[str],
    steering_vector: np.ndarray,
    layer: int,
    alpha: float,
    direction: str = "subtract",
    batch_size: int = 4,
    device: str = "cuda:0",
    max_new_tokens: int = 150,
) -> Dict[str, List]:
    """
    Apply subspace mean-ablation during forward pass via TransformerLens hook.

    For each activation h, removes alpha * its component along the bias direction v:
        h_new = h - alpha * (h · v) * v
    alpha=1.0 → full projection (removes the authorship subspace entirely).
    alpha=0.0 → no change (baseline).
    alpha in (0,1) → partial removal.

    Returns dict with:
    - "generated_texts": raw model outputs
    """
    steer_tensor = torch.tensor(steering_vector, dtype=torch.float32, device=device)
    hook_name = f"blocks.{layer}.hook_resid_post"

    def steer_hook(value, hook):
        v = steer_tensor.to(value.dtype)
        proj = torch.einsum("bsd,d->bs", value, v)  # (batch, seq)
        return value - alpha * proj.unsqueeze(-1) * v

    results = {"generated_texts": [], "raw_outputs": []}

    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start:batch_start + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=900,
        ).to(device)

        output = model.generate(
            tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            hooks=[(hook_name, steer_hook)],
        )
        for i in range(len(batch)):
            new_tokens = output[i][tokens["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results["generated_texts"].append(text)

    return results


@torch.no_grad()
def measure_score_distribution(
    model,
    tokenizer,
    prompts: List[str],
    steering_vector: Optional[np.ndarray],
    layer: int,
    alpha: float,
    direction: str = "subtract",
    batch_size: int = 4,
    device: str = "cuda:0",
    max_seq_len: int = 1024,
    high_score_tokens: List[str] = ("6", "7"),
    low_score_tokens: List[str] = ("1", "2", "3", "4", "5"),
) -> Dict:
    """
    For each prompt, compute log-probabilities of score tokens at the Accuracy position.
    Optionally applies subspace mean-ablation: h_new = h - alpha * (h · v) * v.

    Returns dict with:
    - "high_logprob": (n_samples,) mean logprob of high-score tokens
    - "low_logprob": (n_samples,) mean logprob of low-score tokens
    - "score_metric": (n_samples,) high - low logprob (our bias measure)
    - "argmax_score": (n_samples,) most likely score digit (0-7)
    """
    all_score_digits = [str(i) for i in range(8)]
    digit_ids = [tokenizer.encode(d, add_special_tokens=False)[0] for d in all_score_digits]
    high_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in high_score_tokens]
    low_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in low_score_tokens]

    if steering_vector is not None and alpha > 0:
        steer_tensor = torch.tensor(steering_vector, dtype=torch.float32, device=device)
        hook_name = f"blocks.{layer}.hook_resid_post"

        def steer_hook(value, hook):
            v = steer_tensor.to(value.dtype)
            proj = torch.einsum("bsd,d->bs", value, v)  # (batch, seq)
            return value - alpha * proj.unsqueeze(-1) * v
        hooks = [(hook_name, steer_hook)]
    else:
        hooks = []

    high_lps, low_lps, argmax_scores = [], [], []

    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start:batch_start + batch_size]
        tokenizer.padding_side = "left"
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = tokens["input_ids"].to(device)

        if hooks:
            logits = model.run_with_hooks(input_ids, fwd_hooks=hooks, return_type="logits")
        else:
            logits = model(input_ids, return_type="logits")

        # Score token position: last token (model generates Accuracy digit next)
        score_logits = logits[:, -1, :]  # (batch, vocab)
        log_probs = torch.log_softmax(score_logits, dim=-1)

        for b in range(len(batch)):
            lp = log_probs[b].cpu().float().numpy()
            high_lps.append(float(np.mean([lp[i] for i in high_ids])))
            low_lps.append(float(np.mean([lp[i] for i in low_ids])))
            argmax_scores.append(int(np.argmax([lp[i] for i in digit_ids])))

        del logits

    high_arr = np.array(high_lps)
    low_arr = np.array(low_lps)
    return {
        "high_logprob": high_arr,
        "low_logprob": low_arr,
        "score_metric": high_arr - low_arr,
        "argmax_score": np.array(argmax_scores),
    }


def run_bias_quality_tradeoff(
    model,
    tokenizer,
    ai_prompts: List[str],
    human_prompts: List[str],
    ai_ground_truth: np.ndarray,
    human_ground_truth: np.ndarray,
    steering_vector: np.ndarray,
    layer: int,
    alpha_values: List[float],
    device: str = "cuda:0",
) -> pd.DataFrame:
    """
    For each alpha value, apply subspace mean-ablation (h_new = h - alpha*(h·v)*v) and measure:
    1. bias_delta: AI_mean_score - Human_mean_score (target: 0)
    2. quality_corr_ai: Spearman(predicted, ground_truth) for AI responses
    3. quality_corr_human: same for Human responses

    alpha=0: baseline. alpha=1: full subspace ablation. alpha in (0,1): partial.
    Returns DataFrame with columns [alpha, bias_delta, quality_corr_ai, quality_corr_human].
    """
    from scipy.stats import spearmanr

    rows = []

    all_prompts = ai_prompts + human_prompts
    all_gt = np.concatenate([ai_ground_truth, human_ground_truth])
    n_ai = len(ai_prompts)

    for alpha in tqdm(alpha_values, desc="Steering alphas"):
        steer = steering_vector if alpha > 0 else None
        result = measure_score_distribution(
            model, tokenizer, all_prompts, steer, layer, alpha,
            direction="subtract", device=device,
        )
        pred_scores = result["argmax_score"]
        ai_pred = pred_scores[:n_ai]
        human_pred = pred_scores[n_ai:]

        bias_delta = float(np.mean(ai_pred) - np.mean(human_pred))
        corr_ai, _ = spearmanr(ai_pred, ai_ground_truth)
        corr_human, _ = spearmanr(human_pred, human_ground_truth)

        rows.append({
            "alpha": alpha,
            "bias_delta": bias_delta,
            "ai_mean_score": float(np.mean(ai_pred)),
            "human_mean_score": float(np.mean(human_pred)),
            "quality_corr_ai": float(corr_ai) if not np.isnan(corr_ai) else 0.0,
            "quality_corr_human": float(corr_human) if not np.isnan(corr_human) else 0.0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Head ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def ablate_head(
    model,
    tokens: torch.Tensor,
    layer: int,
    head: int,
    metric_fn,
    ablation_type: str = "mean",
    mean_activation: Optional[torch.Tensor] = None,
    device: str = "cuda:0",
) -> float:
    """
    Ablate a specific attention head by zeroing or replacing its output.
    Returns metric value after ablation.

    Uses hook 'blocks.{layer}.attn.hook_result' shape (batch, seq, n_heads, d_head).
    """
    tokens = tokens.to(device)
    hook_name = f"blocks.{layer}.attn.hook_result"

    if ablation_type == "zero":
        def hook_fn(value, hook):
            out = value.clone()
            out[:, :, head, :] = 0.0
            return out
    elif ablation_type == "mean" and mean_activation is not None:
        def hook_fn(value, hook):
            out = value.clone()
            out[:, :, head, :] = mean_activation[head].unsqueeze(0).unsqueeze(0)
            return out
    else:
        def hook_fn(value, hook):
            out = value.clone()
            out[:, :, head, :] = 0.0
            return out

    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)], return_type="logits")
    return metric_fn(logits).item()


def compute_mean_head_activations(
    model,
    tokenizer,
    prompts: List[str],
    n_layers: int,
    n_heads: int,
    batch_size: int = 8,
    device: str = "cuda:0",
    max_seq_len: int = 1024,
) -> Optional[torch.Tensor]:
    """
    Compute mean head activation per (layer, head) over all prompts.
    Returns tensor shape (n_layers, n_heads, d_head).
    Used for mean-ablation: replace head output with its mean activation.

    Requires model.cfg.use_attn_result = True (set here automatically).
    """
    model.cfg.use_attn_result = True

    # Accumulate running sums per layer to avoid holding all batches in memory
    layer_sums: Dict[int, torch.Tensor] = {}
    layer_counts: Dict[int, int] = {}

    with torch.no_grad():
        for batch_start in range(0, len(prompts), batch_size):
            batch = prompts[batch_start:batch_start + batch_size]
            tokenizer.padding_side = "left"
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_seq_len,
            ).to(device)

            _, cache = model.run_with_cache(
                tokens["input_ids"],
                names_filter=lambda n: "attn.hook_result" in n,
            )
            for layer in range(n_layers):
                key = f"blocks.{layer}.attn.hook_result"
                if key in cache:
                    # Mean over batch and seq dims: (n_heads, d_head)
                    batch_mean = cache[key].mean(dim=(0, 1)).cpu()
                    if layer not in layer_sums:
                        layer_sums[layer] = batch_mean
                        layer_counts[layer] = 1
                    else:
                        layer_sums[layer] = layer_sums[layer] + batch_mean
                        layer_counts[layer] += 1
            del cache

    if not layer_sums:
        return None

    # Build (n_layers, n_heads, d_head) tensor
    first = next(iter(layer_sums.values()))
    result = torch.zeros(n_layers, first.shape[0], first.shape[1])
    for layer, total in layer_sums.items():
        result[layer] = total / layer_counts[layer]
    return result


@torch.no_grad()
def run_head_ablation_study(
    model,
    tokenizer,
    ai_prompts: List[str],
    human_prompts: List[str],
    heads_to_ablate: List[Tuple[int, int]],
    metric_fn,
    ablation_type: str = "mean",
    mean_activations: Optional[torch.Tensor] = None,
    device: str = "cuda:0",
    max_seq_len: int = 1024,
) -> pd.DataFrame:
    """
    Systematically ablate each head and measure bias change.

    For each head (layer, head):
    - Run all AI prompts through model with that head ablated
    - Run all Human prompts through model with that head ablated
    - Compute AI_mean_metric - Human_mean_metric (bias measure)

    Returns DataFrame: [layer, head, ai_metric, human_metric, bias_delta, bias_reduction_pct]
    """
    # Baseline (no ablation)
    baseline_ai = _batch_metric(model, tokenizer, ai_prompts, metric_fn, device, max_seq_len)
    baseline_human = _batch_metric(model, tokenizer, human_prompts, metric_fn, device, max_seq_len)
    baseline_bias = float(np.mean(baseline_ai) - np.mean(baseline_human))

    rows = []
    rows.append({
        "layer": -1, "head": -1, "label": "baseline",
        "ai_metric": float(np.mean(baseline_ai)),
        "human_metric": float(np.mean(baseline_human)),
        "bias_delta": baseline_bias,
        "bias_reduction_pct": 0.0,
    })

    for layer, head in tqdm(heads_to_ablate, desc="Ablating heads"):
        hook_name = f"blocks.{layer}.attn.hook_result"
        ma = None
        if ablation_type == "mean" and mean_activations is not None:
            ma = mean_activations

        ai_metrics = _batch_metric_with_ablation(
            model, tokenizer, ai_prompts, metric_fn, layer, head,
            ablation_type, ma, device, max_seq_len,
        )
        human_metrics = _batch_metric_with_ablation(
            model, tokenizer, human_prompts, metric_fn, layer, head,
            ablation_type, ma, device, max_seq_len,
        )

        bias = float(np.mean(ai_metrics) - np.mean(human_metrics))
        reduction = (baseline_bias - bias) / abs(baseline_bias) * 100 if baseline_bias != 0 else 0.0

        rows.append({
            "layer": layer, "head": head,
            "label": f"L{layer}H{head}",
            "ai_metric": float(np.mean(ai_metrics)),
            "human_metric": float(np.mean(human_metrics)),
            "bias_delta": bias,
            "bias_reduction_pct": reduction,
        })

    return pd.DataFrame(rows)


def _batch_metric(model, tokenizer, prompts, metric_fn, device, max_seq_len, batch_size=4):
    results = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            tokenizer.padding_side = "left"
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_seq_len,
            ).to(device)
            logits = model(tokens["input_ids"], return_type="logits")
            results.append(metric_fn(logits).item())
    return results


def _batch_metric_with_ablation(
    model, tokenizer, prompts, metric_fn, layer, head,
    ablation_type, mean_activation, device, max_seq_len, batch_size=4
):
    model.cfg.use_attn_result = True
    hook_name = f"blocks.{layer}.attn.hook_result"

    if ablation_type == "zero" or mean_activation is None:
        def hook_fn(value, hook):
            out = value.clone(); out[:, :, head, :] = 0.0; return out
    else:
        # mean_activation is (n_layers, n_heads, d_head) — index by specific layer+head
        mean_val = mean_activation[layer, head]
        def hook_fn(value, hook):
            out = value.clone()
            out[:, :, head, :] = mean_val.to(value.device)
            return out

    results = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            tokenizer.padding_side = "left"
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_seq_len,
            ).to(device)
            logits = model.run_with_hooks(
                tokens["input_ids"],
                fwd_hooks=[(hook_name, hook_fn)],
                return_type="logits",
            )
            results.append(metric_fn(logits).item())
    return results


def batch_metric_with_multi_ablation(
    model, tokenizer, prompts, metric_fn,
    heads_to_ablate: List[Tuple[int, int]],
    ablation_type: str = "zero",
    mean_activations=None,
    device: str = "cuda:0",
    max_seq_len: int = 1024,
    batch_size: int = 4,
) -> List[float]:
    """Ablate multiple (layer, head) pairs simultaneously and measure metric."""
    model.cfg.use_attn_result = True
    from collections import defaultdict
    heads_by_layer = defaultdict(list)
    for layer, head in heads_to_ablate:
        heads_by_layer[layer].append(head)

    def make_layer_hook(layer_idx, layer_heads, ma):
        def hook_fn(value, hook):
            out = value.clone()
            for h in layer_heads:
                if ablation_type == "zero" or ma is None:
                    out[:, :, h, :] = 0.0
                else:
                    # ma is (n_layers, n_heads, d_head) — use layer-specific mean
                    out[:, :, h, :] = ma[layer_idx, h].to(value.device)
            return out
        return hook_fn

    fwd_hooks = [
        (f"blocks.{layer}.attn.hook_result",
         make_layer_hook(layer, heads, mean_activations))
        for layer, heads in heads_by_layer.items()
    ]

    results = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            tokenizer.padding_side = "left"
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_seq_len,
            ).to(device)
            logits = model.run_with_hooks(
                tokens["input_ids"],
                fwd_hooks=fwd_hooks,
                return_type="logits",
            )
            results.append(metric_fn(logits).item())
    return results
