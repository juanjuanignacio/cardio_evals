"""Activation patching, head attribution, and residual stream decomposition.

Two backends:
- TransformerLens (HookedTransformer): primary model (Llama 3.1 8B).
  Uses run_with_cache + run_with_hooks — full mechanistic interp API.
- HuggingFace 4-bit (AutoModelForCausalLM): 70B comparison models.
  Uses PyTorch forward hooks on model.model.layers[i].
  Patches the residual stream (layer output[0]) and per-head outputs
  (o_proj pre-hook) — semantically equivalent to TL's hook_resid_post
  and attn.hook_result respectively.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Score metric
# ---------------------------------------------------------------------------

def build_score_metric(
    tokenizer,
    high_score_strings: List[str] = ("6", "7"),
    low_score_strings: List[str] = ("1", "2", "3", "4", "5"),
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a differentiable metric function for activation patching.

    metric(logits) = mean log P(high score digits) - mean log P(low score digits)
    at the last token position (where the model predicts the score digit).

    logits shape: (batch, seq, vocab)
    Returns scalar tensor (mean over batch).
    """
    high_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in high_score_strings]
    low_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in low_score_strings]

    def metric(logits: torch.Tensor) -> torch.Tensor:
        # Use last token logits (model generates score next)
        last_logits = logits[:, -1, :]  # (batch, vocab)
        log_probs = torch.log_softmax(last_logits, dim=-1)
        high_lp = log_probs[:, high_ids].mean(dim=-1)  # (batch,)
        low_lp = log_probs[:, low_ids].mean(dim=-1)    # (batch,)
        return (high_lp - low_lp).mean()

    return metric


def build_source_metric(tokenizer) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Metric for source prediction: log P("ai") - log P("human") at last token.
    """
    ai_ids = tokenizer.encode('"ai"', add_special_tokens=False)[:1]
    human_ids = tokenizer.encode('"human"', add_special_tokens=False)[:1]

    ai_id = ai_ids[0] if ai_ids else tokenizer.encode("ai", add_special_tokens=False)[0]
    human_id = human_ids[0] if human_ids else tokenizer.encode("human", add_special_tokens=False)[0]

    def metric(logits: torch.Tensor) -> torch.Tensor:
        last_logits = logits[:, -1, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)
        return (log_probs[:, ai_id] - log_probs[:, human_id]).mean()

    return metric


# ---------------------------------------------------------------------------
# Activation patching
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_activation_patching(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric_fn: Callable,
    n_layers: Optional[int] = None,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Full activation patching over all (layer, component) combinations.

    Algorithm:
    - clean = AI response (high score) → corrupted = Human response (low score)
    - For each layer: patch clean → corrupted activations
    - Measure change in metric

    Returns dict of normalized effect matrices:
    - "resid_post": (n_layers, ) — residual stream post each layer
    - "attn_out": (n_layers, )   — attention output contribution
    - "mlp_out": (n_layers, )    — MLP output contribution

    Normalization: (metric_patched - metric_corrupted) / (metric_clean - metric_corrupted)
    Values near 1 = this layer carries all the bias effect.
    """
    if n_layers is None:
        n_layers = model.cfg.n_layers

    # Pad both sequences to the same length (required for activation patching:
    # clean cache shape must match corrupted sequence length at every position)
    clean_len = clean_tokens.shape[1]
    corr_len = corrupted_tokens.shape[1]
    if clean_len != corr_len:
        max_len = max(clean_len, corr_len)
        pad_id = model.tokenizer.pad_token_id or 0
        def _pad(t, target):
            pad = torch.full((t.shape[0], target - t.shape[1]), pad_id,
                             dtype=t.dtype, device=t.device)
            return torch.cat([pad, t], dim=1)  # left-pad
        if clean_len < max_len:
            clean_tokens = _pad(clean_tokens, max_len)
        if corr_len < max_len:
            corrupted_tokens = _pad(corrupted_tokens, max_len)

    # Baseline metrics
    clean_logits = model(clean_tokens, return_type="logits")
    clean_metric = metric_fn(clean_logits).item()
    corr_logits = model(corrupted_tokens, return_type="logits")
    corr_metric = metric_fn(corr_logits).item()

    delta = clean_metric - corr_metric
    if abs(delta) < 1e-6:
        delta = 1e-6  # avoid division by zero

    # Get clean cache
    _, clean_cache = model.run_with_cache(
        clean_tokens,
        names_filter=lambda n: "hook_resid_post" in n or "hook_attn_out" in n or "hook_mlp_out" in n,
    )

    def _patch_metric(hook_name: str) -> float:
        """Run corrupted tokens with the last-token position patched from clean cache.

        We patch only position [-1] (the last token, where logits are measured).
        Patching all positions would always give effect=1.0 regardless of layer
        because the clean residual at any layer completely determines subsequent computation.
        """
        clean_last = clean_cache[hook_name][:, -1:, :]  # (1, 1, d_model)

        def hook_fn(value, hook):
            out = value.clone()
            out[:, -1:, :] = clean_last
            return out

        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits",
        )
        return metric_fn(patched_logits).item()

    resid_effects = np.zeros(n_layers)
    attn_effects = np.zeros(n_layers)
    mlp_effects = np.zeros(n_layers)

    for layer in tqdm(range(n_layers), desc="Patching layers", leave=False):
        for arr, hook_suffix in [
            (resid_effects, "hook_resid_post"),
            (attn_effects, "hook_attn_out"),
            (mlp_effects, "hook_mlp_out"),
        ]:
            hook_name = f"blocks.{layer}.{hook_suffix}"
            if hook_name in clean_cache:
                patched_metric = _patch_metric(hook_name)
                arr[layer] = (patched_metric - corr_metric) / delta

    del clean_cache

    return {
        "resid_post": resid_effects,
        "attn_out": attn_effects,
        "mlp_out": mlp_effects,
        "clean_metric": clean_metric,
        "corrupted_metric": corr_metric,
        "delta": delta,
    }


@torch.no_grad()
def run_activation_patching_batch(
    model,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    metric_fn: Callable,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Run activation patching for multiple (clean, corrupted) pairs and average.
    Returns mean effects across all pairs, same structure as run_activation_patching.
    """
    n_layers = model.cfg.n_layers
    all_resid = []
    all_attn = []
    all_mlp = []

    for clean_tokens, corrupted_tokens in tqdm(pairs, desc="Patching pairs"):
        clean_tokens = clean_tokens.to(device)
        corrupted_tokens = corrupted_tokens.to(device)
        result = run_activation_patching(model, clean_tokens, corrupted_tokens, metric_fn, n_layers, device)
        all_resid.append(result["resid_post"])
        all_attn.append(result["attn_out"])
        all_mlp.append(result["mlp_out"])

    return {
        "resid_post": np.array(all_resid),           # (n_pairs, n_layers)
        "resid_post_mean": np.mean(all_resid, axis=0), # (n_layers,)
        "attn_out": np.array(all_attn),
        "attn_out_mean": np.mean(all_attn, axis=0),
        "mlp_out": np.array(all_mlp),
        "mlp_out_mean": np.mean(all_mlp, axis=0),
    }


# ---------------------------------------------------------------------------
# Attention head attribution
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_head_attribution(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric_fn: Callable,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Attribute each attention head's contribution to the bias.

    Uses path patching: patches each head's output independently via
    'blocks.{layer}.attn.hook_result' (shape: batch, seq, n_heads, d_head).

    Returns attribution matrix shape (n_layers, n_heads).
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_tokens = clean_tokens.to(device)
    corrupted_tokens = corrupted_tokens.to(device)

    # Pad to same length so clean cache and corrupted shapes match
    clean_len = clean_tokens.shape[1]
    corr_len = corrupted_tokens.shape[1]
    if clean_len != corr_len:
        max_len = max(clean_len, corr_len)
        pad_id = model.tokenizer.pad_token_id or 0
        def _pad(t, target):
            pad = torch.full((t.shape[0], target - t.shape[1]), pad_id,
                             dtype=t.dtype, device=t.device)
            return torch.cat([pad, t], dim=1)
        if clean_len < max_len:
            clean_tokens = _pad(clean_tokens, max_len)
        if corr_len < max_len:
            corrupted_tokens = _pad(corrupted_tokens, max_len)

    # Baselines
    clean_logits = model(clean_tokens, return_type="logits")
    clean_metric = metric_fn(clean_logits).item()
    corr_logits = model(corrupted_tokens, return_type="logits")
    corr_metric = metric_fn(corr_logits).item()
    delta = clean_metric - corr_metric
    if abs(delta) < 1e-6:
        delta = 1e-6

    # hook_result requires use_attn_result=True in the model config
    model.cfg.use_attn_result = True
    _, clean_cache = model.run_with_cache(
        clean_tokens,
        names_filter=lambda n: "attn.hook_result" in n,
    )

    attribution = np.zeros((n_layers, n_heads))

    for layer in tqdm(range(n_layers), desc="Head attribution", leave=False):
        hook_name = f"blocks.{layer}.attn.hook_result"
        if hook_name not in clean_cache:
            continue
        clean_head_result = clean_cache[hook_name]  # (batch, seq, n_heads, d_head)
        clean_head_last = clean_head_result[:, -1:, :, :]  # (batch, 1, n_heads, d_head)

        for head in range(n_heads):
            def make_hook(h, clean_last):
                def hook_fn(value, hook):
                    out = value.clone()
                    out[:, -1:, h, :] = clean_last[:, :, h, :]
                    return out
                return hook_fn

            patched_logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(hook_name, make_hook(head, clean_head_last))],
                return_type="logits",
            )
            patched_metric = metric_fn(patched_logits).item()
            attribution[layer, head] = (patched_metric - corr_metric) / delta

    del clean_cache
    return attribution


@torch.no_grad()
def run_head_attribution_batch(
    model,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    metric_fn: Callable,
    device: str = "cuda:0",
) -> np.ndarray:
    """Average head attribution over multiple pairs. Returns (n_layers, n_heads)."""
    all_attr = []
    for clean_tokens, corrupted_tokens in tqdm(pairs, desc="Head attribution pairs"):
        attr = run_head_attribution(model, clean_tokens.to(device), corrupted_tokens.to(device), metric_fn, device)
        all_attr.append(attr)
    return np.mean(all_attr, axis=0)


# ---------------------------------------------------------------------------
# Path patching (Wang et al. 2022 two-stage algorithm)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_path_patching(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    sender_hook: str,
    receiver_hook: str,
    metric_fn: Callable,
) -> float:
    """
    Two-stage path patching: measure the effect of the direct path sender → receiver.

    Unlike activation patching (which lets effects cascade through all downstream
    components), this isolates the contribution of the specific edge sender → receiver:

    Stage 1: Run clean tokens but replace sender's output with the corrupted value.
             Cache what the receiver outputs under this intervention.
    Stage 2: Run clean tokens but replace receiver's output with the stage-1 value.
             Measure the metric.

    Effect = (metric_stage2 - metric_clean) / (metric_clean - metric_corrupted)
    Positive value: this path carries the bias (corrupting sender hurts metric via receiver).
    """
    # Pad to same length
    clean_len = clean_tokens.shape[1]
    corr_len = corrupted_tokens.shape[1]
    if clean_len != corr_len:
        max_len = max(clean_len, corr_len)
        pad_id = model.tokenizer.pad_token_id or 0
        def _pad(t, target):
            p = torch.full((t.shape[0], target - t.shape[1]), pad_id, dtype=t.dtype, device=t.device)
            return torch.cat([p, t], dim=1)
        if clean_len < max_len:
            clean_tokens = _pad(clean_tokens, max_len)
        if corr_len < max_len:
            corrupted_tokens = _pad(corrupted_tokens, max_len)

    # Baselines
    clean_metric = metric_fn(model(clean_tokens, return_type="logits")).item()
    corr_metric = metric_fn(model(corrupted_tokens, return_type="logits")).item()
    delta = clean_metric - corr_metric
    if abs(delta) < 1e-6:
        return 0.0

    # Get corrupted activation at the last token of the sender layer.
    # We patch only position [-1] — the token where logits are read.
    # Patching all positions would fully determine all downstream computation
    # and give effect=-1.0 everywhere regardless of which path carries the signal.
    _, corr_cache = model.run_with_cache(
        corrupted_tokens, names_filter=lambda n: n == sender_hook,
    )
    corrupted_sender_last = corr_cache[sender_hook][:, -1:, :].detach().clone()  # (1, 1, d_model)
    del corr_cache

    # Stage 1: clean run with sender[-1] patched → capture receiver[-1].
    # Use detach().clone() immediately inside the hook to get a fully independent tensor,
    # preventing PyTorch from reusing the same buffer across forward passes.
    captured = [None]  # list avoids closure rebinding issues

    def patch_sender_fn(value, hook):
        out = value.clone()
        out[:, -1:, :] = corrupted_sender_last.to(value.dtype)
        return out

    def capture_receiver_fn(value, hook):
        captured[0] = value[:, -1:, :].detach().clone()  # (1, 1, d_model) — isolated copy
        return value

    model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[(sender_hook, patch_sender_fn), (receiver_hook, capture_receiver_fn)],
        return_type=None,
    )
    receiver_stage1_last = captured[0]

    # Stage 2: clean run with receiver[-1] replaced by stage-1 value → measure metric
    stage1_val = receiver_stage1_last  # local variable — not shared across calls

    def patch_receiver_fn(value, hook):
        out = value.clone()
        out[:, -1:, :] = stage1_val.to(value.dtype)
        return out

    patched_logits = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[(receiver_hook, patch_receiver_fn)],
        return_type="logits",
    )
    patched_metric = metric_fn(patched_logits).item()

    return (patched_metric - clean_metric) / delta


@torch.no_grad()
def run_path_patching_flow_map(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric_fn: Callable,
    n_layers: Optional[int] = None,
    components: str = "resid",
) -> np.ndarray:
    """
    Compute a (n_layers, n_layers) path patching flow map.

    flow_map[src][tgt] = effect of path resid_post[src] → resid_post[tgt] on the metric.
    Only the upper triangle (tgt > src) is meaningful (information flows forward).

    components: "resid" uses hook_resid_post for both sender and receiver.
    """
    if n_layers is None:
        n_layers = model.cfg.n_layers

    # Pad to same length
    clean_len = clean_tokens.shape[1]
    corr_len = corrupted_tokens.shape[1]
    if clean_len != corr_len:
        max_len = max(clean_len, corr_len)
        pad_id = model.tokenizer.pad_token_id or 0
        def _pad(t, target):
            p = torch.full((t.shape[0], target - t.shape[1]), pad_id, dtype=t.dtype, device=t.device)
            return torch.cat([p, t], dim=1)
        if clean_len < max_len:
            clean_tokens = _pad(clean_tokens, max_len)
        if corr_len < max_len:
            corrupted_tokens = _pad(corrupted_tokens, max_len)

    flow_map = np.zeros((n_layers, n_layers))

    for src in tqdm(range(n_layers), desc="Path patching flow map"):
        sender_hook = f"blocks.{src}.hook_resid_post"
        for tgt in range(src + 1, n_layers):
            receiver_hook = f"blocks.{tgt}.hook_resid_post"
            flow_map[src, tgt] = run_path_patching(
                model, clean_tokens, corrupted_tokens,
                sender_hook, receiver_hook, metric_fn,
            )

    return flow_map


@torch.no_grad()
def run_path_patching_flow_map_batch(
    model,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    metric_fn: Callable,
    device: str = "cuda:0",
) -> np.ndarray:
    """Average path patching flow map over multiple (clean, corrupted) pairs."""
    n_layers = model.cfg.n_layers
    all_maps = []
    for clean_tokens, corrupted_tokens in tqdm(pairs, desc="Flow map pairs"):
        fm = run_path_patching_flow_map(
            model, clean_tokens.to(device), corrupted_tokens.to(device), metric_fn, n_layers,
        )
        all_maps.append(fm)
    return np.mean(all_maps, axis=0)


# ---------------------------------------------------------------------------
# Residual stream decomposition
# ---------------------------------------------------------------------------

@torch.no_grad()
def decompose_residual_stream(
    model,
    tokens: torch.Tensor,
    score_position: int,
    up_to_layer: int,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Decompose the residual stream at score_position up to layer L into contributions:
    - "embed": embedding contribution
    - "attn_L{i}": attention output of layer i
    - "mlp_L{i}": MLP output of layer i

    Uses TransformerLens cache.decompose_resid() if available, or manual decomposition.
    Returns dict: component_name → (d_model,) contribution vector.
    """
    tokens = tokens.to(device)
    names_filter = lambda n: (
        "hook_resid_post" in n or "hook_attn_out" in n
        or "hook_mlp_out" in n or "hook_embed" in n
    )
    logits, cache = model.run_with_cache(tokens, names_filter=names_filter)

    contributions = {}

    # Embedding
    if "hook_embed" in cache:
        embed = cache["hook_embed"][0, score_position, :].cpu().float().numpy()
        contributions["embed"] = embed

    # Layer contributions
    for layer in range(up_to_layer + 1):
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        if attn_key in cache:
            contributions[f"attn_L{layer}"] = cache[attn_key][0, score_position, :].cpu().float().numpy()
        if mlp_key in cache:
            contributions[f"mlp_L{layer}"] = cache[mlp_key][0, score_position, :].cpu().float().numpy()

    del cache
    return contributions


# ===========================================================================
# HuggingFace 4-bit backend
# For 70B models that don't fit in a single H100 at full precision.
# Uses PyTorch forward hooks instead of TransformerLens.
# ===========================================================================

def _get_out_proj(layer):
    """Return the output projection Linear module for any supported architecture."""
    attn = layer.self_attn
    if hasattr(attn, "o_proj"):
        return attn.o_proj          # Llama, Qwen2, DeepSeek-R1-Llama/Qwen2
    if hasattr(attn, "dense"):
        return attn.dense           # Phi-4
    raise AttributeError(f"Cannot find output projection in {type(attn)}")


def build_score_metric_hf(
    tokenizer,
    high_score_strings: List[str] = ("6", "7"),
    low_score_strings: List[str] = ("1", "2", "3", "4", "5"),
) -> Callable[[torch.Tensor], float]:
    """
    Score metric for HF models.  Accepts logits (batch, seq, vocab) and
    returns a plain Python float (not a tensor).
    """
    high_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in high_score_strings]
    low_ids  = [tokenizer.encode(s, add_special_tokens=False)[0] for s in low_score_strings]

    def metric(logits: torch.Tensor) -> float:
        last = logits[:, -1, :].float()
        lp = torch.log_softmax(last, dim=-1)
        return float((lp[:, high_ids].mean() - lp[:, low_ids].mean()).item())

    return metric


@torch.no_grad()
def run_activation_patching_hf(
    model,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    metric_fn: Callable,
    n_layers: int,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """
    Activation patching for HF 4-bit models using PyTorch forward hooks.

    Patches the residual stream at the last token position layer by layer,
    equivalent to TransformerLens hook_resid_post patching.

    HF layer output[0] = hidden_states after full layer (resid post).
    Only attn and mlp combined (resid_post) is available; separate attn_out
    and mlp_out require additional hooks not implemented here for brevity.

    Returns dict with keys: "resid_post", "clean_metric", "corrupted_metric", "delta".
    """
    layers = model.model.layers

    # --- Step 1: capture clean residual cache (last token at each layer) ---
    clean_cache: Dict[int, torch.Tensor] = {}

    def make_capture(i):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            clean_cache[i] = hs[:, -1:, :].detach()  # (batch, 1, d_model)
        return hook

    handles = [layers[i].register_forward_hook(make_capture(i)) for i in range(n_layers)]
    clean_logits = model(clean_ids).logits
    clean_metric = metric_fn(clean_logits)
    for h in handles:
        h.remove()

    # --- Step 2: corrupted baseline ---
    corr_logits  = model(corrupted_ids).logits
    corr_metric  = metric_fn(corr_logits)

    delta = clean_metric - corr_metric
    if abs(delta) < 1e-6:
        delta = 1e-6

    # --- Step 3: patch each layer independently ---
    resid_effects = np.zeros(n_layers)

    for layer_idx in range(n_layers):
        clean_last = clean_cache[layer_idx]  # (batch, 1, d_model)

        def make_patch(cl):
            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                patched = hs.clone()
                patched[:, -1:, :] = cl.to(hs.device, dtype=hs.dtype)
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return hook

        h = layers[layer_idx].register_forward_hook(make_patch(clean_last))
        patched_logits  = model(corrupted_ids).logits
        patched_metric  = metric_fn(patched_logits)
        h.remove()

        resid_effects[layer_idx] = (patched_metric - corr_metric) / delta

    return {
        "resid_post":       resid_effects,
        "attn_out":         np.zeros(n_layers),   # not decomposed in 4-bit path
        "mlp_out":          np.zeros(n_layers),
        "clean_metric":     clean_metric,
        "corrupted_metric": corr_metric,
        "delta":            delta,
    }


@torch.no_grad()
def run_head_attribution_hf(
    model,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    metric_fn: Callable,
    n_layers: int,
    n_heads: int,
    d_head: int,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Head attribution for HF 4-bit models.

    Patches the output of each head individually by intercepting the input to
    o_proj (the concatenated head outputs before the output projection).
    Shape: (batch, seq, n_heads * d_head).  Head h occupies [:, :, h*d_head:(h+1)*d_head].

    Returns attribution matrix (n_layers, n_heads).
    """
    layers = model.model.layers

    # --- Step 1: capture clean o_proj inputs at last token for all layers ---
    clean_oproj: Dict[int, torch.Tensor] = {}

    def make_capture_pre(i):
        def hook(module, args):
            x = args[0]
            clean_oproj[i] = x[:, -1:, :].detach()  # (batch, 1, n_heads*d_head)
        return hook

    handles = []
    for i in range(n_layers):
        try:
            proj = _get_out_proj(layers[i])
            handles.append(proj.register_forward_pre_hook(make_capture_pre(i)))
        except AttributeError:
            pass

    clean_logits = model(clean_ids).logits
    clean_metric = metric_fn(clean_logits)
    for h in handles:
        h.remove()

    # --- Step 2: corrupted baseline ---
    corr_logits = model(corrupted_ids).logits
    corr_metric = metric_fn(corr_logits)
    delta = clean_metric - corr_metric
    if abs(delta) < 1e-6:
        delta = 1e-6

    # --- Step 3: patch each (layer, head) pair ---
    # Optimisation: for each layer we do n_heads forwards. We cannot batch them
    # (each needs a different hook). But we process all heads per layer in sequence
    # before moving to the next layer, keeping the layer's cache slice in memory.
    attribution = np.zeros((n_layers, n_heads))

    for layer_idx in tqdm(range(n_layers), desc="Head attr (HF)", leave=False):
        if layer_idx not in clean_oproj:
            continue
        clean_slice_full = clean_oproj[layer_idx]  # (batch, 1, n_heads*d_head)

        try:
            proj = _get_out_proj(layers[layer_idx])
        except AttributeError:
            continue

        for head in range(n_heads):
            h_s = head * d_head
            h_e = (head + 1) * d_head
            clean_head = clean_slice_full[:, :, h_s:h_e]

            def make_head_hook(hs, he, ch):
                def hook(module, args):
                    x = args[0].clone()
                    x[:, -1:, hs:he] = ch.to(x.device, dtype=x.dtype)
                    return (x,)
                return hook

            ph = proj.register_forward_pre_hook(
                make_head_hook(h_s, h_e, clean_head)
            )
            patched_logits = model(corrupted_ids).logits
            patched_metric = metric_fn(patched_logits)
            ph.remove()
            attribution[layer_idx, head] = (patched_metric - corr_metric) / delta

    return attribution


@torch.no_grad()
def run_activation_patching_batch_hf(
    model,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    metric_fn: Callable,
    n_layers: int,
    device: str = "cuda:0",
) -> Dict[str, np.ndarray]:
    """Average activation patching over multiple pairs for HF 4-bit models."""
    all_resid = []
    for clean_ids, corr_ids in tqdm(pairs, desc="Patching pairs (HF)"):
        result = run_activation_patching_hf(
            model, clean_ids.to(device), corr_ids.to(device), metric_fn, n_layers, device
        )
        all_resid.append(result["resid_post"])
    return {
        "resid_post":      np.array(all_resid),
        "resid_post_mean": np.mean(all_resid, axis=0),
        "attn_out":        np.zeros((len(pairs), n_layers)),
        "attn_out_mean":   np.zeros(n_layers),
        "mlp_out":         np.zeros((len(pairs), n_layers)),
        "mlp_out_mean":    np.zeros(n_layers),
    }


@torch.no_grad()
def run_head_attribution_batch_hf(
    model,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    metric_fn: Callable,
    n_layers: int,
    n_heads: int,
    d_head: int,
    device: str = "cuda:0",
) -> np.ndarray:
    """Average head attribution over multiple pairs for HF 4-bit models."""
    all_attr = []
    for clean_ids, corr_ids in tqdm(pairs, desc="Head attr pairs (HF)"):
        attr = run_head_attribution_hf(
            model, clean_ids.to(device), corr_ids.to(device),
            metric_fn, n_layers, n_heads, d_head, device
        )
        all_attr.append(attr)
    return np.mean(all_attr, axis=0)
