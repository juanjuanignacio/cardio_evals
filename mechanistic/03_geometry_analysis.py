#!/usr/bin/env python3
"""
Block 1 — Step 3: Geometry analysis of probe directions.

Computes:
1. Cosine similarity / angle between authorship and quality probe directions
2. DLA (Direct Logit Attribution) of the authorship direction
3. Summary statistics

If authorship and quality directions are nearly parallel → model conflates them.

Usage:
    python blocks/block1_representations/03_geometry_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, set_global_seed
from src.probing import (
    compute_geometry,
    get_peak_probe_layer,
    load_probe_results,
)
from src.visualization import plot_geometry_angles, plot_probe_accuracy_by_layer


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model_key", default=None)
    parser.add_argument("--position", default="score_token")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument("--device", default=None, help="Ignored (CPU-only script)")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_global_seed(cfg.seed)

    model_key = args.model_key or cfg.primary_model.key
    probe_dir = str(ROOT / cfg.paths["probe_dir"])
    plot_dir = str(ROOT / cfg.paths["plot_dir"])
    table_dir = str(ROOT / cfg.paths["table_dir"])
    Path(table_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== Block 1.3: Geometry Analysis ===")
    if (Path(table_dir) / f"{model_key}_geometry_{args.position}.json").exists() and not args.force:
        print(f"[{model_key}] Geometry analysis already complete. Use --force to rerun.")
        return
    print(f"Model: {model_key} | Position: {args.position}")

    # Load all probe results
    all_results = []
    probe_results_dict = {}
    for probe_type in cfg.probing.probe_types:
        try:
            results = load_probe_results(probe_dir, model_key, probe_type)
            probe_results_dict[probe_type] = results
            all_results.extend(results)
            print(f"  Loaded {len(results)} results for {probe_type}")
        except FileNotFoundError:
            print(f"  [SKIP] {probe_type} not found")

    if not all_results:
        print("[ERROR] No probe results found. Run 02_train_probes.py first.")
        return

    # Detect which positions are actually present in the loaded results
    available_positions = sorted(set(r.position for r in all_results))
    print(f"  Positions found in probe results: {available_positions}")

    # Plot probe AUC by layer — one plot per available position
    for pos in available_positions:
        has_any = any(
            any(r.position == pos for r in results)
            for results in probe_results_dict.values()
        )
        if not has_any:
            continue
        plot_probe_accuracy_by_layer(
            probe_results_dict,
            position=pos,
            model_key=model_key,
            output_path=f"{plot_dir}/{model_key}_probe_auc_by_layer_{pos}.png",
            metric="auc",
        )
        print(f"  Saved probe AUC plot @ {pos}.")

    # Comparison plot: authorship AUC at score_token vs response_start side by side
    COMPARE_POSITIONS = ["score_token", "response_start"]
    compare_positions = [p for p in COMPARE_POSITIONS if p in available_positions]
    if len(compare_positions) >= 2 and "authorship" in probe_results_dict:
        _plot_authorship_position_comparison(
            probe_results_dict, compare_positions, model_key,
            f"{plot_dir}/{model_key}_authorship_position_comparison.png",
        )
        print("  Saved authorship position comparison plot.")

    # Geometry: angles between probe directions
    direction_pairs = [
        ("authorship", "accuracy"),
        ("authorship", "clarity"),
        ("authorship", "completeness"),
        ("authorship", "cot"),
        ("authorship", "verbosity"),
        ("accuracy",   "verbosity"),
        ("accuracy",   "clarity"),
        ("accuracy",   "completeness"),
        ("clarity",    "completeness"),
    ]
    available_types = set(r.probe_type for r in all_results)
    direction_pairs = [(a, b) for a, b in direction_pairs
                       if a in available_types and b in available_types]

    geometry = compute_geometry(all_results, direction_pairs, position=args.position)

    # Print key findings
    print(f"\nGeometry Analysis @ {args.position}:")
    layers = sorted(set(layer for (_, _, layer) in geometry))

    if "authorship" in available_types:
        auth_peak = get_peak_probe_layer(all_results, args.position, "authorship", "auc")
        for type_b in ["accuracy", "completeness", "clarity", "cot", "verbosity"]:
            if type_b not in available_types:
                continue
            try:
                peak_b = get_peak_probe_layer(all_results, args.position, type_b, "auc")
            except ValueError:
                continue

            for layer, label in [(auth_peak, f"authorship peak L{auth_peak}"),
                                  (peak_b,   f"{type_b} peak L{peak_b}")]:
                key = ("authorship", type_b, layer)
                if key not in geometry:
                    continue
                g = geometry[key]
                print(f"  authorship vs {type_b} @ {label}: "
                      f"cosine={g['cosine']:.3f}, angle={g['angle_deg']:.1f}°")
                if abs(g["cosine"]) > 0.5:
                    print(f"    → [FINDING] High alignment — model conflates authorship with {type_b}!")
                elif abs(g["cosine"]) < 0.2:
                    print(f"    → Nearly orthogonal — separate representations")
                else:
                    print(f"    → Moderate alignment — partial conflation")

    # Save geometry JSON
    geo_serializable = {
        str(key): val for key, val in geometry.items()
    }
    geo_path = Path(table_dir) / f"{model_key}_geometry_{args.position}.json"
    with open(geo_path, "w") as f:
        json.dump(geo_serializable, f, indent=2)
    print(f"\nGeometry saved to: {geo_path}")

    # Plot geometry
    if layers and direction_pairs:
        plot_geometry_angles(
            geometry,
            layers=layers,
            output_path=f"{plot_dir}/{model_key}_geometry_{args.position}.png",
            pairs_to_show=direction_pairs[:4],
        )
        print("Saved geometry plot.")

    # All-layer cosine profiles between probe pairs
    _plot_layerwise_cosines(probe_results_dict, args.position, model_key,
                            plot_dir, table_dir)

    # DLA: project authorship direction onto model vocabulary
    _compute_dla(model_key, args.position, all_results, cfg)


def _plot_authorship_position_comparison(probe_results_dict, positions, model_key, output_path):
    """
    Side-by-side AUC curves for authorship probe at score_token vs response_start.

    Answers: at which layer (and position) does the model first encode authorship?
    - Early peak at response_start → model encodes AI/Human from context *before* reading response
    - Peak only at score_token → model needs to read the full response to determine authorship
    """
    auth_results = probe_results_dict.get("authorship", [])
    if not auth_results:
        return

    # Collect AUC by (position, layer)
    pos_data = {}
    for pos in positions:
        filtered = sorted(
            [r for r in auth_results if r.position == pos],
            key=lambda r: r.layer,
        )
        if filtered:
            pos_data[pos] = filtered

    if len(pos_data) < 2:
        return

    # Color map for positions
    pos_colors = {
        "score_token":    "#E74C3C",
        "response_start": "#3498DB",
        "inst_end":       "#2ECC71",
        "question_end":   "#9B59B6",
    }
    pos_labels = {
        "score_token":    "score_token  (where model generates Accuracy digit)",
        "response_start": "response_start  (token before response content)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    # Left panel: overlay both curves on the same axes
    ax = axes[0]
    peak_info = []
    for pos, results in pos_data.items():
        layers = [r.layer for r in results]
        aucs   = [r.auc   for r in results]
        ci_lo  = [r.auc_ci[0] for r in results]
        ci_hi  = [r.auc_ci[1] for r in results]
        color  = pos_colors.get(pos, "gray")
        label  = pos_labels.get(pos, pos)
        ax.plot(layers, aucs, color=color, linewidth=2, label=label)
        ax.fill_between(layers, ci_lo, ci_hi, alpha=0.12, color=color)
        peak_idx = int(np.argmax(aucs))
        ax.axvline(layers[peak_idx], color=color, linestyle=":", alpha=0.5, linewidth=1)
        peak_info.append((pos, layers[peak_idx], aucs[peak_idx]))

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Authorship probe AUC", fontsize=12)
    ax.set_title(f"Authorship probe — position comparison\n{model_key}", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.4, 1.01)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("white")

    # Right panel: difference curve (score_token AUC − response_start AUC per layer)
    ax2 = axes[1]
    st_data = {r.layer: r.auc for r in pos_data.get("score_token", [])}
    rs_data = {r.layer: r.auc for r in pos_data.get("response_start", [])}
    common_layers = sorted(set(st_data) & set(rs_data))
    if common_layers:
        diffs = [st_data[l] - rs_data[l] for l in common_layers]
        colors_diff = ["#E74C3C" if d > 0 else "#3498DB" for d in diffs]
        ax2.bar(common_layers, diffs, color=colors_diff, alpha=0.7, width=0.7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel("AUC(score_token) − AUC(response_start)", fontsize=12)
        ax2.set_title(
            "Which position carries more authorship info per layer?\n"
            "Red = score_token higher  |  Blue = response_start higher",
            fontsize=11,
        )
        ax2.grid(True, alpha=0.25, axis="y")
        ax2.set_facecolor("white")

        # Annotate summary finding
        early_rs_advantage = sum(1 for l, d in zip(common_layers, diffs) if l < 10 and d < 0)
        if early_rs_advantage > 3:
            ax2.set_title(
                ax2.get_title() + "\n→ response_start leads in early layers (contextual encoding)",
                fontsize=10,
            )

    # Peak layer annotations
    for pos, peak_layer, peak_auc in peak_info:
        print(f"  Authorship @ {pos}: peak AUC={peak_auc:.3f} at layer {peak_layer}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_layerwise_cosines(probe_results_dict, position, model_key, plot_dir, table_dir):
    """
    For each pair (probe_type_A, probe_type_B), compute cosine(dir_A_L, dir_B_L)
    at every layer L and plot as a curve. This answers:
    "Is the accuracy probe at L2 pointing in the same direction as the authorship probe at L2?"
    """
    pairs_to_plot = [
        ("authorship", "accuracy"),
        ("authorship", "verbosity"),
        ("accuracy",   "verbosity"),
        ("authorship", "clarity"),
        ("authorship", "completeness"),
        ("authorship", "cot"),
    ]
    available = set(probe_results_dict.keys())
    pairs_to_plot = [(a, b) for a, b in pairs_to_plot if a in available and b in available]
    if not pairs_to_plot:
        return

    # Build layer → direction dicts per probe type (probe and centroid)
    def get_layer_dirs(probe_type, use_centroid=False):
        results = [r for r in probe_results_dict[probe_type] if r.position == position]
        if use_centroid:
            return {r.layer: r.centroid_direction for r in results
                    if r.centroid_direction is not None}
        return {r.layer: r.direction for r in results if r.direction is not None}

    probe_dicts   = {pt: get_layer_dirs(pt, use_centroid=False) for pt in available}
    centroid_dicts = {pt: get_layer_dirs(pt, use_centroid=True)  for pt in available}
    all_layers = sorted(set.intersection(*[set(d.keys()) for d in probe_dicts.values()]))

    if not all_layers:
        return

    def _cosines_for_pair(dir_dicts, type_a, type_b):
        cosines, layers_valid = [], []
        for L in all_layers:
            da = dir_dicts[type_a].get(L)
            db = dir_dicts[type_b].get(L)
            if da is None or db is None:
                continue
            cos = float(np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-8))
            cosines.append(cos)
            layers_valid.append(L)
        return layers_valid, cosines

    # Compute per-layer cosines for probe and centroid directions
    layerwise_probe    = {}
    layerwise_centroid = {}
    for type_a, type_b in pairs_to_plot:
        layerwise_probe[(type_a, type_b)]    = _cosines_for_pair(probe_dicts,    type_a, type_b)
        if type_a in centroid_dicts and type_b in centroid_dicts:
            layerwise_centroid[(type_a, type_b)] = _cosines_for_pair(centroid_dicts, type_a, type_b)

    # Print findings (probe directions)
    print(f"\n=== Layer-wise Cosines — Probe directions ===")
    for (ta, tb), (layers_v, cosines) in layerwise_probe.items():
        if not cosines:
            continue
        peak_idx = int(np.argmax(np.abs(cosines)))
        print(f"\n  {ta} vs {tb}  [probe]:")
        print(f"    Peak |cosine| = {abs(cosines[peak_idx]):.3f} at L{layers_v[peak_idx]}")
        notable = [(L, c) for L, c in zip(layers_v, cosines) if abs(c) > 0.2]
        for L, c in notable:
            print(f"    L{L:2d}: cosine={c:+.3f}")

    print(f"\n=== Layer-wise Cosines — Centroid directions ===")
    for (ta, tb), (layers_v, cosines) in layerwise_centroid.items():
        if not cosines:
            continue
        peak_idx = int(np.argmax(np.abs(cosines)))
        print(f"\n  {ta} vs {tb}  [centroid]:")
        print(f"    Peak |cosine| = {abs(cosines[peak_idx]):.3f} at L{layers_v[peak_idx]}")

    # Save JSON — both probe and centroid curves
    layerwise_serializable = {}
    for (ta, tb), (lv, cv) in layerwise_probe.items():
        layerwise_serializable[f"{ta}_vs_{tb}_probe"] = {"layers": lv, "cosines": cv}
    for (ta, tb), (lv, cv) in layerwise_centroid.items():
        layerwise_serializable[f"{ta}_vs_{tb}_centroid"] = {"layers": lv, "cosines": cv}
    out_path = Path(table_dir) / f"{model_key}_layerwise_cosines_{position}.json"
    with open(out_path, "w") as f:
        json.dump(layerwise_serializable, f, indent=2)

    # Plot — probe (solid) and centroid (dashed) in the same subplot per pair
    n_pairs = len(pairs_to_plot)
    if n_pairs <= 3:
        n_rows, n_cols = 1, n_pairs
    elif n_pairs <= 6:
        n_cols = 3
        n_rows = (n_pairs + 2) // 3
    else:
        n_cols = 4
        n_rows = (n_pairs + 3) // 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
    axes = np.array(axes).flatten()

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#FF9800", "#795548",
              "#00BCD4", "#607D8B"]
    for ax, (type_a, type_b), color in zip(axes, pairs_to_plot, colors):
        # Probe direction — solid
        if (type_a, type_b) in layerwise_probe:
            lv_p, cos_p = layerwise_probe[(type_a, type_b)]
            ax.plot(cos_p, lv_p, "o-", color=color, linewidth=2, markersize=5,
                    label="probe (LR)")
        # Centroid direction — dashed
        if (type_a, type_b) in layerwise_centroid:
            lv_c, cos_c = layerwise_centroid[(type_a, type_b)]
            ax.plot(cos_c, lv_c, "s--", color=color, linewidth=1.5, markersize=4,
                    alpha=0.6, label="centroid")

        ax.axvline(0,    color="gray", linestyle="--", alpha=0.5)
        ax.axvline( 0.5, color="red",  linestyle=":",  alpha=0.4)
        ax.axvline(-0.5, color="red",  linestyle=":",  alpha=0.4)
        ax.set_xlabel("Cosine similarity", fontsize=11)
        ax.set_title(f"{type_a}\nvs {type_b}", fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=7, loc="lower right")

    for ax in axes[n_pairs:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:n_pairs]):
        if i % n_cols == 0:
            ax.set_ylabel("Layer", fontsize=11)

    fig.suptitle(
        f"Layer-wise Cosine Between Directions (solid=probe, dashed=centroid)\n"
        f"{model_key} @ {position}",
        fontsize=13)
    fig.tight_layout()
    out_plot = Path(plot_dir) / f"{model_key}_layerwise_cosines_{position}.png"
    fig.savefig(str(out_plot), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nLayer-wise cosine plot saved: {out_plot}")


def _compute_dla(model_key, position, all_results, cfg):
    """
    Direct Logit Attribution: how much does the authorship direction push toward
    high vs low score tokens?
    Requires loading the model's unembedding matrix.
    """
    try:
        import torch
        from src.model import get_dtype, load_hooked_transformer
        from src.probing import get_peak_probe_layer

        device = cfg.primary_model.device
        peak_layer = get_peak_probe_layer(all_results, position, "authorship", "auc")
        auth_results = [r for r in all_results
                        if r.probe_type == "authorship" and r.position == position
                        and r.layer == peak_layer]
        if not auth_results:
            return

        direction = torch.tensor(auth_results[0].direction, dtype=torch.float32)

        dtype = get_dtype(cfg.primary_model.dtype)
        model, tokenizer = load_hooked_transformer(
            cfg.primary_model.name, device=device, dtype=dtype,
            hf_token=cfg.dataset.get("hf_token"),
        )

        # Unembed direction to vocabulary
        W_U = model.W_U.detach()  # (d_model, vocab) — detach to avoid grad graph
        direction_gpu = direction.to(device).to(W_U.dtype)
        logit_attrs = (W_U.T @ direction_gpu).cpu().float().numpy()

        high_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in ["6", "7"]]
        low_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in ["1", "2", "3", "4", "5"]]

        dla_high = float(np.mean([logit_attrs[i] for i in high_ids]))
        dla_low = float(np.mean([logit_attrs[i] for i in low_ids]))

        print(f"\nDLA @ layer {peak_layer}:")
        print(f"  Authorship direction → high scores: {dla_high:.4f}")
        print(f"  Authorship direction → low scores:  {dla_low:.4f}")
        print(f"  DLA bias (high - low): {dla_high - dla_low:.4f}")
        if dla_high > dla_low:
            print("  → [FINDING] Authorship direction causally pushes toward higher scores!")

        # Top tokens most attributed by authorship direction
        top_k = 20
        top_idx = np.argsort(logit_attrs)[-top_k:][::-1]
        print(f"\n  Top tokens in authorship direction:")
        for idx in top_idx[:10]:
            tok = tokenizer.decode([idx])
            print(f"    '{tok}': {logit_attrs[idx]:.4f}")

        from src.model import free_model_memory
        free_model_memory(model)

    except Exception as e:
        print(f"[INFO] DLA computation skipped: {e}")

    if torch.cuda.is_initialized():
        from src.model import exit_cleanly
        exit_cleanly()


if __name__ == "__main__":
    main()
