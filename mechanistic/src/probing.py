"""Linear probe training, evaluation, geometry analysis."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    layer: int
    position: str
    probe_type: str
    model_key: str
    accuracy: float
    auc: float
    accuracy_ci: Tuple[float, float]
    auc_ci: Tuple[float, float]
    direction: np.ndarray       # (d_model,) unit-normalized weight vector
    intercept: float
    n_train: int
    n_val: int
    auc_std: float = 0.0                        # std across k-folds (0.0 = not k-fold)
    n_folds: int = 0                            # 0 = holdout split, >0 = k-fold CV
    centroid_direction: Optional[np.ndarray] = None  # mean(class1) - mean(class0), unit-normalized

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["direction"] = self.direction.tolist()
        if self.centroid_direction is not None:
            d["centroid_direction"] = self.centroid_direction.tolist()
        return d


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str = "balanced",
) -> Tuple[LogisticRegression, StandardScaler, Dict]:
    """
    Train a linear probe on hidden states.
    Returns (probe, scaler, val_metrics).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    probe = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=42,
    )
    probe.fit(X_train_s, y_train)

    val_preds = probe.predict(X_val_s)
    val_proba = probe.predict_proba(X_val_s)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, val_preds),
        "auc": roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.5,
    }
    return probe, scaler, metrics


def evaluate_probe(
    probe: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """Evaluate probe with bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    X_s = scaler.transform(X_test)
    preds = probe.predict(X_s)
    proba = probe.predict_proba(X_s)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5

    # Bootstrap CIs
    n = len(y_test)
    boot_acc, boot_auc = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_acc.append(accuracy_score(y_test[idx], preds[idx]))
        if len(np.unique(y_test[idx])) > 1:
            boot_auc.append(roc_auc_score(y_test[idx], proba[idx]))

    acc_ci = (np.percentile(boot_acc, 2.5), np.percentile(boot_acc, 97.5))
    auc_ci = (np.percentile(boot_auc, 2.5), np.percentile(boot_auc, 97.5)) if boot_auc else (auc, auc)

    return {
        "accuracy": acc,
        "auc": auc,
        "accuracy_ci": acc_ci,
        "auc_ci": auc_ci,
        "n_test": n,
    }


def get_probe_direction(probe: LogisticRegression, scaler: StandardScaler) -> np.ndarray:
    """Extract unit-normalized weight vector (probe direction in original space)."""
    # Direction in scaled space
    w = probe.coef_[0]
    # Un-scale: w_orig = w / scaler.scale_
    w_orig = w / scaler.scale_
    norm = np.linalg.norm(w_orig)
    if norm < 1e-8:
        return np.zeros_like(w_orig, dtype=np.float32)
    return (w_orig / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# K-fold probe training
# ---------------------------------------------------------------------------

def train_probe_kfold(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray, List[float]]:
    """
    K-fold CV AUC estimate + full-data direction.

    Returns: (mean_auc, std_auc, direction, fold_aucs)

    AUC is measured via stratified k-fold on ALL provided samples.
    Direction is extracted from a LogisticRegression fit on ALL samples
    (best estimate of the true concept direction — more data = better direction).
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        probe, scaler, _ = train_probe(X[train_idx], y[train_idx], X[val_idx], y[val_idx], C=C)
        metrics = evaluate_probe(probe, scaler, X[val_idx], y[val_idx], n_bootstrap=100)
        fold_aucs.append(metrics["auc"])

    # Full-data fit for direction (not for AUC evaluation)
    scaler_full = StandardScaler().fit(X)
    probe_full = LogisticRegression(
        C=C, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=seed,
    ).fit(scaler_full.transform(X), y)
    direction = get_probe_direction(probe_full, scaler_full)
    intercept = float(probe_full.intercept_[0])

    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), direction, fold_aucs, intercept


def train_probe_elasticnet_kfold(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    l1_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray, List[float], float, float]:
    """
    K-fold CV AUC + full-data elastic-net direction with sparsity report.

    Uses LogisticRegression(penalty='elasticnet', solver='saga') which drives
    many weights to exactly zero (L1 component) while keeping the fit stable
    (L2 component). l1_ratio=0.5 balances both.

    Returns: (mean_auc, std_auc, direction, fold_aucs, intercept, sparsity)
    sparsity: fraction of direction dimensions with |w| < 1e-6 (after un-scaling).
    Direction is from a full-data fit (not held-out), unit-normalised.
    """
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        probe = _LR(
            penalty="elasticnet", solver="saga",
            l1_ratio=l1_ratio, C=C,
            max_iter=2000, class_weight="balanced", random_state=seed,
        ).fit(scaler.transform(X[train_idx]), y[train_idx])
        val_proba = probe.predict_proba(scaler.transform(X[val_idx]))[:, 1]
        auc = roc_auc_score(y[val_idx], val_proba) if len(np.unique(y[val_idx])) > 1 else 0.5
        fold_aucs.append(auc)

    # Full-data fit for direction
    scaler_full = StandardScaler().fit(X)
    probe_full = _LR(
        penalty="elasticnet", solver="saga",
        l1_ratio=l1_ratio, C=C,
        max_iter=2000, class_weight="balanced", random_state=seed,
    ).fit(scaler_full.transform(X), y)
    direction = get_probe_direction(probe_full, scaler_full)
    intercept = float(probe_full.intercept_[0])
    sparsity = float((np.abs(direction) < 1e-6).mean())

    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), direction, fold_aucs, intercept, sparsity


def train_all_probes_kfold(
    h5_path: str,
    model_key: str,
    probe_type: str,
    positions: List[str],
    cfg,
    n_splits: int = 5,
    output_dir: Optional[str] = None,
    suffix: str = "_kfold",
) -> List[ProbeResult]:
    """
    Train probes at all layers using k-fold CV on ALL samples (no train/val/test split).

    Saves to {model_key}_{probe_type}{suffix}_results.pkl (default: ..._kfold_results.pkl).
    Does NOT overwrite existing non-kfold results.
    """
    from src.extraction import load_hidden_states

    results = []
    C = cfg.probing.sklearn_params.get("C", 1.0)

    for position in positions:
        # Load ALL samples (split=None)
        acts_all, meta_all = load_hidden_states(h5_path, position, split=None)
        y_all = _get_labels(meta_all, probe_type, cfg)

        mask_valid = y_all >= 0
        n_valid = mask_valid.sum()
        n_layers_plus_1 = acts_all.shape[1]

        print(f"  [{probe_type}@{position}] {n_valid} labeled samples, {n_layers_plus_1} layers, {n_splits}-fold CV")

        for layer in range(n_layers_plus_1):
            X = acts_all[mask_valid, layer, :]
            y = y_all[mask_valid]

            if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
                continue

            mean_auc, std_auc, direction, fold_aucs, intercept = train_probe_kfold(
                X, y, n_splits=n_splits, C=C
            )
            centroid_dir = compute_centroid_diff(X, y)

            result = ProbeResult(
                layer=layer,
                position=position,
                probe_type=probe_type,
                model_key=model_key,
                accuracy=mean_auc,       # use AUC as proxy for accuracy in k-fold
                auc=mean_auc,
                accuracy_ci=(mean_auc - std_auc, mean_auc + std_auc),
                auc_ci=(mean_auc - std_auc, mean_auc + std_auc),
                direction=direction,
                intercept=intercept,
                n_train=len(y),
                n_val=len(y),
                auc_std=std_auc,
                n_folds=n_splits,
                centroid_direction=centroid_dir,
            )
            results.append(result)

    if output_dir and results:
        probe_type_key = probe_type + suffix
        path = Path(output_dir) / f"{model_key}_{probe_type_key}_results.pkl"
        import pickle
        with open(path, "wb") as f:
            pickle.dump(results, f)
        print(f"  Saved {len(results)} results → {path.name}")

    return results


# ---------------------------------------------------------------------------
# Full probe suite
# ---------------------------------------------------------------------------

def train_all_probes(
    h5_path: str,
    model_key: str,
    probe_type: str,
    positions: List[str],
    cfg,
    output_dir: Optional[str] = None,
) -> List[ProbeResult]:
    """
    Train probes at all layers and all specified positions for one probe_type.
    Labels are determined by probe_type.
    Returns list of ProbeResult.
    """
    from src.extraction import load_hidden_states

    results = []
    sklearn_params = cfg.probing.sklearn_params

    for position in positions:
        # Load activations (all layers) for train/val/test
        acts_train, meta_train = load_hidden_states(h5_path, position, split="train")
        acts_val, meta_val = load_hidden_states(h5_path, position, split="val")
        acts_test, meta_test = load_hidden_states(h5_path, position, split="test")

        y_train = _get_labels(meta_train, probe_type, cfg)
        y_val = _get_labels(meta_val, probe_type, cfg)
        y_test = _get_labels(meta_test, probe_type, cfg)

        # Filter out samples with label -1
        def _filter(acts, y):
            mask = y >= 0
            return acts[mask], y[mask]

        n_layers_plus_1 = acts_train.shape[1]

        for layer in range(n_layers_plus_1):
            X_tr, y_tr = _filter(acts_train[:, layer, :], y_train)
            X_val_, y_val_ = _filter(acts_val[:, layer, :], y_val)
            X_te, y_te = _filter(acts_test[:, layer, :], y_test)

            if len(np.unique(y_tr)) < 2 or len(y_tr) < 10:
                continue

            probe, scaler, _ = train_probe(
                X_tr, y_tr, X_val_, y_val_,
                C=sklearn_params.get("C", 1.0),
                max_iter=sklearn_params.get("max_iter", 1000),
                class_weight=sklearn_params.get("class_weight", "balanced"),
            )
            test_metrics = evaluate_probe(probe, scaler, X_te, y_te,
                                          n_bootstrap=cfg.probing.n_bootstrap)
            direction = get_probe_direction(probe, scaler)

            result = ProbeResult(
                layer=layer,
                position=position,
                probe_type=probe_type,
                model_key=model_key,
                accuracy=test_metrics["accuracy"],
                auc=test_metrics["auc"],
                accuracy_ci=test_metrics["accuracy_ci"],
                auc_ci=test_metrics["auc_ci"],
                direction=direction,
                intercept=float(probe.intercept_[0]),
                n_train=len(y_tr),
                n_val=len(y_val_),
            )
            results.append(result)

            # Save fitted probe
            if output_dir:
                _save_fitted_probe(probe, scaler, result, output_dir)

    if output_dir:
        # Merge with any existing results for other positions so we never
        # overwrite results that were trained in a previous run.
        merged = _merge_probe_results(results, output_dir, model_key, probe_type)
        save_probe_results(merged, output_dir, model_key, probe_type)
    return results


def _merge_probe_results(
    new_results: List[ProbeResult],
    output_dir: str,
    model_key: str,
    probe_type: str,
) -> List[ProbeResult]:
    """
    Merge new_results with any existing pkl, keeping existing (position, layer)
    entries and adding/replacing with new ones.
    """
    path = Path(output_dir) / f"{model_key}_{probe_type}_results.pkl"
    if not path.exists():
        return new_results
    try:
        with open(path, "rb") as f:
            existing = pickle.load(f)
    except Exception:
        return new_results
    # Index existing by (position, layer); new_results take priority
    index = {(r.position, r.layer): r for r in existing}
    for r in new_results:
        index[(r.position, r.layer)] = r
    return list(index.values())


def compute_centroid_diff(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute mean(X[y==1]) - mean(X[y==0]), unit-normalized.
    X: (n, d_model), y: (n,) with values 0, 1, or -1 (excluded).
    Returns zero vector if either class is empty.
    """
    mask1 = y == 1
    mask0 = y == 0
    if mask1.sum() == 0 or mask0.sum() == 0:
        return np.zeros(X.shape[1], dtype=np.float32)
    diff = X[mask1].mean(axis=0) - X[mask0].mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm > 1e-8:
        diff = diff / norm
    return diff.astype(np.float32)


def _get_labels(meta: Dict, probe_type: str, cfg) -> np.ndarray:
    """Get binary labels for a probe type from metadata dict."""
    n = len(meta["response_source"])
    ai_sources = cfg.dataset.get("ai_sources", ["AI", "CoT AI"])
    human_sources = cfg.dataset.get("human_sources", ["Human"])
    sources = meta["response_source"]

    if probe_type == "authorship":
        labels = np.full(n, -1, dtype=np.int8)
        for i, src in enumerate(sources):
            if src in ai_sources:
                labels[i] = 1
            elif src in human_sources:
                labels[i] = 0
        return labels

    elif probe_type == "cot":
        labels = np.full(n, -1, dtype=np.int8)
        for i, src in enumerate(sources):
            if src == "CoT AI":
                labels[i] = 1
            elif src == "AI":
                labels[i] = 0
        return labels

    elif probe_type in ("accuracy", "clarity", "completeness"):
        col = f"{probe_type}_score"
        scores = meta[col].astype(np.float32)
        low_t = cfg.probing.score_low_threshold
        high_t = cfg.probing.score_high_threshold
        valid = scores[scores >= 0]
        if len(valid) > 0:
            n_low = int((valid <= low_t).sum())
            n_high = int((valid >= high_t).sum())
            if min(n_low, n_high) < 20:
                low_t = float(np.percentile(valid, 25))
                high_t = float(np.percentile(valid, 75))
        labels = np.full(n, -1, dtype=np.int8)
        labels[(scores >= 1) & (scores <= low_t)] = 0
        labels[scores >= high_t] = 1
        return labels

    elif probe_type == "verbosity":
        # Response length in tokens: score_token_pos - response_start_pos
        # Short (0) = below 25th percentile, Long (1) = above 75th percentile
        lengths = (meta["score_token_pos"].astype(np.int32)
                   - meta["response_start_pos"].astype(np.int32))
        low_t  = float(np.percentile(lengths, 25))
        high_t = float(np.percentile(lengths, 75))
        labels = np.full(n, -1, dtype=np.int8)
        labels[lengths <= low_t]  = 0  # short
        labels[lengths >= high_t] = 1  # long
        return labels

    raise ValueError(f"Unknown probe_type: {probe_type}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_probe_results(
    results: List[ProbeResult],
    output_dir: str,
    model_key: str,
    probe_type: str,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"{model_key}_{probe_type}_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_probe_results(
    probe_dir: str,
    model_key: str,
    probe_type: str,
) -> List[ProbeResult]:
    path = Path(probe_dir) / f"{model_key}_{probe_type}_results.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_fitted_probe(
    probe: LogisticRegression,
    scaler: StandardScaler,
    result: ProbeResult,
    output_dir: str,
) -> None:
    fname = f"{result.model_key}_{result.probe_type}_{result.position}_L{result.layer:02d}_fitted.pkl"
    path = Path(output_dir) / fname
    with open(path, "wb") as f:
        pickle.dump({"probe": probe, "scaler": scaler}, f)


def load_fitted_probe(
    probe_dir: str,
    model_key: str,
    probe_type: str,
    position: str,
    layer: int,
) -> Tuple[LogisticRegression, StandardScaler]:
    fname = f"{model_key}_{probe_type}_{position}_L{layer:02d}_fitted.pkl"
    with open(Path(probe_dir) / fname, "rb") as f:
        d = pickle.load(f)
    return d["probe"], d["scaler"]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def get_peak_probe_layer(
    results: List[ProbeResult],
    position: str,
    probe_type: str,
    metric: str = "auc",
) -> int:
    """Return layer index with highest metric for given (probe_type, position)."""
    filtered = [r for r in results if r.position == position and r.probe_type == probe_type]
    if not filtered:
        raise ValueError(f"No probe results found for {probe_type} @ {position}")
    return max(filtered, key=lambda r: getattr(r, metric)).layer


def compute_geometry(
    results: List[ProbeResult],
    direction_pairs: List[Tuple[str, str]],
    position: str = "score_token",
) -> Dict:
    """
    Compute cosine similarity (and angle) between pairs of probe directions per layer.

    direction_pairs: list of (probe_type_a, probe_type_b) to compare.
    Returns dict: {(type_a, type_b, layer): {"cosine": float, "angle_deg": float}}.
    """
    # Build direction map: {(probe_type, layer): direction}
    dir_map: Dict[Tuple[str, int], np.ndarray] = {}
    for r in results:
        if r.position == position:
            dir_map[(r.probe_type, r.layer)] = r.direction

    geometry = {}
    layers = sorted(set(r.layer for r in results if r.position == position))

    for type_a, type_b in direction_pairs:
        for layer in layers:
            key_a = (type_a, layer)
            key_b = (type_b, layer)
            if key_a not in dir_map or key_b not in dir_map:
                continue
            d_a = dir_map[key_a]
            d_b = dir_map[key_b]
            cosine = float(np.dot(d_a, d_b))  # both unit vectors
            angle_deg = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            geometry[(type_a, type_b, layer)] = {
                "cosine": cosine,
                "angle_deg": angle_deg,
            }

    return geometry


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between activation matrices X (n, d1) and Y (n, d2).
    Invariant to orthogonal transforms and isotropic scaling — valid across models
    with different d_model or different basis vectors.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X @ Y.T, "fro") ** 2
    hsic_xx = np.linalg.norm(X @ X.T, "fro") ** 2
    hsic_yy = np.linalg.norm(Y @ Y.T, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-10 else 0.0


def compute_cross_model_similarity(
    results_by_model: Dict[str, List[ProbeResult]],
    probe_type: str = "authorship",
    position: str = "score_token",
    h5_paths: Optional[Dict[str, str]] = None,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compare probe representations across models using linear CKA on activations.

    Direct cosine similarity between probe weight vectors is mathematically invalid
    across models — different latent spaces have incomparable basis vectors.
    CKA measures whether the pairwise geometry of representations is consistent
    across models, without requiring alignment.

    If h5_paths is provided, computes CKA from activations. Otherwise falls back
    to comparing AUC profiles (qualitative, not geometric).
    Returns dict: {(model_a, model_b): array of CKA values by layer}.
    """
    import h5py

    model_keys = list(results_by_model.keys())
    n_models = len(model_keys)
    similarities = {}

    for i in range(n_models):
        for j in range(i + 1, n_models):
            ka, kb = model_keys[i], model_keys[j]

            if h5_paths and ka in h5_paths and kb in h5_paths:
                # CKA on activations — valid across different d_model
                try:
                    with h5py.File(h5_paths[ka], "r") as fa, \
                         h5py.File(h5_paths[kb], "r") as fb:
                        acts_a = fa[f"activations/{position}"][:]  # (n, n_layers_a+1, d_a)
                        acts_b = fb[f"activations/{position}"][:]  # (n, n_layers_b+1, d_b)
                    n_layers_a = acts_a.shape[1] - 1
                    n_layers_b = acts_b.shape[1] - 1
                    common_layers = list(range(min(n_layers_a, n_layers_b) + 1))
                    cka_vals = np.array([
                        _linear_cka(
                            acts_a[:, l, :].astype(np.float32),
                            acts_b[:, l, :].astype(np.float32),
                        )
                        for l in common_layers
                    ])
                    similarities[(ka, kb)] = cka_vals
                    print(f"  CKA({ka}, {kb}): computed over {len(common_layers)} layers")
                except Exception as e:
                    print(f"  [WARNING] CKA failed for ({ka}, {kb}): {e}")
            else:
                # Fallback: compare AUC profiles (qualitative)
                ra = {r.layer: r.auc for r in results_by_model[ka]
                      if r.probe_type == probe_type and r.position == position}
                rb = {r.layer: r.auc for r in results_by_model[kb]
                      if r.probe_type == probe_type and r.position == position}
                common_layers = sorted(set(ra) & set(rb))
                if not common_layers:
                    continue
                auc_a = np.array([ra[l] for l in common_layers])
                auc_b = np.array([rb[l] for l in common_layers])
                # Pearson correlation of AUC profiles — no geometric meaning,
                # just whether both models find the same layers informative
                corr = float(np.corrcoef(auc_a, auc_b)[0, 1])
                similarities[(ka, kb)] = np.array([corr])
                print(f"  AUC-profile correlation({ka}, {kb}): {corr:.3f} (no HDF5 for CKA)")

    return similarities
