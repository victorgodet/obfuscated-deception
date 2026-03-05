"""Apollo-style logistic regression probe for deception detection.

Trains a single probe across all layers simultaneously, treating each
detect-token as an independent sample with features concatenated across layers.
"""

import gc

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from probes.config import Config


def train_probe(
    activations: dict[int, list[np.ndarray]],
    detect_counts: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> dict:
    """Train Apollo's LogisticRegressionDetector.

    Each detect-token is an independent sample. Features are concatenated
    across layers: (n_layers * hidden_dim,). The probe learns a single
    direction in the joint activation space.

    Train/test split is at the dialogue level to prevent leakage.

    Args:
        activations: layer -> list of (n_detect_i, hidden_dim) float16 arrays
        detect_counts: (N,) number of detect tokens per dialogue
        labels: (N,) array of 0/1 labels per dialogue
        config: Configuration

    Returns:
        Dict with keys:
        - direction: (n_layers, hidden_dim) — the deception direction
        - scaler_mean: (n_layers, hidden_dim)
        - scaler_scale: (n_layers, hidden_dim)
        - layers: list of layer indices used
        - metrics: dict with auroc, accuracy, tpr_at_fpr_001, tpr_at_fpr_0001, etc.
        - train_idx, test_idx: dialogue-level indices
    """
    layers = sorted(activations.keys())
    n_layers = len(layers)
    hidden_dim = activations[layers[0]][0].shape[1]
    n_dialogues = len(labels)

    print(f"Training Apollo-style probe: {n_layers} layer(s), "
          f"hidden_dim={hidden_dim}, {n_dialogues} dialogues")

    # ── Dialogue-level train/test split ──
    dialogue_indices = np.arange(n_dialogues)
    train_idx, test_idx = train_test_split(
        dialogue_indices,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=labels,
    )

    # ── Extract per-token features ──
    def extract_token_features(dial_indices):
        """Gather detect-token activations for a set of dialogues.

        Returns:
            X: (n_total_tokens, n_layers * hidden_dim) float32
            y: (n_total_tokens,) labels
            dialogue_ids: (n_total_tokens,) which dialogue each token belongs to
        """
        all_features = []
        all_labels = []
        all_dialogue_ids = []

        for di in dial_indices:
            n_detect = detect_counts[di]
            if n_detect == 0:
                continue

            # Concatenate activations across layers for each token
            # activations[layer][di] is already (n_detect, hidden_dim) float16
            token_feats = [activations[layer][di] for layer in layers]
            # (n_detect, n_layers * hidden_dim)
            concat_feats = np.concatenate(token_feats, axis=1)

            all_features.append(concat_feats)
            all_labels.append(np.full(n_detect, labels[di]))
            all_dialogue_ids.append(np.full(n_detect, di))

        X = np.concatenate(all_features, axis=0).astype(np.float32)
        y = np.concatenate(all_labels, axis=0)
        dial_ids = np.concatenate(all_dialogue_ids, axis=0)
        return X, y, dial_ids

    X_train, y_train, train_dial_ids = extract_token_features(train_idx)
    X_test, y_test, test_dial_ids = extract_token_features(test_idx)

    # Free activations (~4 GB) now that features are extracted
    del activations
    gc.collect()

    print(f"  Train tokens: {len(y_train)} ({(y_train == 0).sum()} honest, "
          f"{(y_train == 1).sum()} deceptive)")
    print(f"  Test tokens:  {len(y_test)} ({(y_test == 0).sum()} honest, "
          f"{(y_test == 1).sum()} deceptive)")

    # ── StandardScaler (in-place to avoid float64 copy) ──
    scaler = StandardScaler(copy=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Logistic Regression (C = 1/reg_coeff) ──
    C = 1.0 / config.reg_coeff
    probe = LogisticRegression(
        C=C,
        fit_intercept=False,
        max_iter=1000,
        random_state=config.random_seed,
        solver="lbfgs",
    )
    probe.fit(X_train_scaled, y_train)

    # ── Extract direction and scaler parameters ──
    # coef_ is (1, n_layers * hidden_dim) -> reshape to (n_layers, hidden_dim)
    direction = probe.coef_[0].reshape(n_layers, hidden_dim)
    scaler_mean = scaler.mean_.reshape(n_layers, hidden_dim)
    scaler_scale = scaler.scale_.reshape(n_layers, hidden_dim)

    # ── Token-level metrics ──
    train_scores = X_train_scaled @ probe.coef_[0]
    test_scores = X_test_scaled @ probe.coef_[0]

    train_probs = 1 / (1 + np.exp(-train_scores))
    test_probs = 1 / (1 + np.exp(-test_scores))
    train_preds = (train_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)

    token_auroc = float(roc_auc_score(y_test, test_probs))
    token_accuracy = float(accuracy_score(y_test, test_preds))
    train_auroc = float(roc_auc_score(y_train, train_probs))
    train_accuracy = float(accuracy_score(y_train, train_preds))

    # ── Per-dialogue aggregated scores (mean over detect tokens) ──
    def aggregate_dialogue_scores(scores, dial_ids, dial_indices):
        """Mean-aggregate token scores to dialogue level."""
        dial_scores = []
        dial_labels = []
        for di in dial_indices:
            mask = dial_ids == di
            if mask.sum() == 0:
                continue
            dial_scores.append(scores[mask].mean())
            dial_labels.append(labels[di])
        return np.array(dial_scores), np.array(dial_labels)

    test_dial_scores, test_dial_labels = aggregate_dialogue_scores(
        test_scores, test_dial_ids, test_idx
    )
    train_dial_scores, train_dial_labels = aggregate_dialogue_scores(
        train_scores, train_dial_ids, train_idx
    )

    dial_test_probs = 1 / (1 + np.exp(-test_dial_scores))
    dial_auroc = float(roc_auc_score(test_dial_labels, dial_test_probs))

    # ── TPR at low FPR thresholds ──
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, test_probs)
    tpr_at_fpr_001 = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1])
    tpr_at_fpr_0001 = float(tpr[np.searchsorted(fpr, 0.001, side="right") - 1])

    # Dialogue-level TPR@FPR
    dial_fpr, dial_tpr, _ = roc_curve(test_dial_labels, dial_test_probs)
    dial_tpr_at_fpr_001 = float(
        dial_tpr[np.searchsorted(dial_fpr, 0.01, side="right") - 1]
    )

    metrics = {
        "token_auroc": token_auroc,
        "token_accuracy": token_accuracy,
        "train_token_auroc": train_auroc,
        "train_token_accuracy": train_accuracy,
        "dialogue_auroc": dial_auroc,
        "tpr_at_fpr_001": tpr_at_fpr_001,
        "tpr_at_fpr_0001": tpr_at_fpr_0001,
        "dialogue_tpr_at_fpr_001": dial_tpr_at_fpr_001,
        "num_train_dialogues": len(train_idx),
        "num_test_dialogues": len(test_idx),
        "num_train_tokens": len(y_train),
        "num_test_tokens": len(y_test),
        "layers": layers,
        "reg_coeff": config.reg_coeff,
        "C": C,
    }

    print(f"  Token AUROC:    {token_auroc:.4f} (train: {train_auroc:.4f})")
    print(f"  Dialogue AUROC: {dial_auroc:.4f}")
    print(f"  TPR@FPR=1%:     {tpr_at_fpr_001:.4f}")
    print(f"  TPR@FPR=0.1%:   {tpr_at_fpr_0001:.4f}")

    return {
        "direction": direction,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "layers": layers,
        "metrics": metrics,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "test_dial_scores": test_dial_scores,
        "test_dial_labels": test_dial_labels,
        "train_dial_scores": train_dial_scores,
        "train_dial_labels": train_dial_labels,
    }


def score_activations(activations_per_layer, direction, scaler_mean, scaler_scale):
    """Score activations using the learned deception direction.

    score(x) = (1/L) * SUM_l [ dot( (x_l - mu_l) / sigma_l, w_l ) ]

    Args:
        activations_per_layer: dict[int, ndarray] or list of ndarrays
            Each entry is (..., hidden_dim). Can be (seq_len, hidden_dim) for
            per-token scoring or (hidden_dim,) for single-vector scoring.
        direction: (n_layers, hidden_dim)
        scaler_mean: (n_layers, hidden_dim)
        scaler_scale: (n_layers, hidden_dim)

    Returns:
        scores: ndarray of shape (...,) — averaged across layers
    """
    n_layers = direction.shape[0]
    scores = None

    if isinstance(activations_per_layer, dict):
        layers = sorted(activations_per_layer.keys())
        acts_list = [activations_per_layer[l] for l in layers]
    else:
        acts_list = activations_per_layer

    for i, acts in enumerate(acts_list):
        normalized = (acts - scaler_mean[i]) / scaler_scale[i]
        layer_score = normalized @ direction[i]
        if scores is None:
            scores = layer_score
        else:
            scores = scores + layer_score

    return scores / n_layers
