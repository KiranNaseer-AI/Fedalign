"""fedalign.metrics — per-client metrics and fairness aggregates.

Three per-client metrics, each answering a different question. The accepted
paper used only a fourth one ("global test restricted to the client's classes")
which collapses to single-class recall under extreme skew; that is replaced here.

  P1  local_macro_f1   macro-F1 on the client's OWN held-out split.
                       Deployment utility. Primary endpoint.
  P2  global_bal_acc   balanced accuracy of the client's model view on the
                       global test set. Global competence. Constant across
                       clients under non-personalised algorithms *by
                       construction* (everyone holds the same model) — report
                       it anyway: it is exactly what makes the dissociation
                       between average-case and worst-case fairness visible.
  P3  clientclass_f1   macro-F1 over the client's classes computed on the FULL
                       global test set. False positives from other classes are
                       counted, so unlike the accepted paper's metric this does
                       not degenerate to recall for a one-class client. This is
                       the metric directly comparable to the accepted paper.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def local_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """P1 — macro-F1 over the labels present in this client's own test split."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    labels = np.unique(y_true)
    return float(f1_score(y_true, y_pred, labels=labels,
                          average="macro", zero_division=0))


def global_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """P2 — balanced accuracy on the global test set."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def client_class_macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, client_labels: Sequence[int]
) -> float:
    """P3 — macro-F1 restricted to the client's classes, on the global test set.

    Evaluated over ALL global test rows so that predicting the client's class on
    an out-of-class example is penalised as a false positive. That is the fix
    for the degeneracy in the accepted paper's metric.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = [int(c) for c in client_labels]
    if len(y_true) == 0 or not labels:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=labels,
                          average="macro", zero_division=0))


# --------------------------------------------------------------------------
# Cross-client aggregates — every table reports all of these
# --------------------------------------------------------------------------

def aggregates(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {k: float("nan") for k in
                ("min", "p10", "mean", "median", "hmean", "std", "gap", "gini")}
    return {
        "min": float(v.min()),
        "p10": float(np.percentile(v, 10)),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "hmean": float(_harmonic_mean(v)),
        "std": float(v.std(ddof=0)),
        "gap": float(v.mean() - v.min()),
        "gini": float(_gini(v)),
    }


def _harmonic_mean(v: np.ndarray) -> float:
    """Harmonic mean; 0 if any client scores 0 (correct — it is a fairness measure)."""
    if np.any(v <= 0):
        return 0.0
    return float(len(v) / np.sum(1.0 / v))


def _gini(v: np.ndarray) -> float:
    if v.size == 0 or np.allclose(v, 0):
        return 0.0
    x = np.sort(np.clip(v, 0, None))
    n = x.size
    cum = np.cumsum(x)
    if cum[-1] == 0:
        return 0.0
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def oscillation_index(series: Sequence[float], last_k: int = 15) -> float:
    """Std of the worst-client metric over the final `last_k` evaluated rounds.

    Converts the accepted paper's eyeballed 'it oscillates' into a number with
    a confidence interval.
    """
    v = np.asarray([x for x in series if x is not None and np.isfinite(x)], dtype=float)
    if v.size < 2:
        return float("nan")
    return float(v[-last_k:].std(ddof=0))


def summarise_round(
    per_client: List[Dict[str, float]], metric: str
) -> Dict[str, float]:
    """Aggregate one metric across clients for a single round."""
    return aggregates([c.get(metric, float("nan")) for c in per_client])
