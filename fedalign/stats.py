"""fedalign.stats — the pre-registered statistical analysis plan, as code.

Written and unit-tested BEFORE the grid runs. Analysis code authored after
seeing results is where p-hacking enters, so this module is frozen with the
protocol.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------
# Effect size
# --------------------------------------------------------------------------

def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Tuple[float, str]:
    """Cliff's delta with the conventional magnitude labels."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan"), "undefined"
    diff = np.sign(a[:, None] - b[None, :])
    d = float(diff.sum() / (a.size * b.size))
    ad = abs(d)
    mag = ("negligible" if ad < 0.147 else
           "small" if ad < 0.33 else
           "medium" if ad < 0.474 else "large")
    return d, mag


# --------------------------------------------------------------------------
# Paired comparison — the primary endpoint machinery
# --------------------------------------------------------------------------

def paired_comparison(
    method: Sequence[float],
    baseline: Sequence[float],
    label: str = "",
) -> Dict[str, object]:
    """Paired t-test (primary) + Wilcoxon signed-rank (robustness companion).

    Note on power: with n=5 seeds the two-sided Wilcoxon cannot reach p<0.05
    (minimum attainable p is 0.0625). The primary endpoint therefore uses 10
    seeds, where the minimum attainable p is ~0.002.
    """
    m, b = np.asarray(method, float), np.asarray(baseline, float)
    if m.shape != b.shape:
        raise ValueError(f"paired arrays must match: {m.shape} vs {b.shape}")
    d = m - b
    n = d.size
    out: Dict[str, object] = {
        "label": label,
        "n_pairs": int(n),
        "mean_method": float(m.mean()) if n else float("nan"),
        "mean_baseline": float(b.mean()) if n else float("nan"),
        "mean_diff": float(d.mean()) if n else float("nan"),
        "std_diff": float(d.std(ddof=1)) if n > 1 else float("nan"),
    }
    if n < 2 or np.allclose(d, 0):
        out.update({"t_stat": float("nan"), "p_ttest": 1.0,
                    "p_wilcoxon": 1.0, "wilcoxon_attainable": False})
    else:
        t, p = sps.ttest_rel(m, b)
        out["t_stat"], out["p_ttest"] = float(t), float(p)
        try:
            _, pw = sps.wilcoxon(m, b)
            out["p_wilcoxon"] = float(pw)
        except ValueError:
            out["p_wilcoxon"] = float("nan")
        # Smallest attainable two-sided Wilcoxon p is 2/2**n.
        out["wilcoxon_attainable"] = bool(2.0 / (2 ** n) <= 0.05)
    delta, mag = cliffs_delta(m, b)
    out["cliffs_delta"], out["cliffs_magnitude"] = delta, mag
    lo, hi = bootstrap_ci(d)
    out["diff_ci_lo"], out["diff_ci_hi"] = lo, hi
    return out


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> Dict[str, object]:
    """Holm-Bonferroni step-down within one table family."""
    p = np.asarray(pvals, float)
    n = p.size
    if n == 0:
        return {"p_adjusted": [], "reject": [], "alpha": alpha}
    order = np.argsort(p)
    adj = np.empty(n, float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return {"p_adjusted": adj.tolist(),
            "reject": (adj <= alpha).tolist(),
            "alpha": alpha}


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    stat=np.mean,
) -> Tuple[float, float]:
    """Percentile bootstrap CI over seeds. Used for every figure band."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1)
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return lo, hi


# --------------------------------------------------------------------------
# H1 — the predictor test that carries the paper
# --------------------------------------------------------------------------

def h1_predictor_test(
    alignment_scores: Sequence[float],
    fm_effects: Sequence[float],
    predicted_signs: Optional[Sequence[str]] = None,
    rho_threshold: float = 0.6,
) -> Dict[str, object]:
    """Spearman rho between a-priori alignment and observed FM worst-client effect.

    `fm_effects` = (FM worst-client metric) - (TextCNN worst-client metric),
    per cell, averaged over seeds. Pre-registered success: rho >= 0.6, p < 0.05.
    Alignment is scored so that HIGHER = BETTER aligned (negate pseudo-perplexity).
    """
    x = np.asarray(alignment_scores, float)
    y = np.asarray(fm_effects, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out: Dict[str, object] = {"n_cells": int(x.size), "rho_threshold": rho_threshold}
    if x.size < 3:
        out.update({"spearman_rho": float("nan"), "p_value": float("nan"),
                    "pearson_r": float("nan"), "passes": False})
    else:
        rho, p = sps.spearmanr(x, y)
        r, pr = sps.pearsonr(x, y)
        out.update({
            "spearman_rho": float(rho), "p_value": float(p),
            "pearson_r": float(r), "pearson_p": float(pr),
            "passes": bool(rho >= rho_threshold and p < 0.05),
        })
        lo, hi = _spearman_bootstrap(x, y)
        out["rho_ci_lo"], out["rho_ci_hi"] = lo, hi

    if predicted_signs is not None:
        pred = np.asarray(predicted_signs)[mask] if len(predicted_signs) == mask.size \
            else np.asarray(predicted_signs)
        obs = np.where(y > 0, "protective", np.where(y < 0, "harmful", "neutral"))
        n = min(len(pred), len(obs))
        correct = int(np.sum(pred[:n] == obs[:n]))
        out["sign_correct"] = correct
        out["sign_total"] = int(n)
        out["sign_accuracy"] = float(correct / n) if n else float("nan")
    return out


def _spearman_bootstrap(x, y, n_boot=10000, seed=0, ci=0.95):
    rng = np.random.default_rng(seed)
    n = x.size
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        vals.append(sps.spearmanr(x[idx], y[idx])[0])
    if not vals:
        return float("nan"), float("nan")
    vals = np.asarray(vals)
    return (float(np.percentile(vals, 100 * (1 - ci) / 2)),
            float(np.percentile(vals, 100 * (1 + ci) / 2)))


def fmt_p(p: float) -> str:
    """Exact p-values, per the analysis plan (never 'p < 0.05')."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "p < 0.0001"
    return f"p = {p:.4f}".rstrip("0").rstrip(".") if p >= 1e-4 else f"p = {p:.2e}"
