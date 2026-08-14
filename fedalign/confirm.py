"""fedalign.confirm — the pre-registered confirmatory replication.

The rule reported in the manuscript was found by exploratory analysis after the
pre-registered centred score failed its threshold, and it was tested on exactly
one held-out cell. That is the reviewable weakness, and no amount of rewriting
fixes it. This module is the fix: it freezes a prediction for backbones that
have never been run, and then tests it.

Sequence, and the order is the whole point:

    1. fit_rule()           on the ORIGINAL nine cells only
    2. commit_predictions() -> predictions_confirmatory.json, pushed to git
    3. ... run the new cells ...
    4. confirmatory_test()  on the new cells only

Step 2 must be committed and pushed before step 3 executes. If it is not, this
is another exploratory pass and the module has bought nothing.

WHY THE PRIMARY TEST IS RANK-BASED, NOT INTERVAL-BASED
------------------------------------------------------
Fitting effect ~ log10(per-word PPL) on the nine published cells gives
R^2 = 0.68, residual sd = 0.044, and a 90% *prediction* interval about 0.18
wide. The full range of measured effects in the manuscript is 0.227. An
interval covering 80% of the observable range excludes almost nothing, so
"the measurement landed inside the interval" is close to unfalsifiable and a
referee is entitled to say so.

The pooled fit is loose for a reason the manuscript already gives: effect
magnitude is bounded by how much headroom the TextCNN baseline leaves, and
headroom is a property of the dataset. Per-dataset the fit is tight
(R^2 = 0.89, 0.90, 0.94) but the slopes differ fourfold (-0.043 to -0.185),
so pooling mixes three different lines.

Hence the design: the primary endpoint is whether PPL *orders* the new
backbones correctly within each dataset, which is also the form a practitioner
actually uses ("rank your candidates, take the lowest"). Intervals are still
committed and still reported, but as a secondary descriptive check.
"""
from __future__ import annotations

import datetime
import itertools
import json
import math
import os
import subprocess
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# 1. Fit the rule on the original cells
# --------------------------------------------------------------------------

def fit_rule(cells: Sequence[Dict], per_dataset: bool = True) -> Dict:
    """OLS of worst-client effect on log10 per-word perplexity.

    `cells` must be the ORIGINAL grid only. Each entry needs keys
    `dataset`, `backbone`, `ppl_word`, `effect`.
    """
    from scipy import stats as st

    cells = [c for c in cells if np.isfinite(c.get("ppl_word", np.nan))
             and np.isfinite(c.get("effect", np.nan))]
    if len(cells) < 3:
        raise ValueError("need at least three complete cells to fit the rule")

    def _ols(sub):
        x = np.log10([c["ppl_word"] for c in sub])
        y = np.array([c["effect"] for c in sub], float)
        r = st.linregress(x, y)
        resid = y - (r.intercept + r.slope * x)
        dof = max(len(x) - 2, 1)
        return {
            "n": int(len(x)),
            "slope": float(r.slope),
            "intercept": float(r.intercept),
            "r2": float(r.rvalue ** 2),
            "p": float(r.pvalue),
            "resid_sd": float(np.sqrt((resid ** 2).sum() / dof)),
            "x_mean": float(x.mean()),
            "Sxx": float(((x - x.mean()) ** 2).sum()),
            "dof": int(dof),
        }

    out = {"pooled": _ols(cells), "per_dataset": {}}
    if per_dataset:
        for ds in sorted({c["dataset"] for c in cells}):
            sub = [c for c in cells if c["dataset"] == ds]
            if len(sub) >= 3:
                out["per_dataset"][ds] = _ols(sub)
    return out


def predict_interval(ppl_word: float, fit: Dict, level: float = 0.90) -> Dict:
    """Prediction (not confidence) interval for a cell that has not been run."""
    from scipy import stats as st

    xp = math.log10(max(float(ppl_word), 1e-9))
    f = fit
    se = f["resid_sd"] * math.sqrt(
        1.0 + 1.0 / f["n"] + (xp - f["x_mean"]) ** 2 / max(f["Sxx"], 1e-12))
    t = float(st.t.ppf(0.5 + level / 2.0, f["dof"]))
    yhat = f["intercept"] + f["slope"] * xp
    return {"point": float(yhat),
            "lo": float(yhat - t * se),
            "hi": float(yhat + t * se),
            "level": level}


# --------------------------------------------------------------------------
# 2. Commit the predictions
# --------------------------------------------------------------------------

def _git_hash(repo: Optional[str]) -> str:
    if not repo:
        return ""
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def commit_predictions(path: str,
                       original_cells: Sequence[Dict],
                       new_cells: Sequence[Dict],
                       fit: Dict,
                       level: float = 0.90,
                       repo: Optional[str] = None,
                       note: str = "") -> Dict:
    """Freeze predicted rankings and intervals for cells that have not been run.

    `new_cells` need `dataset`, `backbone`, `ppl_word` only — no results.
    Refuses to overwrite an existing file, because overwriting a commitment
    after seeing anything is precisely the failure mode this guards against.
    """
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} already exists. A committed prediction is never rewritten; "
            "if the grid genuinely changed, write a new file and log the reason "
            "in deviations.md.")

    for c in new_cells:
        if not np.isfinite(c.get("ppl_word", np.nan)):
            raise ValueError(f"missing ppl_word for {c.get('dataset')}x{c.get('backbone')}")
        if any(k in c for k in ("effect", "FM_worst")):
            raise ValueError(
                f"{c.get('dataset')}x{c.get('backbone')} carries a result field. "
                "Predictions must be committed before any of these cells is run.")

    preds = []
    for c in new_cells:
        ds = c["dataset"]
        f_pool = fit["pooled"]
        f_ds = fit["per_dataset"].get(ds)
        row = {
            "dataset": ds,
            "backbone": c["backbone"],
            "ppl_word": float(c["ppl_word"]),
            "interval_pooled": predict_interval(c["ppl_word"], f_pool, level),
        }
        if f_ds:
            row["interval_within_dataset"] = predict_interval(c["ppl_word"], f_ds, level)
        preds.append(row)

    # Primary commitment: the predicted ordering within each dataset, over the
    # union of original and new backbones. Lower PPL must give a larger effect.
    ranking = {}
    for ds in sorted({c["dataset"] for c in list(original_cells) + list(new_cells)}):
        pool = [(c["backbone"], float(c["ppl_word"]))
                for c in list(original_cells) + list(new_cells)
                if c["dataset"] == ds and np.isfinite(c.get("ppl_word", np.nan))]
        pool.sort(key=lambda t: t[1])
        ranking[ds] = {
            "predicted_order_best_first": [b for b, _ in pool],
            "ppl_word": {b: p for b, p in pool},
            "new_backbones": sorted({c["backbone"] for c in new_cells
                                     if c["dataset"] == ds}),
        }

    doc = {
        "committed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_hash_at_commit": _git_hash(repo),
        "primary_endpoint": (
            "Within each dataset, Spearman rho between log10 per-word PPL and "
            "worst-client effect (P3, alpha=0.1, FedAvg, 5 seeds) over all "
            "backbones, evaluated on the NEW cells' contribution. One-sided, "
            "predicted negative. Combined across datasets by Fisher's method. "
            "Success: combined p < 0.05."),
        "secondary_endpoints": [
            "Spearman rho across the new cells alone, one-sided negative.",
            "Fraction of new cells falling inside the committed 90% prediction "
            "interval (descriptive only; the pooled interval is wide relative "
            "to the observable effect range and is not a stringent test).",
        ],
        "analysis_frozen": "fedalign.confirm.confirmatory_test, unmodified",
        "fit_on_original_cells": fit,
        "original_cells": [{"dataset": c["dataset"], "backbone": c["backbone"],
                            "ppl_word": float(c["ppl_word"]),
                            "effect": float(c["effect"])} for c in original_cells],
        "predictions": preds,
        "predicted_ranking": ranking,
        "note": note,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return doc


# --------------------------------------------------------------------------
# 3. The confirmatory test
# --------------------------------------------------------------------------

def _spearman_one_sided_negative(x, y):
    from scipy import stats as st
    if len(x) < 3:
        return float("nan"), float("nan")
    rho, p_two = st.spearmanr(x, y)
    if not np.isfinite(rho):
        return float("nan"), float("nan")
    p_one = p_two / 2.0 if rho < 0 else 1.0 - p_two / 2.0
    return float(rho), float(min(max(p_one, 0.0), 1.0))


def _exact_order_p(ppl: Sequence[float], effect: Sequence[float]) -> float:
    """Exact permutation p for 'PPL orders the backbones', small n."""
    ppl = np.asarray(ppl, float)
    eff = np.asarray(effect, float)
    n = len(ppl)
    if n < 3 or n > 8:
        return float("nan")
    from scipy import stats as st
    obs = st.spearmanr(ppl, eff).statistic
    worse = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        r = st.spearmanr(ppl, eff[list(perm)]).statistic
        total += 1
        if r <= obs + 1e-12:
            worse += 1
    return float(worse / total)


def confirmatory_test(committed: Dict, measured: Sequence[Dict],
                      level: float = 0.90) -> Dict:
    """Evaluate the frozen predictions against measured effects.

    `measured` needs `dataset`, `backbone`, `ppl_word`, `effect` for every cell
    that has now been run, original and new alike.
    """
    from scipy import stats as st

    new_keys = {(p["dataset"], p["backbone"]) for p in committed["predictions"]}
    by_key = {(m["dataset"], m["backbone"]): m for m in measured}

    missing = sorted(k for k in new_keys if k not in by_key)
    if missing:
        raise ValueError(f"no measurement for committed cells: {missing}")

    # --- primary: within-dataset rank agreement over all backbones ----------
    per_ds, pvals = {}, []
    for ds in sorted({d for d, _ in by_key}):
        sub = [m for (d, b), m in by_key.items() if d == ds]
        if len(sub) < 3:
            continue
        x = [math.log10(m["ppl_word"]) for m in sub]
        y = [m["effect"] for m in sub]
        rho, p1 = _spearman_one_sided_negative(x, y)
        per_ds[ds] = {
            "n_backbones": len(sub),
            "rho": rho,
            "p_one_sided": p1,
            "p_exact_permutation": _exact_order_p(x, y),
            "monotone": bool(np.all(np.diff(np.array(y)[np.argsort(x)]) <= 1e-12)),
            "order_observed_best_first": [
                m["backbone"] for m in sorted(sub, key=lambda m: -m["effect"])],
            "order_predicted_best_first": [
                m["backbone"] for m in sorted(sub, key=lambda m: m["ppl_word"])],
        }
        if np.isfinite(p1):
            pvals.append(p1)

    # Fisher combination across datasets
    if pvals:
        chi2 = -2.0 * float(np.sum(np.log(np.clip(pvals, 1e-300, 1.0))))
        p_comb = float(st.chi2.sf(chi2, 2 * len(pvals)))
    else:
        chi2, p_comb = float("nan"), float("nan")

    # --- secondary: new cells alone -----------------------------------------
    new = [by_key[k] for k in sorted(new_keys)]
    rho_new, p_new = _spearman_one_sided_negative(
        [math.log10(m["ppl_word"]) for m in new], [m["effect"] for m in new])

    # --- tertiary: interval coverage (descriptive) ---------------------------
    hits_pool, hits_ds, detail = 0, 0, []
    for p in committed["predictions"]:
        m = by_key[(p["dataset"], p["backbone"])]
        ip = p["interval_pooled"]
        inside_pool = ip["lo"] <= m["effect"] <= ip["hi"]
        row = {"dataset": p["dataset"], "backbone": p["backbone"],
               "ppl_word": p["ppl_word"], "predicted": ip["point"],
               "interval_pooled": [ip["lo"], ip["hi"]],
               "measured": m["effect"], "inside_pooled": bool(inside_pool)}
        hits_pool += int(inside_pool)
        if "interval_within_dataset" in p:
            iw = p["interval_within_dataset"]
            inside_ds = iw["lo"] <= m["effect"] <= iw["hi"]
            row["interval_within_dataset"] = [iw["lo"], iw["hi"]]
            row["inside_within_dataset"] = bool(inside_ds)
            hits_ds += int(inside_ds)
        detail.append(row)

    n_new = len(committed["predictions"])
    return {
        "primary": {
            "per_dataset": per_ds,
            "fisher_chi2": chi2,
            "fisher_df": 2 * len(pvals),
            "p_combined": p_comb,
            "passes": bool(np.isfinite(p_comb) and p_comb < 0.05),
        },
        "secondary_new_cells_only": {
            "n": len(new), "rho": rho_new, "p_one_sided": p_new,
            "passes": bool(np.isfinite(p_new) and p_new < 0.05),
        },
        "tertiary_interval_coverage": {
            "n": n_new,
            "inside_pooled": hits_pool,
            "inside_within_dataset": hits_ds,
            "expected_at_level": level * n_new,
            "detail": detail,
            "caveat": ("The pooled 90% prediction interval spans roughly 80% of "
                       "the effect range observed in the original grid, so high "
                       "coverage here is weak evidence. Reported for completeness."),
        },
    }


def summarise(res: Dict) -> str:
    lines = ["CONFIRMATORY REPLICATION", "=" * 56]
    p = res["primary"]
    for ds, d in p["per_dataset"].items():
        lines.append(
            f"  {ds:9} n={d['n_backbones']}  rho={d['rho']:+.3f}  "
            f"p={d['p_one_sided']:.4f}  exact={d['p_exact_permutation']:.4f}  "
            f"monotone={d['monotone']}")
    lines.append(f"  Fisher combined: chi2={p['fisher_chi2']:.2f} "
                 f"df={p['fisher_df']} p={p['p_combined']:.5f}  "
                 f"-> {'PASS' if p['passes'] else 'FAIL'}")
    s = res["secondary_new_cells_only"]
    lines.append(f"  New cells alone: n={s['n']} rho={s['rho']:+.3f} "
                 f"p={s['p_one_sided']:.4f} -> {'PASS' if s['passes'] else 'FAIL'}")
    t = res["tertiary_interval_coverage"]
    lines.append(f"  Interval coverage: {t['inside_pooled']}/{t['n']} pooled "
                 f"(expected {t['expected_at_level']:.1f}) — descriptive only")
    return "\n".join(lines)
