"""fedalign.analysis — every table and figure in the paper, built from results.jsonl.

Reads only the append-only log, so analysis never touches training state and can
be re-run any number of times. No single-run curves appear anywhere: every
figure carries a bootstrap CI band over seeds, and every table carries mean+-std.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import read_jsonl
from .stats import (bootstrap_ci, cliffs_delta, fmt_p, h1_predictor_test,
                    holm_bonferroni, paired_comparison)
from .metrics import oscillation_index

# Pre-registered endpoints. P1 is maximised by a constant local-prior predictor
# when a client holds a single class (precision is 1 by construction on a
# single-class test split), so it rewards collapse under extreme skew. P3 scores
# on the FULL global test set, where out-of-class false positives are punished.
PRIMARY = "clientclass_f1"          # P3 — primary endpoint
DEPLOYMENT = "local_macro_f1"       # P1 — deployment utility, co-reported
SECONDARY = "global_bal_acc"        # P2 — global competence
COMPARABLE = "local_macro_f1"       # kept so legacy call sites resolve


def load_results(path: str, drop_heavy: bool = True) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    if drop_heavy:
        for r in rows:
            r.pop("per_client", None)
            r.pop("probe", None)
    return pd.DataFrame(rows)


def load_results_full(path: str) -> List[Dict]:
    return read_jsonl(path)


def final_rounds(df: pd.DataFrame) -> pd.DataFrame:
    """Last evaluated round per run — the convergence snapshot used in tables."""
    if df.empty:
        return df
    ev = df[df[f"{PRIMARY}__min"].notna()] if f"{PRIMARY}__min" in df else df
    return (ev.sort_values("round").groupby("run_id", as_index=False).last())


def _cell_stats(sub: pd.DataFrame, metric: str, agg: str = "min") -> Tuple[float, float, int]:
    col = f"{metric}__{agg}"
    v = sub[col].dropna().values if col in sub else np.array([])
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


# --------------------------------------------------------------------------
# T2 — the headline alignment grid
# --------------------------------------------------------------------------

def table2_alignment_grid(fin: pd.DataFrame, alignment: pd.DataFrame,
                          metric: str = PRIMARY, alpha: float = 0.1) -> pd.DataFrame:
    """FM worst-client effect per cell vs TextCNN, with predicted vs observed sign."""
    rows = []
    sel = fin[(fin.alpha == alpha) & (fin.algo == "fedavg") & (fin.tag.fillna("") == "")]
    for _, a in alignment.iterrows():
        ds, bb = a["dataset"], a["backbone"]
        fm = sel[(sel.dataset == ds) & (sel.model == bb)]
        cnn = sel[(sel.dataset == ds) & (sel.model == "textcnn")]
        m_fm, s_fm, n_fm = _cell_stats(fm, metric, "min")
        m_cnn, s_cnn, n_cnn = _cell_stats(cnn, metric, "min")
        eff = m_fm - m_cnn
        obs = "protective" if eff > 0.01 else ("harmful" if eff < -0.01 else "neutral")
        rows.append({
            "dataset": ds, "backbone": bb,
            "alignment_score": a.get("alignment_score", np.nan),
            "pseudo_ppl": a.get("pseudo_ppl", np.nan),
            "FM_worst": m_fm, "FM_sd": s_fm, "n_seeds_fm": n_fm,
            "CNN_worst": m_cnn, "CNN_sd": s_cnn, "n_seeds_cnn": n_cnn,
            "effect": eff,
            "predicted": a.get("predicted_sign", "?"),
            "observed": obs,
            "sign_correct": a.get("predicted_sign", "?") == obs,
        })
    return pd.DataFrame(rows).sort_values("alignment_score", ascending=False)


def h1_test_from_grid(t2: pd.DataFrame) -> Dict:
    ok = t2.dropna(subset=["alignment_score", "effect"])
    return h1_predictor_test(ok["alignment_score"].values, ok["effect"].values,
                             ok["predicted"].values)


# --------------------------------------------------------------------------
# T3 — alpha sweep, all aggregates, both metrics
# --------------------------------------------------------------------------

def table3_alpha_sweep(fin: pd.DataFrame, cells: Sequence[Tuple[str, str]],
                       metrics=(PRIMARY, COMPARABLE)) -> pd.DataFrame:
    rows = []
    sel = fin[(fin.algo == "fedavg") & (fin.tag.fillna("") == "")]
    for ds, bb in cells:
        for model in (bb, "textcnn"):
            for a in sorted(sel.alpha.unique()):
                sub = sel[(sel.dataset == ds) & (sel.model == model) & (sel.alpha == a)]
                if sub.empty:
                    continue
                row = {"dataset": ds, "cell_backbone": bb, "model": model,
                       "alpha": a, "n_seeds": len(sub)}
                for m in metrics:
                    for agg in ("min", "p10", "mean", "hmean", "gap"):
                        mu, sd, _ = _cell_stats(sub, m, agg)
                        row[f"{m}_{agg}"] = mu
                        row[f"{m}_{agg}_sd"] = sd
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# T4 — method vs baselines, Holm-corrected
# --------------------------------------------------------------------------

def table4_method_vs_baselines(fin: pd.DataFrame, cells: Sequence[Tuple[str, str]],
                               method: str = "ours_a", reference: str = "ditto",
                               metric: str = PRIMARY, alpha: float = 0.1) -> pd.DataFrame:
    rows, pvals = [], []
    sel = fin[(fin.alpha == alpha) & (fin.tag.fillna("") == "")]
    for ds, bb in cells:
        base = sel[(sel.dataset == ds) & (sel.model == bb)]
        algos = sorted(base.algo.unique())
        m_runs = base[base.algo == method].set_index("seed")[f"{metric}__min"]
        for algo in algos:
            sub = base[base.algo == algo]
            mu, sd, n = _cell_stats(sub, metric, "min")
            mmu, _, _ = _cell_stats(sub, metric, "mean")
            row = {"dataset": ds, "backbone": bb, "algo": algo, "n_seeds": n,
                   "worst": mu, "worst_sd": sd, "mean": mmu}
            if algo != method and not m_runs.empty:
                b_runs = sub.set_index("seed")[f"{metric}__min"]
                common = sorted(set(m_runs.index) & set(b_runs.index))
                if len(common) >= 2:
                    cmp_ = paired_comparison(m_runs.loc[common].values,
                                             b_runs.loc[common].values,
                                             f"{method} vs {algo} @ {ds}/{bb}")
                    row.update({"vs_method_diff": cmp_["mean_diff"],
                                "p_raw": cmp_["p_ttest"],
                                "p_wilcoxon": cmp_["p_wilcoxon"],
                                "wilcoxon_has_power": cmp_["wilcoxon_attainable"],
                                "cliffs_delta": cmp_["cliffs_delta"],
                                "cliffs_mag": cmp_["cliffs_magnitude"],
                                "n_pairs": cmp_["n_pairs"]})
                    pvals.append((len(rows), cmp_["p_ttest"]))
            rows.append(row)
    out = pd.DataFrame(rows)
    if pvals:  # Holm-Bonferroni within this table family
        idx = [i for i, _ in pvals]
        adj = holm_bonferroni([p for _, p in pvals])
        out.loc[idx, "p_holm"] = adj["p_adjusted"]
        out.loc[idx, "significant"] = adj["reject"]
    return out


# --------------------------------------------------------------------------
# T6 — mechanism
# --------------------------------------------------------------------------

def table6_mechanism(df: pd.DataFrame, alignment: pd.DataFrame,
                     last_k: int = 10) -> pd.DataFrame:
    rows = []
    sel = df[(df.algo == "fedavg") & (df.tag.fillna("") == "")]
    for _, a in alignment.iterrows():
        ds, bb = a["dataset"], a["backbone"]
        for alpha in sorted(sel.alpha.unique()):
            sub = sel[(sel.dataset == ds) & (sel.model == bb) & (sel.alpha == alpha)]
            if sub.empty or "conflict_rate" not in sub:
                continue
            late = sub[sub["round"] >= sub["round"].max() - last_k]
            cr = late["conflict_rate"].dropna().values
            cm = late["cos_minority"].dropna().values
            if cr.size == 0:
                continue
            lo, hi = bootstrap_ci(cr)
            rows.append({"dataset": ds, "backbone": bb, "alpha": alpha,
                         "alignment_score": a.get("alignment_score", np.nan),
                         "conflict_rate": float(cr.mean()),
                         "conflict_ci_lo": lo, "conflict_ci_hi": hi,
                         "cos_minority": float(cm.mean()) if cm.size else np.nan})
    return pd.DataFrame(rows)


def oscillation_table(df: pd.DataFrame, metric: str = PRIMARY,
                      last_k: int = 15) -> pd.DataFrame:
    rows = []
    col = f"{metric}__min"
    if col not in df:
        return pd.DataFrame()
    for rid, sub in df.groupby("run_id"):
        sub = sub.sort_values("round")
        v = sub[col].dropna().values
        if v.size < 3:
            continue
        head = sub.iloc[0]
        rows.append({"run_id": rid, "dataset": head.dataset, "model": head.model,
                     "algo": head.algo, "alpha": head.alpha, "seed": head.seed,
                     "oscillation": oscillation_index(v, last_k)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _fig(figsize=(7, 5)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def fig1_alignment_scatter(t2: pd.DataFrame, h1: Dict, out: Optional[str] = None):
    """F1 — the paper in one image: alignment predicts the FM's worst-client effect."""
    import matplotlib.pyplot as plt
    fig, ax = _fig((7.2, 5.4))
    d = t2.dropna(subset=["alignment_score", "effect"])
    colors = {"agnews": "#2563eb", "sent140": "#dc2626", "pubmed": "#059669"}
    marks = {"distilbert": "o", "bertweet": "s", "pubmedbert": "^"}
    for _, r in d.iterrows():
        ax.errorbar(r.alignment_score, r.effect,
                    yerr=np.sqrt(r.FM_sd ** 2 + r.CNN_sd ** 2) / max(1, np.sqrt(r.n_seeds_fm)),
                    fmt=marks.get(r.backbone, "o"), color=colors.get(r.dataset, "k"),
                    markersize=10, capsize=3, alpha=0.9)
        ax.annotate(f"{r.dataset[:4]}x{r.backbone[:4]}",
                    (r.alignment_score, r.effect), fontsize=7,
                    xytext=(6, 4), textcoords="offset points")
    if len(d) >= 3:
        z = np.polyfit(d.alignment_score, d.effect, 1)
        xs = np.linspace(d.alignment_score.min(), d.alignment_score.max(), 50)
        ax.plot(xs, np.polyval(z, xs), "--", color="0.4", linewidth=1.3)
    ax.axhline(0, color="0.2", linewidth=1)
    ax.set_xlabel("A-priori alignment score  (higher = better aligned)")
    ax.set_ylabel("FM worst-client effect\n(FM $-$ TextCNN, local macro-F1)")
    rho, p = h1.get("spearman_rho", np.nan), h1.get("p_value", np.nan)
    ax.set_title(f"Alignment predicts the FM worst-client effect\n"
                 f"Spearman $\\rho$ = {rho:.2f}, {fmt_p(p)}, "
                 f"signs {h1.get('sign_correct','?')}/{h1.get('sign_total','?')}")
    ax.text(0.02, 0.04, "FM protects worst client", transform=ax.transAxes,
            fontsize=8, color="0.35")
    ax.text(0.02, 0.96, "", transform=ax.transAxes)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def fig2_gap_vs_alpha(fin: pd.DataFrame, cells: Sequence[Tuple[str, str]],
                      metric: str = PRIMARY, out: Optional[str] = None):
    """F2 — worst-client gap vs alpha, one line per backbone, CI bands."""
    import matplotlib.pyplot as plt
    datasets = sorted({d for d, _ in cells})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.2),
                             sharey=True, squeeze=False)
    sel = fin[(fin.algo == "fedavg") & (fin.tag.fillna("") == "")]
    for ax, ds in zip(axes[0], datasets):
        models = sorted(sel[sel.dataset == ds].model.unique())
        for m in models:
            sub = sel[(sel.dataset == ds) & (sel.model == m)]
            alphas = sorted(sub.alpha.unique())
            mus, los, his = [], [], []
            for a in alphas:
                v = sub[sub.alpha == a][f"{metric}__gap"].dropna().values
                if v.size == 0:
                    mus.append(np.nan); los.append(np.nan); his.append(np.nan); continue
                lo, hi = bootstrap_ci(v)
                mus.append(v.mean()); los.append(lo); his.append(hi)
            ax.plot(alphas, mus, "o-", label=m, linewidth=1.8, markersize=5)
            ax.fill_between(alphas, los, his, alpha=0.18)
        ax.set_xscale("log")
        ax.set_xlabel(r"Dirichlet $\alpha$  (lower = more heterogeneous)")
        ax.set_title(ds)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel("Worst-client gap (mean $-$ min)")
    fig.suptitle("Worst-client gap vs heterogeneity — the transition moves with alignment",
                 fontsize=11)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def fig3_dynamics(df: pd.DataFrame, cells: Sequence[Tuple[str, str]],
                  alpha: float = 0.1, metric: str = PRIMARY, out: Optional[str] = None):
    """F3 — training dynamics with 5-seed CI bands: aligned vs misaligned cell."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cells), figsize=(5.0 * len(cells), 4.2),
                             sharey=True, squeeze=False)
    sel = df[(df.alpha == alpha) & (df.algo == "fedavg") & (df.tag.fillna("") == "")]
    for ax, (ds, bb) in zip(axes[0], cells):
        for m, color in ((bb, "#dc2626"), ("textcnn", "#2563eb")):
            sub = sel[(sel.dataset == ds) & (sel.model == m)]
            if sub.empty:
                continue
            for agg, style in (("mean", "-"), ("min", "--")):
                col = f"{metric}__{agg}"
                piv = sub.pivot_table(index="round", columns="seed", values=col)
                rounds = piv.index.values
                mu = piv.mean(axis=1).values
                lo = np.array([bootstrap_ci(piv.loc[r].dropna().values)[0] for r in rounds])
                hi = np.array([bootstrap_ci(piv.loc[r].dropna().values)[1] for r in rounds])
                ax.plot(rounds, mu, style, color=color, linewidth=1.7,
                        label=f"{m} {agg}")
                ax.fill_between(rounds, lo, hi, color=color, alpha=0.15)
        ax.set_title(f"{ds} x {bb}  ($\\alpha$={alpha})")
        ax.set_xlabel("Communication round")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel("Local macro-F1")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def fig4_conflict(t6: pd.DataFrame, out: Optional[str] = None):
    """F4 — mechanism: adapter update conflict vs alpha, coloured by alignment."""
    import matplotlib.pyplot as plt
    fig, ax = _fig((7.0, 4.6))
    if t6.empty:
        return fig
    sc = None
    for (ds, bb), sub in t6.groupby(["dataset", "backbone"]):
        sub = sub.sort_values("alpha")
        sc = ax.scatter(sub.alpha, sub.conflict_rate, c=sub.alignment_score,
                        cmap="coolwarm_r", s=70, vmin=t6.alignment_score.min(),
                        vmax=t6.alignment_score.max(), zorder=3)
        ax.plot(sub.alpha, sub.conflict_rate, "-", color="0.7", linewidth=1, zorder=2)
        ax.fill_between(sub.alpha, sub.conflict_ci_lo, sub.conflict_ci_hi, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet $\alpha$")
    ax.set_ylabel("Adapter update conflict rate")
    ax.set_title("Global–local interference rises with skew and falls with alignment")
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="alignment score")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def fig5_method_bars(t4: pd.DataFrame, metric_label: str = "worst-client local macro-F1",
                     out: Optional[str] = None):
    """F5 — method vs baselines with CIs and significance markers."""
    import matplotlib.pyplot as plt
    if t4.empty:
        return _fig()[0]
    cells = t4.groupby(["dataset", "backbone"])
    fig, axes = plt.subplots(1, len(cells), figsize=(4.8 * len(cells), 4.4),
                             sharey=True, squeeze=False)
    for ax, ((ds, bb), sub) in zip(axes[0], cells):
        sub = sub.sort_values("worst")
        x = np.arange(len(sub))
        cols = ["#dc2626" if a.startswith("ours") else "#94a3b8" for a in sub.algo]
        ax.bar(x, sub.worst, yerr=sub.worst_sd, color=cols, capsize=3, alpha=0.9)
        for i, (_, r) in enumerate(sub.iterrows()):
            if r.get("significant") is True:
                ax.text(i, r.worst + (r.worst_sd or 0) + 0.01, "*",
                        ha="center", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.algo, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{ds} x {bb}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, axis="y", linewidth=0.6)
    axes[0][0].set_ylabel(metric_label)
    fig.suptitle("Worst-client performance: method vs baselines "
                 "(* = Holm-corrected p < 0.05 vs Ditto)", fontsize=10)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def latex_table(df: pd.DataFrame, caption: str, label: str,
                float_fmt: str = "%.3f") -> str:
    return df.to_latex(index=False, float_format=lambda v: float_fmt % v,
                       caption=caption, label=label, escape=True,
                       na_rep="--", longtable=False)
