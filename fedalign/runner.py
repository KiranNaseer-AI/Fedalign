"""fedalign.runner — manifest, resume, and the gate-ordered execution queue.

The manifest is the whole reliability story. Every run is a row with a status;
the queue claims the next pending row, checkpoints as it goes, and marks the row
done. A Colab disconnect therefore costs at most one partial run, and re-running
the driver cell simply continues.

Runs are ordered by PRIORITY, not by convenience, so that the decision gates in
the protocol are reached as early as possible:

  p=0   smoke tests
  p=10  G1 cells (3 diagonal + 2 corners, alpha=0.1) -> tests the sign predictions
  p=20  remainder of the 3x3 grid at alpha in {0.1, 1.0}
  p=30  alpha sweep on the 5 primary cells      -> G2 (H1 rho)
  p=40  full baseline suite at alpha=0.1
  p=50  method candidates A/B/C                 -> G3
  p=60  10-seed primary endpoint (method vs Ditto)
  p=70  appendix conditions

If H1 dies, it dies after roughly 10-15% of the compute rather than after all of it.
"""
from __future__ import annotations

import os
import time
import traceback
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .core import (RunConfig, append_jsonl, atomic_write_json, atomic_write_torch,
                   get_device, set_seed)

# ---- the 3x3 alignment grid -------------------------------------------------
# Matched-capacity backbones (~110-125M params, 12 layers) so that pretraining
# DOMAIN is the only systematic difference across the grid. DistilBERT (6L, 66M)
# would confound domain with capacity, so it is demoted to appendix continuity
# runs that link back to the accepted paper.
DIAGONAL = [("agnews", "roberta"), ("sent140", "bertweet"), ("pubmed", "pubmedbert")]
CORNERS = [("pubmed", "bertweet"), ("sent140", "pubmedbert")]
OFF_GRID = [("agnews", "bertweet"), ("agnews", "pubmedbert"),
            ("sent140", "roberta"), ("pubmed", "roberta")]
PRIMARY_CELLS = DIAGONAL + CORNERS
ALL_CELLS = DIAGONAL + CORNERS + OFF_GRID

BASELINES = ["fedprox", "scaffold", "qfedavg", "ditto", "ffa_lora", "local_only"]
METHODS = ["ours_a", "ours_b", "ours_c"]

DEFAULT_LR = {"textcnn": 1e-3, "distilbert": 5e-5, "bertweet": 5e-5,
              "pubmedbert": 5e-5, "roberta": 5e-5}


def _mk(dataset, model, algo, alpha, seed, priority, **kw) -> Dict:
    cfg = RunConfig(dataset=dataset, model=model, algo=algo, alpha=alpha, seed=seed,
                    priority=priority,
                    lr=kw.pop("lr", DEFAULT_LR.get(model, 5e-5)),
                    optimizer=kw.pop("optimizer",
                                     "adamw" if model != "textcnn" else "adamw"),
                    **kw)
    cfg.run_id = cfg.make_id()
    return cfg.to_dict()


def build_manifest(seeds=(0, 1, 2, 3, 4), primary_seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                   rounds: int = 40, include_appendix: bool = True) -> pd.DataFrame:
    rows: List[Dict] = []
    base = dict(rounds=rounds, n_clients=10, local_epochs=1, batch_size=32,
                n_train=20000, max_len=128, eval_every=2)

    # p=10 : G1 — the five cells whose predicted signs decide whether to continue
    for ds, bb in PRIMARY_CELLS:
        for s in seeds:
            rows.append(_mk(ds, bb, "fedavg", 0.1, s, 10, **base))
    for ds in sorted({d for d, _ in PRIMARY_CELLS}):
        for s in seeds:
            rows.append(_mk(ds, "textcnn", "fedavg", 0.1, s, 10, **base))

    # p=20 : remainder of the 3x3 grid at alpha in {0.1, 1.0}
    for ds, bb in ALL_CELLS:
        for a in (0.1, 1.0):
            for s in seeds:
                r = _mk(ds, bb, "fedavg", a, s, 20, **base)
                if not any(x["run_id"] == r["run_id"] for x in rows):
                    rows.append(r)
    for ds in ("agnews", "sent140", "pubmed"):
        for a in (0.1, 1.0):
            for s in seeds:
                r = _mk(ds, "textcnn", "fedavg", a, s, 20, **base)
                if not any(x["run_id"] == r["run_id"] for x in rows):
                    rows.append(r)

    # p=30 : alpha sweep on the primary cells (locates the transition)
    for ds, bb in PRIMARY_CELLS:
        for a in (0.05, 0.3, 0.5):
            for s in seeds:
                rows.append(_mk(ds, bb, "fedavg", a, s, 30, **base))
    for ds in sorted({d for d, _ in PRIMARY_CELLS}):
        for a in (0.05, 0.3, 0.5):
            for s in seeds:
                rows.append(_mk(ds, "textcnn", "fedavg", a, s, 30, **base))

    # p=40 : full baseline suite at alpha = 0.1 on the diagonal
    for ds, bb in DIAGONAL:
        for algo in BASELINES:
            for s in seeds:
                extra = dict(base)
                if algo == "fedprox":
                    extra["fedprox_mu"] = 0.01
                if algo == "ditto":
                    extra["ditto_lambda"] = 1.0
                if algo == "qfedavg":
                    extra["qfedavg_q"] = 1.0
                rows.append(_mk(ds, bb, algo, 0.1, s, 40, **extra))

    # p=50 : method candidates
    for ds, bb in DIAGONAL + CORNERS[:1]:
        for algo in METHODS:
            for s in seeds:
                extra = dict(base)
                extra["ditto_lambda"] = 1.0
                extra["ours_warmup_rounds"] = 5 if algo == "ours_c" else 0
                rows.append(_mk(ds, bb, algo, 0.1, s, 50, **extra))

    # p=60 : 10-seed primary endpoint (method vs Ditto)
    for ds, bb in DIAGONAL:
        for algo in ("ours_a", "ditto"):
            for s in primary_seeds:
                extra = dict(base)
                extra["ditto_lambda"] = 1.0
                r = _mk(ds, bb, algo, 0.1, s, 60, **extra)
                if not any(x["run_id"] == r["run_id"] for x in rows):
                    rows.append(r)

    if include_appendix:
        # matched gradient steps (the E=5 vs E=1 confound in the accepted paper)
        for ds, bb in DIAGONAL:
            for m in (bb, "textcnn"):
                for s in seeds[:3]:
                    rows.append(_mk(ds, m, "fedavg", 0.1, s, 70,
                                    **{**base, "matched_compute": "steps",
                                       "target_steps": 60, "tag": "mstep"}))
        # full-size control: is the effect an artefact of subsampling to 20k?
        # MUST use a grid backbone, or it controls for nothing.
        for s in seeds[:3]:
            rows.append(_mk("agnews", "roberta", "fedavg", 0.1, s, 70,
                            **{**base, "n_train": 120000, "tag": "fullsize"}))
            rows.append(_mk("agnews", "textcnn", "fedavg", 0.1, s, 70,
                            **{**base, "n_train": 120000, "tag": "fullsize"}))
        # K = 50 cross-device realism (also makes the P10 metric meaningful)
        for s in seeds[:3]:
            rows.append(_mk("agnews", "roberta", "fedavg", 0.1, s, 70,
                            **{**base, "n_clients": 50, "tag": "K50"}))
        # natural (LEAF-style) partition by Twitter user; alpha < 0 triggers it
        for s in seeds[:3]:
            for m in ("roberta", "bertweet", "textcnn"):
                rows.append(_mk("sent140", m, "fedavg", -1.0, s, 70,
                                **{**base, "n_clients": 50, "tag": "natural"}))
        # LoRA rank ablation, on a grid backbone
        for s in seeds[:3]:
            for r_ in (4, 16):
                rows.append(_mk("agnews", "roberta", "fedavg", 0.1, s, 70,
                                **{**base, "lora_r": r_, "tag": f"rank{r_}"}))
        # DistilBERT + FedAvgW: explicit continuity with the accepted paper
        for s in seeds[:3]:
            for a in (0.1, 1.0):
                rows.append(_mk("agnews", "distilbert", "fedavg", a, s, 70,
                                **{**base, "tag": "continuity"}))
            for b_ in (0.1, 0.5):
                rows.append(_mk("agnews", "distilbert", "fedavgw", 0.1, s, 70,
                                **{**base, "fedavgw_beta": b_, "tag": f"beta{b_}"}))

    df = pd.DataFrame(rows).drop_duplicates(subset="run_id").reset_index(drop=True)
    df["status"] = "pending"
    df["error"] = ""
    df["wall_s"] = 0.0
    df["last_round"] = 0
    return df.sort_values(["priority", "run_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, root: str, cache_dir: str, manifest_path: str,
                 checkpoint_every: int = 5, verbose: bool = True):
        self.root = root
        self.cache_dir = cache_dir
        self.manifest_path = manifest_path
        self.results = os.path.join(root, "results.jsonl")
        self.ckpt_dir = os.path.join(root, "checkpoints")
        self.part_dir = os.path.join(root, "partitions")
        self.checkpoint_every = checkpoint_every
        self.verbose = verbose
        for d in (root, self.ckpt_dir, self.part_dir, cache_dir):
            os.makedirs(d, exist_ok=True)
        self._data_cache: Dict[str, tuple] = {}

    # ---- manifest ----
    TEXT_COLS = ("run_id", "status", "error", "tag", "dataset", "model",
                 "algo", "optimizer", "matched_compute")

    def load_manifest(self) -> pd.DataFrame:
        df = pd.read_csv(self.manifest_path)
        # CSV round-trips turn all-empty text columns into float64, which then
        # rejects string writes. Pin them to object up front.
        for c in self.TEXT_COLS:
            if c in df.columns:
                df[c] = df[c].astype(object).where(df[c].notna(), "")
        return df

    def save_manifest(self, df: pd.DataFrame) -> None:
        tmp = self.manifest_path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.manifest_path)

    def set_status(self, run_id: str, **fields) -> None:
        df = self.load_manifest()
        mask = df.run_id == run_id
        for k, v in fields.items():
            if k not in df.columns:
                df[k] = "" if isinstance(v, str) else np.nan
            if isinstance(v, str) and df[k].dtype != object:
                df[k] = df[k].astype(object)
            df.loc[mask, k] = v
        self.save_manifest(df)

    # ---- data / partition ----
    def get_data(self, cfg):
        from .data import cache_path_for, load_cached
        path = cache_path_for(cfg, self.cache_dir)
        if path not in self._data_cache:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing frozen cache {path}. Run Section 2 (data prep) first.")
            self._data_cache[path] = load_cached(path)
        return self._data_cache[path]

    def get_partition(self, cfg, labels, groups):
        from .partition import build_partition
        import json
        name = f"{cfg.dataset}_a{cfg.alpha}_s{cfg.seed}_K{cfg.n_clients}_n{cfg.n_train}.json"
        p = os.path.join(self.part_dir, name)
        if os.path.exists(p):
            return json.load(open(p))
        part = build_partition(np.asarray(labels), cfg.n_clients, cfg.alpha,
                               cfg.seed, groups=np.asarray(groups))
        atomic_write_json(p, part)
        return part

    # ---- one run ----
    def run_one(self, cfg: RunConfig, time_budget_s: Optional[float] = None) -> Dict:
        import torch
        from .fedcore import run_federated
        from .models import build_model

        set_seed(cfg.seed)
        device = get_device()
        train, gtest, groups, meta = self.get_data(cfg)
        n_classes = int(meta["n_classes"])
        vocab_size = int(train["input_ids"].max().item()) + 2

        part = self.get_partition(cfg, train["labels"].numpy(), groups)
        rng = np.random.default_rng(cfg.seed)
        n_g = len(gtest["labels"])
        gidx = np.sort(rng.permutation(n_g)[:min(cfg.global_eval_n, n_g)])

        ckpt_path = os.path.join(self.ckpt_dir, f"{cfg.run_id}.pt")
        resume = None
        if os.path.exists(ckpt_path):
            try:
                resume = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if self.verbose:
                    print(f"    resuming from round {resume['round']}")
            except Exception:
                resume = None

        def log_fn(row):
            flat = {"run_id": cfg.run_id, "dataset": cfg.dataset, "model": cfg.model,
                    "algo": cfg.algo, "alpha": cfg.alpha, "seed": cfg.seed,
                    "tag": cfg.tag, "round": row["round"],
                    "mean_loss": row.get("mean_loss"), "time_s": row.get("time_s")}
            for m in ("local_macro_f1", "global_bal_acc", "clientclass_f1"):
                if m in row:
                    for k, v in row[m].items():
                        flat[f"{m}__{k}"] = v
            pr = row.get("probe", [])
            if pr:
                cos = [p.get("cos_with_agg") for p in pr if p.get("cos_with_agg") is not None]
                flat["conflict_rate"] = float(np.mean([p.get("conflict", 0) for p in pr]))
                flat["cos_mean"] = float(np.mean(cos)) if cos else float("nan")
                flat["cos_minority"] = pr[int(np.argmin([p["n_train"] for p in pr]))].get("cos_with_agg")
            flat["per_client"] = row.get("per_client")
            flat["probe"] = pr
            append_jsonl(self.results, [flat])

        def ckpt_fn(state):
            atomic_write_torch(ckpt_path, state)

        t0 = time.time()
        out = run_federated(
            cfg, train, gtest, part, device,
            model_factory=lambda: build_model(cfg, vocab_size, n_classes),
            gtest_idx=gidx, resume=resume, checkpoint_fn=ckpt_fn, log_fn=log_fn,
            checkpoint_every=self.checkpoint_every, time_budget_s=time_budget_s)
        out["wall_s"] = time.time() - t0
        if out["completed"] and os.path.exists(ckpt_path):
            os.remove(ckpt_path)   # free Drive space once a run is finished
        return out

    # ---- the queue ----
    def run_pending(self, max_hours: float = 10.0, max_runs: int = 10 ** 6,
                    priority_max: int = 10 ** 6) -> pd.DataFrame:
        t_start = time.time()
        budget = max_hours * 3600
        n_done = 0
        # Reclaim rows stranded as 'running' by a session that died mid-run.
        # Only one session runs a manifest at a time, so anything still marked
        # 'running' when a new session starts is orphaned, not active. Its
        # per-round checkpoint survives on Drive, so it resumes where it stopped.
        _df = self.load_manifest()
        _stale = _df.status == "running"
        if _stale.any():
            print(f"Reclaiming {int(_stale.sum())} run(s) stranded by a dead session.")
            _df.loc[_stale, "status"] = "pending"
            self.save_manifest(_df)
        while n_done < max_runs:
            df = self.load_manifest()
            todo = df[(df.status.isin(["pending", "interrupted"]))
                      & (df.priority <= priority_max)]
            if todo.empty:
                print("No pending runs at this priority.")
                break
            elapsed = time.time() - t_start
            if elapsed > budget * 0.92:
                print(f"Session budget reached ({elapsed/3600:.2f} h). Stopping cleanly.")
                break

            row = todo.iloc[0]
            cfg = RunConfig.from_dict(row.to_dict())
            cfg.run_id = row.run_id
            self.set_status(cfg.run_id, status="running")
            if self.verbose:
                print(f"[{n_done+1}] p{row.priority} {cfg.run_id}")
            try:
                out = self.run_one(cfg, time_budget_s=max(60.0, budget - elapsed - 120))
                self.set_status(cfg.run_id,
                                status="done" if out["completed"] else "interrupted",
                                wall_s=round(out["wall_s"], 1),
                                last_round=out["last_round"], error="")
                if self.verbose:
                    h = out["history"][-1] if out["history"] else {}
                    p1 = h.get("local_macro_f1", {})
                    print(f"    round {out['last_round']}  "
                          f"P1 min={p1.get('min', float('nan')):.3f} "
                          f"mean={p1.get('mean', float('nan')):.3f}  "
                          f"{out['wall_s']/60:.1f} min")
                if not out["completed"]:
                    print("    (paused mid-run; will resume next session)")
                    break
            except Exception as e:
                self.set_status(cfg.run_id, status="failed",
                                error=f"{type(e).__name__}: {e}"[:400])
                print(f"    FAILED {type(e).__name__}: {e}")
                traceback.print_exc()
            n_done += 1
        return self.load_manifest()

    def progress(self) -> pd.DataFrame:
        df = self.load_manifest()
        g = (df.groupby(["priority", "status"]).size().unstack(fill_value=0))
        g["total"] = g.sum(axis=1)
        return g
