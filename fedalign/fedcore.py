"""fedalign.fedcore — local training, aggregation algorithms, and the round loop.

Algorithms implemented (all share one interface):
  bounds      local_only, centralized
  standard    fedavg, fedprox, scaffold
  fairness    qfedavg, ditto
  fed-LoRA    ffa_lora, fedavgw  (fedavgw kept only for continuity with the
                                  accepted paper; it is not a contribution)
  ours        ours_a  conflict-gated dual adapters   (favoured a priori)
              ours_b  subspace-protected aggregation
              ours_c  alignment-conditioned local warm-up

Personalised algorithms (local_only, ditto, ours_a) evaluate each client on its
OWN model; the rest evaluate every client on the shared global model. This
distinction is recorded so tables never mix the two silently.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import load_rng_state, rng_state
from .metrics import (aggregates, client_class_macro_f1, global_balanced_accuracy,
                      local_macro_f1)

PERSONALISED = {"local_only", "ditto", "ours_a"}
StateDict = Dict[str, torch.Tensor]


# --------------------------------------------------------------------------
# State-dict arithmetic
# --------------------------------------------------------------------------

def sd_clone(a: StateDict) -> StateDict:
    return {k: v.detach().clone() for k, v in a.items()}


def sd_sub(a: StateDict, b: StateDict) -> StateDict:
    return {k: a[k].float() - b[k].float() for k in a}


def sd_add(a: StateDict, b: StateDict, scale: float = 1.0) -> StateDict:
    return {k: a[k].float() + scale * b[k].float() for k in a}


def sd_scale(a: StateDict, s: float) -> StateDict:
    return {k: a[k].float() * s for k in a}


def sd_weighted_sum(states: List[StateDict], weights) -> StateDict:
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    out = {k: torch.zeros_like(states[0][k], dtype=torch.float32) for k in states[0]}
    for wi, st in zip(w, states):
        for k in out:
            out[k] += float(wi) * st[k].float()
    return out


def sd_flat(a: StateDict, keys: Optional[List[str]] = None) -> torch.Tensor:
    keys = keys if keys is not None else sorted(a.keys())
    return torch.cat([a[k].float().reshape(-1) for k in keys])


def sd_dot(a: StateDict, b: StateDict, keys=None) -> float:
    return float(torch.dot(sd_flat(a, keys), sd_flat(b, keys)))


def sd_norm(a: StateDict, keys=None) -> float:
    return float(torch.norm(sd_flat(a, keys)))


def cosine(a: StateDict, b: StateDict, keys=None) -> float:
    na, nb = sd_norm(a, keys), sd_norm(b, keys)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return sd_dot(a, b, keys) / (na * nb)


def lora_keys(state: StateDict) -> List[str]:
    return sorted([k for k in state if "lora_" in k])


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def iterate_batches(data, idx: np.ndarray, batch_size: int, shuffle: bool,
                    generator: Optional[torch.Generator] = None):
    idx = np.asarray(idx)
    if shuffle:
        if generator is not None:
            perm = torch.randperm(len(idx), generator=generator).numpy()
        else:
            perm = np.random.permutation(len(idx))
        idx = idx[perm]
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        if len(b) == 0:
            continue
        yield (data["input_ids"][b], data["attention_mask"][b], data["labels"][b])


def make_optimizer(params, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9,
                               weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


# --------------------------------------------------------------------------
# Local training
# --------------------------------------------------------------------------

def local_train(model, cfg, data, train_idx, device,
                global_state: Optional[StateDict] = None,
                prox_mu: float = 0.0,
                scaffold: Optional[Tuple[StateDict, StateDict]] = None,
                max_steps: Optional[int] = None,
                epochs: Optional[int] = None,
                generator: Optional[torch.Generator] = None) -> Dict:
    """One client's local update. Returns loss and step count."""
    model.train()
    opt = make_optimizer(model.trainable_parameters(), cfg)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_epochs = epochs if epochs is not None else cfg.local_epochs
    gref = {k: v.to(device) for k, v in global_state.items()} if (
        prox_mu > 0 and global_state is not None) else None

    total_loss, n_steps, n_seen = 0.0, 0, 0
    stop = False
    for _ in range(n_epochs):
        if stop:
            break
        for xb, mb, yb in iterate_batches(data, train_idx, cfg.batch_size,
                                          shuffle=True, generator=generator):
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(xb, mb)
                loss = F.cross_entropy(logits, yb)

            if gref is not None:  # FedProx proximal term (fp32, outside autocast)
                prox = 0.0
                for n, p in model.named_parameters():
                    if p.requires_grad and n in gref:
                        prox = prox + torch.sum((p.float() - gref[n]) ** 2)
                loss = loss + 0.5 * prox_mu * prox

            scaler.scale(loss).backward()
            # Unscale before touching .grad, so SCAFFOLD correction and clipping
            # operate on true gradients rather than loss-scaled ones.
            scaler.unscale_(opt)

            if scaffold is not None:  # SCAFFOLD variance correction
                c_global, c_local = scaffold
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None and n in c_global:
                        p.grad.add_(c_global[n].to(device) - c_local[n].to(device))

            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                               cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.item()) * len(yb)
            n_seen += len(yb)
            n_steps += 1
            if max_steps is not None and n_steps >= max_steps:
                stop = True
                break

    return {"loss": total_loss / max(1, n_seen), "steps": n_steps, "n_seen": n_seen}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def predict(model, data, idx, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    use_amp = (device.type == "cuda")
    for xb, mb, _ in iterate_batches(data, idx, batch_size, shuffle=False):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(xb.to(device), mb.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


def evaluate_client(model, data, gdata, client, gtest_idx, device,
                    global_pred=None) -> Dict[str, float]:
    """The three per-client metrics: P1 local macro-F1, P2 global balanced acc, P3 client-class F1.

    `global_pred` lets the caller supply predictions on the global test set that
    were computed once. Under non-personalised algorithms every client holds the
    identical global model, so recomputing them per client is pure waste.
    """
    out = {}
    te = np.asarray(client["test"], dtype=np.int64)
    if len(te) > 0:
        yp = predict(model, data, te, device)
        yt = data["labels"][te].numpy()
        out["local_macro_f1"] = local_macro_f1(yt, yp)
        out["local_acc"] = float((yt == yp).mean())
    else:
        out["local_macro_f1"] = float("nan")
        out["local_acc"] = float("nan")

    gp = global_pred if global_pred is not None else predict(model, gdata, gtest_idx, device)
    gt = gdata["labels"][gtest_idx].numpy()
    out["global_bal_acc"] = global_balanced_accuracy(gt, gp)
    out["clientclass_f1"] = client_class_macro_f1(gt, gp, client["labels_present"])
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate(algo: str, cfg, global_state: StateDict,
              client_states: List[StateDict], sizes: List[int],
              losses: List[float], minority_idx: int) -> StateDict:
    n = np.asarray(sizes, dtype=np.float64)

    if algo in ("fedavg", "fedprox", "scaffold", "ffa_lora", "ditto",
                "ours_a", "ours_c"):
        return sd_weighted_sum(client_states, n)

    if algo == "fedavgw":
        # Inverse-size weighting on LoRA params only; FedAvg elsewhere.
        lk = set(lora_keys(global_state))
        w_inv = (1.0 / np.maximum(n, 1.0)) ** cfg.fedavgw_beta
        w_inv = w_inv / w_inv.sum()
        w_std = n / n.sum()
        out = {}
        for k in global_state:
            w = w_inv if k in lk else w_std
            out[k] = sum(float(wi) * st[k].float()
                         for wi, st in zip(w, client_states))
        return out

    if algo == "fedavg_wscale":
        # Mechanism intervention (manuscript Section 5.4). FedAvg weights,
        # except the smallest client's weight is multiplied by
        # cfg.minority_weight_mult before renormalisation.
        #   mult = 1.0  -> exactly FedAvg (verified numerically)
        #   mult = 0.0  -> the smallest client still trains and is still
        #                  evaluated, but contributes nothing to the aggregate
        # Two accounts survive the manuscript's measurements: the aggregate
        # carries information the smallest client cannot obtain locally, and
        # the backbone governs how efficiently it absorbs that information.
        # Only a manipulation of how much aggregate reaches that client can
        # separate them from correlation.
        w = n.copy()
        w[minority_idx] *= float(cfg.minority_weight_mult)
        if w.sum() <= 0:
            raise ValueError("aggregation weights collapsed to zero")
        return sd_weighted_sum(client_states, w)

    if algo == "qfedavg":
        # Li et al. (2020b). Deltas scaled by loss^q, normalised by Lipschitz proxy.
        q, lr = cfg.qfedavg_q, max(cfg.lr, 1e-12)
        deltas, hs = [], []
        for st, L in zip(client_states, losses):
            dw = sd_scale(sd_sub(global_state, st), 1.0 / lr)
            Lq = max(float(L), 1e-8) ** q
            deltas.append(sd_scale(dw, Lq))
            hs.append(q * (max(float(L), 1e-8) ** (q - 1)) * (sd_norm(dw) ** 2)
                      + Lq / lr)
        h = max(sum(hs), 1e-12)
        acc = {k: torch.zeros_like(global_state[k], dtype=torch.float32)
               for k in global_state}
        for d in deltas:
            for k in acc:
                acc[k] += d[k]
        return {k: global_state[k].float() - acc[k] / h for k in global_state}

    if algo == "ours_b":
        # Subspace-protected aggregation: remove the component of the aggregate
        # update that directly opposes the minority client's update.
        agg = sd_weighted_sum(client_states, n)
        lk = lora_keys(global_state)
        if not lk:
            return agg
        d_agg = sd_sub(agg, global_state)
        d_min = sd_sub(client_states[minority_idx], global_state)
        if sd_dot(d_agg, d_min, lk) < 0:
            u_norm = sd_norm(d_min, lk)
            if u_norm > 1e-12:
                proj = sd_dot(d_agg, d_min, lk) / (u_norm ** 2)
                for k in lk:
                    d_agg[k] = d_agg[k] - proj * d_min[k].float()
                return {k: (global_state[k].float() + d_agg[k]) if k in d_agg
                        else agg[k] for k in global_state}
        return agg

    if algo == "local_only":
        return sd_clone(global_state)

    raise ValueError(f"unknown algorithm: {algo}")


# --------------------------------------------------------------------------
# The round loop
# --------------------------------------------------------------------------

def run_federated(cfg, data, gdata, partition, device,
                  model_factory: Callable,
                  gtest_idx: np.ndarray,
                  resume: Optional[Dict] = None,
                  checkpoint_fn: Optional[Callable] = None,
                  log_fn: Optional[Callable] = None,
                  checkpoint_every: int = 5,
                  time_budget_s: Optional[float] = None) -> Dict:
    """Run one federated experiment, resumable at round granularity."""
    clients = partition["clients"]
    K = len(clients)
    sizes = [max(1, c["n_train"]) for c in clients]
    minority_idx = int(np.argmin(sizes))
    personalised = cfg.algo in PERSONALISED
    # ours_c warms up the smallest 20% of clients (at least one) before they join.
    n_warm = max(1, int(round(0.2 * K)))
    warmup_clients = set(int(i) for i in np.argsort(sizes)[:n_warm])

    model = model_factory().to(device)
    init_state = model.get_trainable_state()

    gen = torch.Generator().manual_seed(cfg.seed * 7919 + 13)

    if resume:
        start_round = int(resume["round"]) + 1
        global_state = {k: v for k, v in resume["global_state"].items()}
        personal = resume.get("personal")
        local_cache = resume.get("local_cache") or [sd_clone(init_state) for _ in range(K)]
        c_global = resume.get("c_global")
        c_local = resume.get("c_local")
        lambdas = resume.get("lambdas", [cfg.ditto_lambda] * K)
        history = resume.get("history", [])
        # Restore randomness, otherwise an interrupted run diverges from an
        # uninterrupted one and the seed stops being reproducible.
        if resume.get("gen_state") is not None:
            gen.set_state(resume["gen_state"])
        load_rng_state(resume.get("rng_state"))
    else:
        start_round = 1
        global_state = init_state
        personal = [sd_clone(init_state) for _ in range(K)] if personalised else None
        local_cache = [sd_clone(init_state) for _ in range(K)]
        c_global = {k: torch.zeros_like(v) for k, v in init_state.items()} \
            if cfg.algo == "scaffold" else None
        c_local = [{k: torch.zeros_like(v) for k, v in init_state.items()}
                   for _ in range(K)] if cfg.algo == "scaffold" else None
        lambdas = [cfg.ditto_lambda] * K
        history = []
    t_start = time.time()
    stopped_early = False

    for rnd in range(start_round, cfg.rounds + 1):
        round_t0 = time.time()
        client_states, losses, steps_used, probe_rows = [], [], [], []

        # ---- matched-compute: equalise gradient steps across model classes ----
        max_steps = cfg.target_steps if (cfg.matched_compute == "steps"
                                         and cfg.target_steps > 0) else None

        for k, c in enumerate(clients):
            tr = np.asarray(c["train"], dtype=np.int64)
            if len(tr) == 0:
                client_states.append(sd_clone(global_state))
                losses.append(float("nan"))
                steps_used.append(0)
                continue

            # ours_c: low-alignment minority clients warm up locally before joining
            warmup_hold = (cfg.algo == "ours_c" and rnd <= cfg.ours_warmup_rounds
                           and k in warmup_clients)
            if cfg.algo == "local_only" or warmup_hold:
                start_state = local_cache[k]
            else:
                start_state = global_state

            model.set_trainable_state(start_state)
            sc = (c_global, c_local[k]) if cfg.algo == "scaffold" else None
            info = local_train(model, cfg, data, tr, device,
                               global_state=global_state,
                               prox_mu=cfg.fedprox_mu if cfg.algo == "fedprox" else 0.0,
                               scaffold=sc, max_steps=max_steps, generator=gen)
            new_state = model.get_trainable_state()
            client_states.append(new_state)
            local_cache[k] = sd_clone(new_state)
            losses.append(info["loss"])
            steps_used.append(info["steps"])

            if cfg.algo == "scaffold" and info["steps"] > 0:
                # c_i+ = c_i - c + (w_global - w_i) / (steps * lr)
                denom = max(info["steps"] * cfg.lr, 1e-12)
                dw = sd_scale(sd_sub(global_state, new_state), 1.0 / denom)
                c_local[k] = {kk: c_local[k][kk].float() - c_global[kk].float() + dw[kk]
                              for kk in c_local[k]}

            probe_rows.append({"client": k, "n_train": len(tr),
                               "loss": info["loss"], "steps": info["steps"]})

        # ---- aggregate ----
        new_global = aggregate(cfg.algo, cfg, global_state, client_states,
                               sizes, losses, minority_idx)

        if cfg.algo == "scaffold" and c_global is not None:
            c_global = {k: c_global[k].float()
                        + sum((c_local[i][k].float() - c_global[k].float())
                              for i in range(K)) / K
                        for k in c_global}

        # ---- mechanism probe: conflict between each client and the aggregate ----
        d_agg = sd_sub(new_global, global_state)
        lk = lora_keys(global_state) or None
        agg_norm = sd_norm(d_agg, lk)
        for k in range(K):
            d_k = sd_sub(client_states[k], global_state)
            cs = cosine(d_k, d_agg, lk)
            if k < len(probe_rows):
                probe_rows[k].update({
                    "cos_with_agg": cs,
                    "conflict": int(cs < 0),
                    "update_norm": sd_norm(d_k, lk),
                    "agg_norm": agg_norm,
                })

        # ---- personalised branches ----
        if cfg.algo in ("ditto", "ours_a"):
            for k, c in enumerate(clients):
                tr = np.asarray(c["train"], dtype=np.int64)
                if len(tr) == 0:
                    continue
                if cfg.algo == "ours_a":
                    # Conflict-gated: high conflict -> weaker pull toward the global
                    # model, so the minority client's adaptation survives aggregation.
                    cs = probe_rows[k].get("cos_with_agg", 0.0)
                    gate = 0.5 * (1.0 + cs)
                    lambdas[k] = float(np.clip(cfg.ditto_lambda * gate,
                                               0.05 * cfg.ditto_lambda,
                                               cfg.ditto_lambda))
                model.set_trainable_state(personal[k])
                local_train(model, cfg, data, tr, device,
                            global_state=new_global, prox_mu=lambdas[k],
                            epochs=cfg.ditto_local_epochs,
                            max_steps=max_steps, generator=gen)
                personal[k] = model.get_trainable_state()
        elif cfg.algo == "local_only":
            personal = [sd_clone(s) for s in client_states]

        global_state = new_global

        # ---- evaluate ----
        row = {"round": rnd, "time_s": time.time() - round_t0}
        do_eval = (rnd % cfg.eval_every == 0) or rnd == cfg.rounds or rnd == 1
        if do_eval:
            per_client = []
            # Non-personalised algorithms give every client the same model, so
            # the global-test predictions are identical: compute them once.
            shared_gp = None
            if not personalised:
                model.set_trainable_state(global_state)
                shared_gp = predict(model, gdata, gtest_idx, device)
            for k, c in enumerate(clients):
                if personalised and personal is not None:
                    view = personal[k]
                elif (cfg.algo == "ours_c" and rnd <= cfg.ours_warmup_rounds
                      and k in warmup_clients):
                    view = local_cache[k]   # still warming up locally
                else:
                    view = global_state
                model.set_trainable_state(view)
                gp_k = shared_gp if view is global_state else None
                m = evaluate_client(model, data, gdata, c, gtest_idx, device,
                                    global_pred=gp_k)
                m.update({"client": k, "n_train": c["n_train"],
                          "labels_present": c["labels_present"]})
                per_client.append(m)
            row["per_client"] = per_client
            for metric in ("local_macro_f1", "global_bal_acc", "clientclass_f1"):
                row[metric] = aggregates([pc[metric] for pc in per_client])
        row["probe"] = probe_rows
        row["mean_loss"] = float(np.nanmean(losses)) if losses else float("nan")
        history.append(row)

        if log_fn is not None:
            log_fn(row)

        need_ckpt = (rnd % checkpoint_every == 0) or rnd == cfg.rounds
        over_budget = time_budget_s is not None and (time.time() - t_start) > time_budget_s
        if checkpoint_fn is not None and (need_ckpt or over_budget):
            checkpoint_fn({
                "round": rnd, "global_state": global_state, "personal": personal,
                "local_cache": local_cache, "c_global": c_global, "c_local": c_local,
                "lambdas": lambdas, "history": history[-40:],
                "gen_state": gen.get_state(), "rng_state": rng_state(),
            })
        if over_budget:
            stopped_early = True
            break

    return {"history": history, "global_state": global_state,
            "personal": personal, "completed": not stopped_early,
            "last_round": history[-1]["round"] if history else 0,
            "minority_idx": minority_idx}
