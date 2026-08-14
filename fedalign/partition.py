"""fedalign.partition — Non-IID partitioning and client-local splits.

Two independent concerns, deliberately separated:

1. *Federated partition*: how the training pool is split across clients.
   Label-Dirichlet (Hsu et al., 2019) for the alpha sweep, or a natural
   grouping (LEAF-style, e.g. by Twitter user) when alpha < 0.

2. *Client-local split*: how each client's own shard is split into
   train/val/test. This is group-aware: when a group key exists (e.g.
   PubMed abstract id), a whole group lands entirely in one split, so
   correlated sentences from the same abstract can never straddle a
   client's train and test sets. Skipping this inflates every number.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
    max_tries: int = 200,
) -> List[np.ndarray]:
    """Label-skew partition. Smaller alpha => more extreme heterogeneity.

    Retries until every client holds at least `min_size` samples, which keeps
    local train/val/test splits well defined at alpha = 0.05.
    """
    labels = np.asarray(labels)
    n_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    for attempt in range(max_tries):
        idx_per_client: List[List[int]] = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                idx_per_client[k].extend(part.tolist())
        sizes = [len(v) for v in idx_per_client]
        if min(sizes) >= min_size:
            break
    else:
        # Guarantee viability rather than loop forever at very small alpha.
        idx_per_client = _rescue_min_size(idx_per_client, min_size, rng)

    out = []
    for v in idx_per_client:
        arr = np.array(sorted(v), dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def _rescue_min_size(idx_per_client, min_size, rng) -> List[List[int]]:
    """Move samples from the largest client to any client below min_size."""
    idx_per_client = [list(v) for v in idx_per_client]
    for _ in range(1000):
        sizes = [len(v) for v in idx_per_client]
        lo = int(np.argmin(sizes))
        if sizes[lo] >= min_size:
            break
        hi = int(np.argmax(sizes))
        if sizes[hi] <= min_size:
            break
        need = min_size - sizes[lo]
        take = idx_per_client[hi][:need]
        idx_per_client[hi] = idx_per_client[hi][need:]
        idx_per_client[lo].extend(take)
    return idx_per_client


def natural_partition(
    groups: Sequence,
    n_clients: int,
    seed: int,
    min_size: int = 10,
) -> List[np.ndarray]:
    """LEAF-style partition: sample `n_clients` real groups (e.g. Twitter users).

    Groups are sampled from the largest ones so that each client is trainable,
    then indices are taken whole. Answers the 'Dirichlet is synthetic' objection.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)
    uniq, counts = uniq[order], counts[order]
    eligible = uniq[counts >= min_size]
    if len(eligible) < n_clients:
        eligible = uniq[:n_clients]
    rng = np.random.default_rng(seed)
    pool = eligible[: max(n_clients * 5, n_clients)]
    chosen = rng.choice(pool, size=n_clients, replace=False)
    return [np.where(groups == g)[0].astype(np.int64) for g in chosen]


def client_local_split(
    indices: np.ndarray,
    labels: np.ndarray,
    seed: int,
    groups: Optional[np.ndarray] = None,
    fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    min_test: int = 2,
) -> Dict[str, np.ndarray]:
    """Split one client's shard into train/val/test.

    Group-aware when `groups` is given (whole group -> one split).
    Stratified by label otherwise, so rare local classes appear in the test split.
    """
    indices = np.asarray(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n = len(indices)
    if n < 4:  # degenerate client: everything goes to train, evaluation skipped
        return {"train": indices, "val": indices[:0], "test": indices[:0]}

    if groups is not None:
        g = np.asarray(groups)[indices]
        uniq = np.unique(g)
        rng.shuffle(uniq)
        n_g = len(uniq)
        n_tr = max(1, int(round(fracs[0] * n_g)))
        n_va = max(0, int(round(fracs[1] * n_g)))
        n_tr = min(n_tr, max(1, n_g - 1))
        n_va = min(n_va, max(0, n_g - n_tr - 1))
        sets = {
            "train": uniq[:n_tr],
            "val": uniq[n_tr:n_tr + n_va],
            "test": uniq[n_tr + n_va:],
        }
        out = {k: indices[np.isin(g, v)] for k, v in sets.items()}
    else:
        y = np.asarray(labels)[indices]
        tr, va, te = [], [], []
        for c in np.unique(y):
            pos = np.where(y == c)[0]
            rng.shuffle(pos)
            m = len(pos)
            n_tr = max(1, int(round(fracs[0] * m)))
            n_va = int(round(fracs[1] * m))
            if m >= 3:
                n_tr = min(n_tr, m - 2)
                n_va = min(n_va, m - n_tr - 1)
            else:
                n_tr, n_va = max(1, m - 1), 0
            tr.extend(indices[pos[:n_tr]])
            va.extend(indices[pos[n_tr:n_tr + n_va]])
            te.extend(indices[pos[n_tr + n_va:]])
        out = {"train": np.array(tr, dtype=np.int64),
               "val": np.array(va, dtype=np.int64),
               "test": np.array(te, dtype=np.int64)}

    # Guard: a client with no test samples cannot contribute to the primary metric.
    if len(out["test"]) < min_test and len(out["train"]) > min_test * 2:
        move = out["train"][:min_test]
        out["train"] = out["train"][min_test:]
        out["test"] = np.concatenate([out["test"], move])
    return out


def build_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
    groups: Optional[np.ndarray] = None,
    group_aware_splits: bool = True,
) -> Dict:
    """Full partition record: shards, local splits, and per-client label sets.

    Serialised to disk per (dataset, alpha, seed) so every run is reproducible
    and post-hoc analysis can recover exactly who held what.
    """
    if alpha is not None and alpha < 0:
        if groups is None:
            raise ValueError("natural partition requested but dataset has no group key")
        shards = natural_partition(groups, n_clients, seed)
    else:
        shards = dirichlet_partition(labels, n_clients, alpha, seed)

    split_groups = groups if (group_aware_splits and groups is not None) else None
    clients = []
    for k, shard in enumerate(shards):
        sp = client_local_split(shard, labels, seed=seed * 1000 + k, groups=split_groups)
        present = np.unique(np.asarray(labels)[shard]).astype(int).tolist()
        clients.append({
            "client_id": k,
            "n_total": int(len(shard)),
            "n_train": int(len(sp["train"])),
            "n_val": int(len(sp["val"])),
            "n_test": int(len(sp["test"])),
            "labels_present": present,
            "train": sp["train"].tolist(),
            "val": sp["val"].tolist(),
            "test": sp["test"].tolist(),
        })
    sizes = [c["n_total"] for c in clients]
    return {
        "alpha": alpha,
        "seed": seed,
        "n_clients": n_clients,
        "clients": clients,
        "size_min": int(min(sizes)),
        "size_max": int(max(sizes)),
        "size_ratio": float(max(sizes) / max(1, min(sizes))),
    }
