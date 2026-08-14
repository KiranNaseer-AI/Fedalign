"""fedalign.data — load once, freeze forever.

Downloading and tokenising inside the training loop would waste session time,
add a network dependency to every run, and — worst — let an upstream dataset
revision silently change the data underneath a pre-registered protocol. So all
of that happens ONCE in Phase 0 and the result is frozen to disk as small .npz
arrays that every later run memory-maps.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import DATASETS, TEXTCNN_TOKENIZER, BACKBONES, atomic_write_json

# Candidate column names, since HF dataset schemas vary between mirrors.
TEXT_CANDIDATES = ["text", "sentence", "content", "sms", "tweet", "abstract_text"]
LABEL_CANDIDATES = ["label", "labels", "sentiment", "target", "class", "category"]
GROUP_CANDIDATES = ["abstract_id", "user", "user_id", "doc_id", "group", "id"]


def _load(hf_id: str):
    """load_dataset with a fallback for legacy script-based datasets.

    datasets>=4.0 refuses to execute loading scripts (Sentiment140 ships one).
    The Hub auto-converts every dataset to parquet under the
    `refs/convert/parquet` revision, so read that rather than hunt for a mirror.
    """
    from datasets import load_dataset
    try:
        return load_dataset(hf_id)
    except (RuntimeError, ValueError) as e:
        if "scripts are no longer supported" not in str(e):
            raise
    from collections import defaultdict
    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(hf_id, repo_type="dataset",
                                        revision="refs/convert/parquet")
             if f.endswith(".parquet")]
    if not files:
        raise RuntimeError(f"No parquet conversion available for {hf_id}.")
    groups = defaultdict(list)
    for f in files:
        split = f.split("/")[-2] if "/" in f else "train"
        groups[split].append(f"hf://datasets/{hf_id}@refs/convert/parquet/{f}")
    print(f"    [{hf_id}] script dataset -> parquet, "
          f"splits={ {k: len(v) for k, v in groups.items()} }")
    return load_dataset("parquet", data_files=dict(groups))


def describe_dataset(dataset_key: str, n_peek: int = 3) -> Dict:
    """Phase 0 sanity check — ALWAYS run this before trusting the loader.

    Prints the real column names and label values so schema drift is caught in
    minutes rather than after a grid launch.
    """
    spec = DATASETS[dataset_key]
    ds = _load(spec["hf_id"])
    split = "train" if "train" in ds else list(ds.keys())[0]
    cols = ds[split].column_names
    sample = ds[split][:n_peek]
    info = {
        "hf_id": spec["hf_id"],
        "splits": {k: len(v) for k, v in ds.items()},
        "columns": cols,
        "resolved_text_col": _pick(cols, TEXT_CANDIDATES),
        "resolved_label_col": _pick(cols, LABEL_CANDIDATES),
        "resolved_group_col": _pick(cols, GROUP_CANDIDATES),
        "sample": {c: sample[c] for c in cols},
    }
    return info


def _pick(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _encode_labels(values) -> Tuple[np.ndarray, Dict]:
    """Map arbitrary label values (ints or strings) to contiguous 0..C-1."""
    uniq = sorted(set(values), key=lambda v: (isinstance(v, str), v))
    mapping = {v: i for i, v in enumerate(uniq)}
    return np.array([mapping[v] for v in values], dtype=np.int64), \
        {str(k): int(v) for k, v in mapping.items()}


def load_and_freeze(
    dataset_key: str,
    tokenizer_name: str,
    cache_dir: str,
    n_train: int = 20000,
    n_test: int = 5000,
    max_len: int = 128,
    seed: int = 42,
    force: bool = False,
) -> str:
    """Download -> subsample -> tokenise -> save .npz. Returns the cache path.

    Subsampling to a common size matters: without it, dataset scale is
    confounded with domain in the 3x3 alignment grid.
    """
    tag = tokenizer_name.replace("/", "-")
    out = os.path.join(cache_dir, f"{dataset_key}__{tag}__n{n_train}_L{max_len}.npz")
    if os.path.exists(out) and not force:
        return out

    from transformers import AutoTokenizer

    spec = DATASETS[dataset_key]
    ds = _load(spec["hf_id"])
    train_split = "train" if "train" in ds else list(ds.keys())[0]
    cols = ds[train_split].column_names
    tcol = _pick(cols, TEXT_CANDIDATES)
    lcol = _pick(cols, LABEL_CANDIDATES)
    gcol = _pick(cols, GROUP_CANDIDATES)
    if tcol is None or lcol is None:
        raise RuntimeError(
            f"Could not resolve text/label columns for {dataset_key}. "
            f"Columns are {cols}. Run describe_dataset() and set them explicitly."
        )

    rng = np.random.default_rng(seed)
    tr = ds[train_split]
    idx = rng.permutation(len(tr))[:min(n_train, len(tr))]
    idx = np.sort(idx)
    tr = tr.select(idx.tolist())

    # Held-out global test set: prefer an official split, else carve from train.
    test_split = next((s for s in ("test", "validation", "dev") if s in ds), None)
    if test_split is not None and len(ds[test_split]) >= 500:
        te = ds[test_split]
        if len(te) > n_test:
            te = te.select(rng.permutation(len(te))[:n_test].tolist())
    else:
        rest = np.setdiff1d(np.arange(len(ds[train_split])), idx)
        pick = rng.permutation(rest)[:n_test]
        te = ds[train_split].select(np.sort(pick).tolist())

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def enc(split):
        texts = [str(t) for t in split[tcol]]
        e = tok(texts, truncation=True, padding="max_length",
                max_length=max_len, return_tensors="np")
        return e["input_ids"].astype(np.int32), e["attention_mask"].astype(np.int8)

    xi_tr, am_tr = enc(tr)
    xi_te, am_te = enc(te)

    y_all, mapping = _encode_labels(list(tr[lcol]) + list(te[lcol]))
    y_tr, y_te = y_all[:len(tr)], y_all[len(tr):]

    if gcol is not None:
        g_raw = list(tr[gcol])
        _, gmap = _encode_labels(g_raw)
        g_tr = np.array([gmap[str(v)] for v in g_raw], dtype=np.int64)
    else:
        g_tr = np.arange(len(tr), dtype=np.int64)  # every row its own group

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        out,
        train_input_ids=xi_tr, train_attention_mask=am_tr,
        train_labels=y_tr, train_groups=g_tr,
        test_input_ids=xi_te, test_attention_mask=am_te, test_labels=y_te,
    )
    atomic_write_json(out.replace(".npz", "_meta.json"), {
        "dataset": dataset_key, "hf_id": spec["hf_id"], "tokenizer": tokenizer_name,
        "text_col": tcol, "label_col": lcol, "group_col": gcol,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_classes": int(y_all.max() + 1), "max_len": max_len, "seed": seed,
        "label_map": mapping,
        "has_real_groups": gcol is not None,
        "n_groups": int(len(np.unique(g_tr))),
    })
    return out


def load_cached(path: str):
    """Return (train_bundle, test_bundle, groups, meta) as torch tensors."""
    import torch
    z = np.load(path)
    meta = json.load(open(path.replace(".npz", "_meta.json")))
    train = {
        "input_ids": torch.from_numpy(z["train_input_ids"].astype(np.int64)),
        "attention_mask": torch.from_numpy(z["train_attention_mask"].astype(np.int64)),
        "labels": torch.from_numpy(z["train_labels"].astype(np.int64)),
    }
    test = {
        "input_ids": torch.from_numpy(z["test_input_ids"].astype(np.int64)),
        "attention_mask": torch.from_numpy(z["test_attention_mask"].astype(np.int64)),
        "labels": torch.from_numpy(z["test_labels"].astype(np.int64)),
    }
    return train, test, z["train_groups"], meta


def cache_path_for(cfg, cache_dir: str) -> str:
    """TextCNN shares one tokenizer across datasets so tokenisation is not a confound."""
    tokname = TEXTCNN_TOKENIZER if cfg.model == "textcnn" else BACKBONES[cfg.model]["hf_id"]
    tag = tokname.replace("/", "-")
    return os.path.join(cache_dir,
                        f"{cfg.dataset}__{tag}__n{cfg.n_train}_L{cfg.max_len}.npz")
