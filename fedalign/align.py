"""fedalign.align — a-priori pretraining/task domain alignment (Table 7).

Computed BEFORE any federated run, then committed with a signed prediction per
cell. This is what makes H1 predictive rather than a post-hoc story, and it is
the difference between this paper and the accepted one (which invoked "domain
alignment" only afterwards, to explain away a sign flip).

Three measures, none of which requires access to the pretraining corpus:

  1. PRIMARY   masked-LM pseudo-perplexity on task text (lower = aligned).
  2. SECONDARY subword fertility: mean subword tokens per whitespace word
               (lower = aligned; Rust et al., 2021).
  3. TERTIARY  vocabulary coverage: 1 - UNK rate over task tokens
               (higher = aligned).

PROTOCOL DEVIATION (log in deviations.md): the frozen protocol listed
"embedding centroid distance" as the tertiary measure. That requires a sample
of each backbone's pretraining corpus, which is not reliably obtainable for
BERTweet or PubMedBERT; fertility and coverage are self-contained, standard,
and cheaper. Recorded as an improvement, not a silent change.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def pseudo_perplexity(backbone_key: str, texts: Sequence[str], max_len: int = 128,
                      n_samples: int = 2000, n_passes: int = 3, mask_prob: float = 0.15,
                      seed: int = 42, batch_size: int = 32,
                      device=None) -> float:
    """Random-mask pseudo-perplexity. Cheap stand-in for exhaustive PLL.

    Exhaustive pseudo-log-likelihood masks one token at a time (O(L) forward
    passes per sentence). Averaging a few random 15% masking passes is the
    standard cheap estimator and is stable to well under the between-cell
    differences we care about.
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from .core import BACKBONES

    hf_id = BACKBONES[backbone_key]["hf_id"]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(hf_id).to(device).eval()

    rng = np.random.default_rng(seed)
    texts = list(texts)
    if len(texts) > n_samples:
        texts = [texts[i] for i in rng.permutation(len(texts))[:n_samples]]

    if tok.mask_token_id is None:
        raise RuntimeError(f"{hf_id} has no mask token; cannot compute pseudo-PPL.")

    total_nll, total_tok = 0.0, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for _ in range(n_passes):
            for s in range(0, len(texts), batch_size):
                batch = texts[s:s + batch_size]
                enc = tok(batch, truncation=True, padding=True,
                          max_length=max_len, return_tensors="pt")
                ids = enc["input_ids"]
                am = enc["attention_mask"]
                special = torch.tensor(
                    [[t in tok.all_special_ids for t in row.tolist()] for row in ids])
                cand = (am.bool()) & (~special)
                probs = torch.rand(ids.shape, generator=g)
                mask = cand & (probs < mask_prob)
                if mask.sum() == 0:
                    continue
                labels = ids.clone()
                labels[~mask] = -100
                masked = ids.clone()
                masked[mask] = tok.mask_token_id
                out = model(input_ids=masked.to(device),
                            attention_mask=am.to(device),
                            labels=labels.to(device))
                n = int(mask.sum())
                total_nll += float(out.loss.item()) * n
                total_tok += n

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if total_tok == 0:
        return float("nan")
    return float(np.exp(total_nll / total_tok))


def subword_fertility(backbone_key: str, texts: Sequence[str],
                      n_samples: int = 5000, seed: int = 42) -> Dict[str, float]:
    """Mean subword tokens per whitespace word, plus UNK rate."""
    from transformers import AutoTokenizer
    from .core import BACKBONES

    tok = AutoTokenizer.from_pretrained(BACKBONES[backbone_key]["hf_id"], use_fast=True)
    rng = np.random.default_rng(seed)
    texts = list(texts)
    if len(texts) > n_samples:
        texts = [texts[i] for i in rng.permutation(len(texts))[:n_samples]]

    n_words, n_pieces, n_unk, n_tok = 0, 0, 0, 0
    unk_id = tok.unk_token_id
    for t in texts:
        words = str(t).split()
        if not words:
            continue
        ids = tok(str(t), add_special_tokens=False)["input_ids"]
        n_words += len(words)
        n_pieces += len(ids)
        n_tok += len(ids)
        if unk_id is not None:
            n_unk += sum(1 for i in ids if i == unk_id)
    return {
        "fertility": float(n_pieces / max(1, n_words)),
        "unk_rate": float(n_unk / max(1, n_tok)),
        "vocab_coverage": float(1.0 - n_unk / max(1, n_tok)),
    }


def composite_alignment(rows: List[Dict]) -> List[Dict]:
    """Alignment = the backbone x dataset INTERACTION in log per-word perplexity.

    Three corrections, each fixing a distinct confound. All were identified by
    the diagonal calibration check in Notebook 4.1, BEFORE any prediction was
    committed and before any federated run existed; see deviations.md.

    1. CROSS-TOKENIZER COMPARABILITY. Token-level perplexity is not comparable
       across vocabularies (64k BPE vs 30k WordPiece). Convert to per-WORD by
       multiplying the per-token NLL by fertility (tokens per word).

    2. TWO MAIN EFFECTS, NOT ONE. A stronger backbone has lower perplexity on
       everything (capacity), and some corpora are intrinsically less
       predictable than others (difficulty) - tweets are harder than news for
       EVERY model, however it was pretrained. Removing only the backbone
       effect leaves dataset difficulty intact and makes every backbone look
       best-aligned to the easiest corpus. So the matrix is centred on BOTH
       axes:

           residual(b,d) = M(b,d) - mean_d M(b,.) - mean_b M(.,d) + grand_mean

       This is the standard two-factor decomposition: the main effects are
       nuisance parameters and the interaction is the quantity of interest.
       "Does THIS backbone fit THIS domain better than its general quality and
       that corpus's general difficulty would predict" is exactly what domain
       alignment means.

    3. NO Z-SCORED COMPOSITE. Vocabulary coverage is ~1.000 for every cell
       (subword tokenizers essentially never emit UNK) and fertility varies by
       under 5% within a backbone. Z-scoring gave those two the same weight as
       a 300x perplexity range, so noise dominated signal. Perplexity alone is
       the alignment score; fertility and coverage are retained as descriptive
       columns in Table 7 and reported, not blended in.

    Requires a complete backbone x dataset grid: a missing cell biases both
    marginal means. The caller must refuse to commit a partial table.
    """
    if not rows:
        return rows

    for r in rows:
        r["log_ppl_word"] = float(np.log(max(r["pseudo_ppl"], 1e-9)) * r["fertility"])
        r["ppl_word"] = float(np.exp(r["log_ppl_word"]))

    datasets = sorted({r["dataset"] for r in rows})
    backbones = sorted({r["backbone"] for r in rows})
    idx = {(r["dataset"], r["backbone"]): r["log_ppl_word"] for r in rows}
    if len(idx) != len(datasets) * len(backbones):
        raise ValueError(
            f"Alignment grid is incomplete ({len(idx)} of "
            f"{len(datasets) * len(backbones)} cells). Two-way centring needs "
            f"every cell; a missing one biases the marginal means.")

    M = np.array([[idx[(d, b)] for b in backbones] for d in datasets], dtype=float)
    residual = (M - M.mean(axis=1, keepdims=True)
                  - M.mean(axis=0, keepdims=True) + M.mean())

    for r in rows:
        i, j = datasets.index(r["dataset"]), backbones.index(r["backbone"])
        r["alignment_score"] = float(-residual[i, j])       # higher = better aligned
        r["dataset_difficulty"] = float(M[i, :].mean())     # main effect, reported
        r["backbone_quality"] = float(-M[:, j].mean())      # main effect, reported
        r["alignment_raw"] = float(-r["log_ppl_word"])      # uncentred, reported
    return rows


def predict_sign(alignment_score: float, low: float = -0.35, high: float = 0.35) -> str:
    """Pre-committed prediction of the FM's worst-client effect for a cell.

    Effect is defined as (FM worst-client metric - TextCNN worst-client metric):
    positive = the FM protects the worst client, negative = the FM harms it.
    Thresholds are set once, here, before any run.
    """
    if alignment_score >= high:
        return "protective"
    if alignment_score <= low:
        return "harmful"
    return "neutral"
