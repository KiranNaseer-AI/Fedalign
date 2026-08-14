"""fedalign.core — configuration, seeding, atomic IO, device handling."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------

# HF ids. Verify PubMedBERT id in Phase 0 (Microsoft renamed PubMedBERT -> BiomedBERT).
BACKBONES: Dict[str, Dict[str, Any]] = {
    "distilbert": {
        "hf_id": "distilbert-base-uncased",
        "lora_targets": ["q_lin", "v_lin"],   # DistilBERT names differ from BERT!
        "family": "distilbert",
    },
    "bertweet": {
        "hf_id": "vinai/bertweet-base",
        "lora_targets": ["query", "value"],
        "family": "roberta",
    },
    "pubmedbert": {
        "hf_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "lora_targets": ["query", "value"],
        "family": "bert",
    },
    "roberta": {  # appendix scale check
        "hf_id": "roberta-base",
        "lora_targets": ["query", "value"],
        "family": "roberta",
    },

    # --- confirmatory replication set (Section 12) -------------------------
    # Added after the 3x3 grid completed, to test the perplexity rule on
    # backbones that played no part in discovering it. All four are 12-layer /
    # 768-hidden encoders with a genuine MLM head.
    #
    # ELECTRA and DeBERTa-v3 are deliberately absent. Both are replaced-token-
    # detection models: ELECTRA's discriminator has no MLM head at all (its
    # generator is a separate, smaller model) and DeBERTa-v3 uses RTD with
    # gradient-disentangled embedding sharing. The predictor is undefined for
    # them. That is a scope statement for the paper, not an omission.
    "bert": {
        "hf_id": "bert-base-uncased",
        "lora_targets": ["query", "value"],
        "family": "bert",
    },
    "biobert": {   # continued pretraining FROM BERT on PubMed
        "hf_id": "dmis-lab/biobert-base-cased-v1.1",
        "lora_targets": ["query", "value"],
        "family": "bert",
    },
    "scibert": {   # from scratch on scientific text, own vocabulary
        "hf_id": "allenai/scibert_scivocab_uncased",
        "lora_targets": ["query", "value"],
        "family": "bert",
    },
    "twitroberta": {  # continued pretraining FROM RoBERTa on tweets
        "hf_id": "cardiffnlp/twitter-roberta-base",
        "lora_targets": ["query", "value"],
        "family": "roberta",
    },
}

DATASETS: Dict[str, Dict[str, Any]] = {
    "agnews": {
        "hf_id": "fancyzhx/ag_news",
        "text_key": "text",
        "label_key": "label",
        "n_classes": 4,
        "group_key": None,          # no grouping needed
        "max_len": 128,
    },
    "sent140": {
        "hf_id": "stanfordnlp/sentiment140",
        "text_key": "text",
        "label_key": "sentiment",   # 0 / 4 -> remapped to 0 / 1
        "n_classes": 2,
        "group_key": "user",        # enables the natural (LEAF-style) partition
        "max_len": 128,
    },
    "pubmed": {
        "hf_id": "armanc/pubmed-rct20k",
        "text_key": "text",
        "label_key": "label",
        "n_classes": 5,
        "group_key": "abstract_id",  # CRITICAL: partition by abstract, not sentence
        "max_len": 128,
    },
}

# TextCNN shares one tokenizer across all datasets so that tokenisation is not a
# confound in the CNN-vs-FM comparison.
TEXTCNN_TOKENIZER = "distilbert-base-uncased"


# --------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------

_FP_SKIP_IF_DEFAULT: Dict[str, Any] = {"minority_weight_mult": 1.0}


@dataclass
class RunConfig:
    """One experimental run. Serialised into every result row."""
    run_id: str = ""
    dataset: str = "agnews"
    model: str = "distilbert"          # backbone key, or "textcnn"
    algo: str = "fedavg"
    alpha: float = 0.1                 # Dirichlet concentration; <0 => natural partition
    seed: int = 0

    # federated protocol
    n_clients: int = 10
    rounds: int = 40
    local_epochs: int = 1
    batch_size: int = 32
    eval_every: int = 2

    # optimisation
    lr: float = 5e-5
    weight_decay: float = 0.01
    optimizer: str = "adamw"           # "adamw" | "sgd"
    max_grad_norm: float = 1.0

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # algorithm-specific
    fedprox_mu: float = 0.0
    ditto_lambda: float = 1.0
    ditto_local_epochs: int = 1
    qfedavg_q: float = 1.0
    fedavgw_beta: float = 0.5
    ours_gate_init: float = 0.5
    ours_warmup_rounds: int = 0
    minority_weight_mult: float = 1.0   # Section 12 mechanism intervention

    # data scale
    n_train: int = 20000
    max_len: int = 128
    global_eval_n: int = 2000

    # bookkeeping
    matched_compute: str = "epochs"    # "epochs" | "steps"
    target_steps: int = 0              # used when matched_compute == "steps"
    priority: int = 100                # lower runs first (gate ordering)
    tag: str = ""

    def fingerprint(self) -> str:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("run_id", "priority", "tag")}
        # Fields added AFTER runs have been logged are dropped from the hash
        # while they hold their default value, so every run_id already sitting
        # in manifest.csv and results.jsonl keeps the identity it was written
        # under. Without this, adding one field with a harmless default
        # re-hashes all 313 completed runs, the manifest stops matching the
        # results log, and the queue silently re-runs the entire grid.
        # A non-default value still changes the hash, so new conditions get
        # their own ids. Never remove a name from this dict.
        for _k, _default in _FP_SKIP_IF_DEFAULT.items():
            if _k in d and d[_k] == _default:
                d.pop(_k)
        blob = json.dumps(d, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()[:10]

    def make_id(self) -> str:
        base = (f"{self.dataset}_{self.model}_{self.algo}"
                f"_a{self.alpha}_s{self.seed}_{self.matched_compute}")
        if self.tag:
            base += f"_{self.tag}"
        return f"{base}_{self.fingerprint()}"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        clean = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            clean[k] = v
        cfg = cls(**clean)
        if not cfg.run_id:
            cfg.run_id = cfg.make_id()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def load_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    try:
        import torch
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu()
                                if hasattr(state["torch"], "cpu") else state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except (ImportError, RuntimeError, TypeError):
        pass


# --------------------------------------------------------------------------
# Atomic IO  (a Drive-mounted crash mid-write must never corrupt state)
# --------------------------------------------------------------------------

def atomic_write_bytes(path: str, writer) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_json(path: str, obj: Any) -> None:
    def _w(fh):
        fh.write(json.dumps(obj, indent=2, default=_json_default).encode())
    atomic_write_bytes(path, _w)


def atomic_write_torch(path: str, obj: Any) -> None:
    import torch

    def _w(fh):
        torch.save(obj, fh)
    atomic_write_bytes(path, _w)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append-only results log. Never overwritten, so a crash cannot destroy history."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard kill
    return out


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
