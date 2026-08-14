"""fedalign.models — TextCNN baseline and LoRA-adapted foundation models.

LoRA is implemented directly (Hu et al., 2022) rather than via `peft`. Reasons:
  * no version-compatibility surprises inside Colab,
  * FFA-LoRA (freeze A, train only B) becomes a one-line flag,
  * adapter deltas are extracted exactly, which the mechanism probe needs,
  * classifier heads are uniform across backbones, so `modules_to_save`
    differences between DistilBERT / BERT / RoBERTa cannot silently leave a
    randomly-initialised head frozen.

Every model exposes the same interface used by the federated loop:
    get_trainable_state() / set_trainable_state() / trainable_parameters()
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import BACKBONES


# --------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = W0 x + (alpha/r) * B A x, with W0 frozen, B initialised to zero."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)   # delta W = 0 at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + h * self.scaling

    def delta_w(self) -> torch.Tensor:
        """The effective weight update, for subspace analysis."""
        return (self.lora_B @ self.lora_A) * self.scaling


def inject_lora(module: nn.Module, targets: List[str], r: int,
                alpha: int, dropout: float) -> int:
    """Replace every nn.Linear whose attribute name is in `targets`."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r, alpha, dropout))
            n += 1
        else:
            n += inject_lora(child, targets, r, alpha, dropout)
    return n


# --------------------------------------------------------------------------
# Shared interface
# --------------------------------------------------------------------------

class BaseFedModel(nn.Module):
    """Common trainable-state plumbing for the federated loop."""

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_names(self) -> List[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def get_trainable_state(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone().cpu()
                for n, p in self.named_parameters() if p.requires_grad}

    def set_trainable_state(self, state: Dict[str, torch.Tensor]) -> None:
        own = dict(self.named_parameters())
        with torch.no_grad():
            for n, v in state.items():
                if n in own:
                    own[n].copy_(v.to(own[n].device, dtype=own[n].dtype))

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TextCNN(BaseFedModel):
    """Kim (2014). Same tokenisation as the FM, so tokenisation is not a confound."""

    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int = 128,
                 n_filters: int = 128, kernel_sizes=(2, 3, 4), dropout: float = 0.5,
                 pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, ks) for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)
        self.min_len = max(kernel_sizes)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_ids.size(1) < self.min_len:
            pad = self.min_len - input_ids.size(1)
            input_ids = F.pad(input_ids, (0, pad), value=self.embedding.padding_idx)
        x = self.embedding(input_ids).transpose(1, 2)      # B, E, L
        feats = [F.relu(c(x)).max(dim=2).values for c in self.convs]
        return self.fc(self.dropout(torch.cat(feats, dim=1)))


class FMClassifier(BaseFedModel):
    """Frozen pretrained encoder + LoRA adapters + a trainable linear head.

    Pooling is the position-0 token ([CLS] for BERT-family, <s> for RoBERTa-family),
    which is uniform across DistilBERT, BERT and RoBERTa backbones.
    """

    def __init__(self, backbone_key: str, n_classes: int, r: int = 8,
                 alpha: int = 32, dropout: float = 0.1,
                 ffa_lora: bool = False, local_dir: Optional[str] = None):
        super().__init__()
        from transformers import AutoModel

        spec = BACKBONES[backbone_key]
        src = local_dir or spec["hf_id"]
        self.encoder = AutoModel.from_pretrained(src)
        hidden = self.encoder.config.hidden_size

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        n_inject = inject_lora(self.encoder, spec["lora_targets"], r, alpha, dropout)
        if n_inject == 0:
            raise RuntimeError(
                f"No LoRA modules injected for '{backbone_key}'. "
                f"Expected target names {spec['lora_targets']}. "
                f"Inspect module names with: [n for n,_ in model.encoder.named_modules()]"
            )
        self.n_lora_modules = n_inject

        if ffa_lora:  # FFA-LoRA: A frozen at init, only B trained/aggregated
            for m in self.encoder.modules():
                if isinstance(m, LoRALinear):
                    m.lora_A.requires_grad_(False)

        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(self.head_dropout(cls))

    def adapter_deltas(self) -> Dict[str, torch.Tensor]:
        """Effective delta-W per adapted projection, for the mechanism probe."""
        return {n: m.delta_w().detach().cpu()
                for n, m in self.encoder.named_modules() if isinstance(m, LoRALinear)}


def build_model(cfg, vocab_size: int, n_classes: int, local_dir: Optional[str] = None):
    if cfg.model == "textcnn":
        return TextCNN(vocab_size=vocab_size, n_classes=n_classes)
    return FMClassifier(
        backbone_key=cfg.model, n_classes=n_classes,
        r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
        ffa_lora=(cfg.algo == "ffa_lora"), local_dir=local_dir,
    )
