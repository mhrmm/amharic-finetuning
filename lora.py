import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps nn.Linear with a low-rank LoRA adapter (Hu et al. 2021).

    The frozen base weight W0 is augmented with a learnable delta B@A scaled
    by alpha/r.  Only A and B are updated during training.
    """

    def __init__(self, linear: nn.Linear, r: int, lora_alpha: float, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = lora_alpha / r

        for p in self.linear.parameters():
            p.requires_grad = False

        in_f = linear.in_features
        out_f = linear.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = F.linear(self.dropout(x), self.lora_B @ self.lora_A) * self.scaling
        return base + lora

    def merge(self) -> nn.Linear:
        """Merge LoRA delta into W0 and return a plain nn.Linear."""
        with torch.no_grad():
            self.linear.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        for p in self.linear.parameters():
            p.requires_grad = True
        return self.linear


def apply_lora(
    model: nn.Module,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    target_modules: List[str],
) -> nn.Module:
    """Replace every nn.Linear whose attribute name is in target_modules with a LoRALinear."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.split(".")[-1]
        if leaf_name not in target_modules:
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, r, lora_alpha, lora_dropout))
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRALinear adapters into their base weights in-place."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.merge())
    return model
