"""NLLB / HuggingFace model integration via pure-PyTorch nllb_pt module.

Loads weights directly from an HF checkpoint directory without using
the transformers library at runtime.

Expected on-disk layouts after training
----------------------------------------
Plain finetune (no adapters):
    <experiment_dir>/
        config.json
        model.pt               ← our format (saved by M2M100ForConditionalGeneration)

LoRA finetune (adapters NOT yet merged):
    <experiment_dir>/
        hf_base_model.json     ← {"base_model_name_or_path": "..."}
        lora_weights.pt        ← only the LoRA A/B matrices

After merge_lora_and_save():
    <experiment_dir>/
        config.json
        model.pt               ← merged weights in our format
"""

import json
import os
from pathlib import Path

import torch

from nllb_pt import M2M100ForConditionalGeneration


def is_hf_checkpoint(path: str) -> bool:
    """True if path should be loaded as an M2M100/NLLB model.

    Distinguishes our two local formats:
      - M2M100: config.json has model_type == "m2m_100"
      - Seq2SeqTransformer: config.json has no model_type, model.pt present
    Hub IDs (non-directory paths) are always treated as M2M100/HF.
    """
    p = Path(path)
    if not p.is_dir():
        return True
    cfg = p / "config.json"
    if cfg.exists():
        with open(cfg) as f:
            model_type = json.load(f).get("model_type", "")
        if model_type == "m2m_100":
            return True
    return not (p / "model.pt").exists()


def load_for_finetuning(model_name_or_path: str, torch_dtype=None):
    """Load an NLLB (or any M2M100) model ready for finetuning."""
    return M2M100ForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)


def load_for_inference(model_path: str, torch_dtype=None):
    """Load an M2M100/NLLB model for inference."""
    return M2M100ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch_dtype)


def patch_lora_save(model, base_model_name_or_path: str):
    """Replace model.save_pretrained so it saves only the LoRA delta.

    The patched save writes:
        hf_base_model.json   ← records the base model identifier
        lora_weights.pt      ← only A/B matrices (small)
    """
    def _lora_save(save_path, **_kwargs):
        os.makedirs(save_path, exist_ok=True)
        lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        torch.save(lora_state, os.path.join(save_path, "lora_weights.pt"))
        with open(os.path.join(save_path, "hf_base_model.json"), "w") as f:
            json.dump({"base_model_name_or_path": str(base_model_name_or_path)}, f)

    model.save_pretrained = _lora_save


def merge_lora_and_save(experiment_dir: str, ft_params):
    """Merge LoRA delta into the base model and save in our format."""
    from lora import apply_lora, merge_lora

    info_path = Path(experiment_dir) / "hf_base_model.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"hf_base_model.json not found in {experiment_dir}. "
            "Was the model saved with patch_lora_save()?"
        )
    with open(info_path) as f:
        base_name = json.load(f)["base_model_name_or_path"]

    model = M2M100ForConditionalGeneration.from_pretrained(base_name)
    apply_lora(
        model,
        ft_params.lora_r,
        ft_params.lora_alpha,
        ft_params.lora_dropout,
        ft_params.lora_target_modules,
    )
    lora_state = torch.load(
        Path(experiment_dir) / "lora_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(lora_state, strict=False)
    merge_lora(model)
    info_path.unlink(missing_ok=True)
    (Path(experiment_dir) / "lora_weights.pt").unlink(missing_ok=True)
    model.save_pretrained(experiment_dir)
