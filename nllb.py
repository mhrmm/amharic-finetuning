"""Optional NLLB / HuggingFace model integration.

This module lazy-imports `transformers` so the rest of the codebase stays
pure PyTorch.  Call any public function here only when you actually need to
work with an NLLB (or other HuggingFace seq2seq) checkpoint.

Expected on-disk layouts after training
----------------------------------------
Plain finetune (no adapters):
    <experiment_dir>/           ← standard HuggingFace save_pretrained output
        config.json
        pytorch_model.bin  (or model.safetensors shards)

LoRA finetune (adapters NOT yet merged):
    <experiment_dir>/
        hf_base_model.json      ← {"base_model_name_or_path": "..."}
        lora_weights.pt         ← only the LoRA A/B matrices

After merge_lora_and_save():
    <experiment_dir>/           ← standard HuggingFace format, LoRA baked in
        config.json
        pytorch_model.bin
"""

import json
import os
from pathlib import Path

import torch


def _require_transformers():
    try:
        from transformers import AutoModelForSeq2SeqLM
        return AutoModelForSeq2SeqLM
    except ImportError:
        raise ImportError(
            "transformers is required to load NLLB / HuggingFace models.\n"
            "Install with: pip install transformers sentencepiece"
        )


def is_hf_checkpoint(path: str) -> bool:
    """True if path is a HuggingFace model hub ID or a local HF directory.

    Our format is identified by the presence of model.pt.  Anything else
    (including bare hub IDs like "facebook/nllb-200-distilled-600M") is
    treated as HuggingFace.
    """
    p = Path(path)
    if not p.is_dir():
        return True  # hub ID or non-existent path → assume HF
    return not (p / "model.pt").exists()


def load_for_finetuning(model_name_or_path: str, torch_dtype=None):
    """Load an NLLB (or any HF seq2seq) model ready for finetuning."""
    AutoModelForSeq2SeqLM = _require_transformers()
    kwargs = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
    return AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)


def load_for_inference(model_path: str, torch_dtype=None):
    """Load an HF seq2seq model for inference.

    Handles three cases:
      1. Plain checkpoint (HF format with config.json + pytorch_model.bin)
      2. LoRA checkpoint saved with patch_lora_save — NOT handled here
         (LoRA should have been merged before inference; see merge_lora_and_save)
      3. PrefixTuning checkpoint — NOT handled here; myutil handles it.
    """
    AutoModelForSeq2SeqLM = _require_transformers()
    kwargs = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
    return AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)


def patch_lora_save(model, base_model_name_or_path: str):
    """Replace model.save_pretrained so it saves only the LoRA delta.

    Called after apply_lora() on an HF model.  Without this, calling
    model.save_pretrained() would bake the LoRA wrapper module names into
    the saved state dict (e.g. 'out_proj.linear.weight' instead of
    'out_proj.weight'), making the checkpoint unloadable by HuggingFace.

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
    """Merge LoRA delta into an HF base model and save in standard HF format.

    Reads the hf_base_model.json written by patch_lora_save to find the
    original model name, then re-applies LoRA wrappers, loads the saved
    delta, merges, and writes a clean HF checkpoint back to experiment_dir.
    """
    from lora import apply_lora, merge_lora

    info_path = Path(experiment_dir) / "hf_base_model.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"hf_base_model.json not found in {experiment_dir}. "
            "Was the model saved with patch_lora_save()?"
        )
    with open(info_path) as f:
        info = json.load(f)
    base_name = info["base_model_name_or_path"]

    AutoModelForSeq2SeqLM = _require_transformers()
    model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
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
    # Remove the LoRA-format marker before saving in clean HF format.
    info_path.unlink(missing_ok=True)
    (Path(experiment_dir) / "lora_weights.pt").unlink(missing_ok=True)
    model.save_pretrained(experiment_dir)
