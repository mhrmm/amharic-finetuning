from configure import USE_CUDA
import gc
import json
import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
from lora import apply_lora, merge_lora, LoRALinear
from model import Seq2SeqConfig, Seq2SeqTransformer
import nllb as _nllb


def logger(s, to_stderr=False):
    if to_stderr:
        sys.stderr.write(str(s))
        sys.stderr.write("\n")
        sys.stderr.flush()
    else:
        sys.stdout.write(str(s))
        sys.stdout.write("\n")
        sys.stdout.flush()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def _print_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")


class PrefixTuning(nn.Module):
    """P-Tuning v2 adapted for pure-PyTorch Seq2SeqTransformer and HF (NLLB) models.

    Encoder: learnable prefix embeddings prepended to source embeddings before
    encoding, conditioning all encoder self-attention layers.

    Decoder (cross-attention): learnable prefix vectors concatenated onto the
    encoder memory so every decoder cross-attention layer attends to them.
    This avoids per-layer KV injection and works with any seq2seq base model
    that accepts encoder_outputs as a pre-computed tuple.

    Works with:
    - Our Seq2SeqTransformer (pure PyTorch)
    - HuggingFace NLLB / M2M100 (optional transformers dependency)
    """

    def __init__(self, model, num_virtual_tokens, encoder_hidden_size, prefix_projection):
        super().__init__()
        self.base_model = model
        self.num_virtual_tokens = num_virtual_tokens
        self.prefix_projection = prefix_projection
        self.encoder_hidden_size = encoder_hidden_size

        for param in model.parameters():
            param.requires_grad = False

        d = model.config.d_model

        if prefix_projection:
            self.enc_prefix_emb = nn.Embedding(num_virtual_tokens, encoder_hidden_size)
            self.enc_prefix_mlp = nn.Sequential(
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(encoder_hidden_size, d),
            )
            self.dec_prefix_emb = nn.Embedding(num_virtual_tokens, encoder_hidden_size)
            self.dec_prefix_mlp = nn.Sequential(
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(encoder_hidden_size, d),
            )
        else:
            self.enc_prefix = nn.Parameter(torch.zeros(num_virtual_tokens, d))
            self.dec_prefix = nn.Parameter(torch.zeros(num_virtual_tokens, d))

    # ------------------------------------------------------------------
    # Prefix vector helpers
    # ------------------------------------------------------------------

    def _enc_prefix_vectors(self, batch_size, dtype, device):
        tok = torch.arange(self.num_virtual_tokens, device=device)
        if self.prefix_projection:
            prefix = self.enc_prefix_mlp(self.enc_prefix_emb(tok))
        else:
            prefix = self.enc_prefix
        return prefix.to(dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _dec_prefix_vectors(self, batch_size, dtype, device):
        tok = torch.arange(self.num_virtual_tokens, device=device)
        if self.prefix_projection:
            prefix = self.dec_prefix_mlp(self.dec_prefix_emb(tok))
        else:
            prefix = self.dec_prefix
        return prefix.to(dtype).unsqueeze(0).expand(batch_size, -1, -1)

    # ------------------------------------------------------------------
    # Encoder with prefix prepended
    # ------------------------------------------------------------------

    def _encode_with_prefix(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        encoder = self.base_model.get_encoder()

        token_embeds = encoder.embed_tokens(input_ids)
        prefix_embeds = self._enc_prefix_vectors(batch_size, token_embeds.dtype, input_ids.device)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            batch_size, self.num_virtual_tokens,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        encoder_outputs = encoder(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
        return encoder_outputs, extended_mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def forward(self, input_ids, attention_mask, **kwargs):
        encoder_outputs, extended_mask = self._encode_with_prefix(input_ids, attention_mask)
        batch_size = input_ids.shape[0]
        dtype = encoder_outputs[0].dtype

        dec_prefix = self._dec_prefix_vectors(batch_size, dtype, input_ids.device)
        extended_memory = torch.cat([dec_prefix, encoder_outputs[0]], dim=1)

        dec_prefix_mask = torch.ones(
            batch_size, self.num_virtual_tokens,
            dtype=extended_mask.dtype, device=extended_mask.device,
        )
        extended_cross_mask = torch.cat([dec_prefix_mask, extended_mask], dim=1)

        return self.base_model(
            input_ids=input_ids,
            attention_mask=extended_cross_mask,
            encoder_outputs=(extended_memory,),
            **kwargs,
        )

    def generate(self, input_ids, attention_mask, **kwargs):
        encoder_outputs, extended_mask = self._encode_with_prefix(input_ids, attention_mask)
        batch_size = input_ids.shape[0]
        dtype = encoder_outputs[0].dtype

        dec_prefix = self._dec_prefix_vectors(batch_size, dtype, input_ids.device)
        extended_memory = torch.cat([dec_prefix, encoder_outputs[0]], dim=1)

        dec_prefix_mask = torch.ones(
            batch_size, self.num_virtual_tokens,
            dtype=extended_mask.dtype, device=extended_mask.device,
        )
        extended_cross_mask = torch.cat([dec_prefix_mask, extended_mask], dim=1)

        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=extended_cross_mask,
            encoder_outputs=(extended_memory,),
            **kwargs,
        )

    def save_pretrained(self, save_path):
        """Save prefix parameters.

        Saves only the learnable prefix weights (not the frozen base model).
        Also saves enough information to reconstruct the base model on load.
        """
        os.makedirs(save_path, exist_ok=True)
        prefix_state = {k: v for k, v in self.state_dict().items()
                        if not k.startswith("base_model.")}
        torch.save(prefix_state, os.path.join(save_path, "prefix_tuning_weights.pt"))

        base_cfg = self.base_model.config
        if hasattr(base_cfg, 'save'):
            # Our Seq2SeqTransformer
            base_cfg.save(save_path)
            base_model_format = "seq2seq_transformer"
            base_model_id = ""
        else:
            # HuggingFace model (NLLB etc.)
            base_cfg.save_pretrained(save_path)
            base_model_format = "huggingface"
            base_model_id = getattr(base_cfg, "_name_or_path", "")

        cfg = {
            "type": "prefix_tuning",
            "num_virtual_tokens": self.num_virtual_tokens,
            "encoder_hidden_size": self.encoder_hidden_size,
            "prefix_projection": self.prefix_projection,
            "base_model_format": base_model_format,
            "base_model_name_or_path": base_model_id,
        }
        with open(os.path.join(save_path, "prefix_tuning_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base_model.parameters()) + trainable
        print(
            f"trainable params: {trainable:,} || "
            f"all params: {total:,} || "
            f"trainable%: {100 * trainable / total:.4f}"
        )


# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------

def prepare_model_for_finetuning(ft_params):
    use_hf = _nllb.is_hf_checkpoint(ft_params.base_model)

    if use_hf:
        model = _nllb.load_for_finetuning(ft_params.base_model)
        print(f"loaded NLLB/HF pretrained model from {ft_params.base_model}")
    elif ft_params.should_finetune:
        model = Seq2SeqTransformer.from_pretrained(ft_params.base_model)
        print("loaded pretrained model")
    else:
        config = Seq2SeqConfig.load(ft_params.base_model)
        model = Seq2SeqTransformer.from_config(config)
        print("loaded architecture only")

    if ft_params.freeze_decoder:
        print("--> DECODER FROZEN <--")
        for param in model.get_decoder().parameters():
            param.requires_grad = False
    else:
        print("--> decoder NOT frozen <--")

    if ft_params.freeze_encoder:
        print("--> ENCODER FROZEN <--")
        for param in model.get_encoder().parameters():
            param.requires_grad = False
    else:
        print("--> encoder NOT frozen <--")

    if ft_params.use_lora:
        apply_lora(
            model,
            ft_params.lora_r,
            ft_params.lora_alpha,
            ft_params.lora_dropout,
            ft_params.lora_target_modules,
        )
        if use_hf:
            _nllb.patch_lora_save(model, ft_params.base_model)
        print(f"--> LoRA ENABLED (r={ft_params.lora_r}, alpha={ft_params.lora_alpha}) <--")
        _print_trainable(model)

    if ft_params.use_prefix_tuning:
        model = PrefixTuning(
            model,
            ft_params.prefix_num_virtual_tokens,
            ft_params.prefix_encoder_hidden_size,
            ft_params.prefix_projection,
        )
        print(
            f"--> P-Tuning v2 ENABLED "
            f"(num_virtual_tokens={ft_params.prefix_num_virtual_tokens}, "
            f"projection={ft_params.prefix_projection}) <--"
        )
        model.print_trainable_parameters()

    if USE_CUDA:
        torch.cuda.set_device(0)
        model.cuda()
    return model


def merge_lora_checkpoint(experiment_dir, ft_params):
    """Merge the LoRA adapter into base weights and resave as a clean checkpoint."""
    if (Path(experiment_dir) / "hf_base_model.json").exists():
        # NLLB + LoRA path
        _nllb.merge_lora_and_save(experiment_dir, ft_params)
    else:
        # Our Seq2SeqTransformer + LoRA path
        config = Seq2SeqConfig.load(experiment_dir)
        model = Seq2SeqTransformer.from_config(config)
        apply_lora(
            model,
            ft_params.lora_r,
            ft_params.lora_alpha,
            ft_params.lora_dropout,
            ft_params.lora_target_modules,
        )
        state = torch.load(
            os.path.join(experiment_dir, "model.pt"), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        merge_lora(model)
        model.save_pretrained(experiment_dir)


def load_model_for_inference(model_path, torch_dtype=None):
    """Load any supported checkpoint for inference.

    Supported formats
    -----------------
    prefix_tuning_config.json present → PrefixTuning wrapper (our or HF base)
    model.pt present                  → Seq2SeqTransformer (our pure-PyTorch)
    otherwise                         → HuggingFace model via transformers
    """
    dtype_kwargs = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
    prefix_config_path = Path(model_path) / "prefix_tuning_config.json"

    if prefix_config_path.exists():
        with open(prefix_config_path) as f:
            pt_cfg = json.load(f)

        base_fmt = pt_cfg.get("base_model_format", "seq2seq_transformer")
        if base_fmt == "huggingface":
            base_model = _nllb.load_for_inference(
                pt_cfg["base_model_name_or_path"], torch_dtype=torch_dtype
            )
        else:
            config = Seq2SeqConfig.load(model_path)
            base_model = Seq2SeqTransformer.from_config(config)

        model = PrefixTuning(
            base_model,
            pt_cfg["num_virtual_tokens"],
            pt_cfg["encoder_hidden_size"],
            pt_cfg["prefix_projection"],
        )
        weights = torch.load(
            Path(model_path) / "prefix_tuning_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(weights, strict=False)
        if torch_dtype is not None:
            model = model.to(torch_dtype)
        return model

    if _nllb.is_hf_checkpoint(model_path):
        return _nllb.load_for_inference(model_path, torch_dtype=torch_dtype)

    return Seq2SeqTransformer.from_pretrained(model_path, **dtype_kwargs)
