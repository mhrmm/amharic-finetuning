from configure import USE_CUDA
import gc
import json
import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from transformers import AutoTokenizer


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


def what_nllb_token_is_this(token_id, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    return tokenizer.convert_ids_to_tokens([token_id])[0]


class PrefixTuning(nn.Module):
    """P-Tuning v2 for enc-dec models (e.g. NLLB/M2M100).

    Encoder: learnable prefix embeddings prepended to the source input at the
    embedding layer, conditioning all encoder self-attention layers naturally.

    Decoder self-attention: prefix KV pairs injected at every decoder layer via
    the 4-tuple past_key_values format (prefix_self_k, prefix_self_v, cross_k, cross_v).
    Using 4-tuples (rather than 2-tuples) is critical: M2M100DecoderLayer splits
    past_key_value as [:2] for self-attention and [-2:] for cross-attention.  With
    2-tuples both slices resolve to the same prefix, replacing the cross-attention
    to the encoder output.  The 4-tuple fixes this by placing pre-computed encoder
    KV in positions [-2:], so cross-attention continues to attend to the actual
    source sentence.  Mathematically, pre-computing cross-attention KV outside the
    decoder layer is identical to computing them on-the-fly (the encoder output
    does not change across decoder positions), so gradients flow correctly.

    The extended encoder attention mask (1s for prefix + source positions) is passed
    as attention_mask so the decoder's causal mask and cross-attention mask account
    for the prefix tokens.

    When prefix_projection=True an MLP reparameterises the prefix embeddings,
    matching the conditioning style described in the P-Tuning v2 paper.
    """

    def __init__(self, model, num_virtual_tokens, encoder_hidden_size, prefix_projection):
        super().__init__()
        self.base_model = model
        self.num_virtual_tokens = num_virtual_tokens
        self.prefix_projection = prefix_projection
        self.encoder_hidden_size = encoder_hidden_size

        # Freeze the base model — only prefix params are trainable.
        for param in model.parameters():
            param.requires_grad = False

        cfg = model.config
        d = cfg.d_model
        n_dec = cfg.decoder_layers
        h = cfg.decoder_attention_heads
        self.d_model = d
        self.num_dec_heads = h
        self.head_dim = d // h
        self.num_decoder_layers = n_dec

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
                nn.Linear(encoder_hidden_size, n_dec * 2 * d),
            )
        else:
            self.enc_prefix = nn.Parameter(torch.zeros(num_virtual_tokens, d))
            self.dec_prefix = nn.Parameter(torch.zeros(n_dec, 2, num_virtual_tokens, d))

    # ------------------------------------------------------------------
    # Encoder forward with prefix
    # ------------------------------------------------------------------

    def _encode_with_prefix(self, input_ids, attention_mask):
        """Run encoder with virtual tokens prepended; return (encoder_outputs, extended_mask)."""
        batch_size = input_ids.shape[0]
        encoder = self.base_model.get_encoder()

        # embed_tokens uses M2M100ScaledWordEmbedding which applies the scale internally.
        token_embeds = encoder.embed_tokens(input_ids)
        prefix_embeds = self._enc_prefix_vectors(batch_size, token_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            batch_size, self.num_virtual_tokens,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        encoder_outputs = encoder(inputs_embeds=inputs_embeds, attention_mask=extended_mask)
        return encoder_outputs, extended_mask

    def _enc_prefix_vectors(self, batch_size, dtype):
        """Encoder prefix embeddings: (batch, num_virtual_tokens, d_model)."""
        tok = torch.arange(self.num_virtual_tokens, device=next(self.parameters()).device)
        if self.prefix_projection:
            prefix = self.enc_prefix_mlp(self.enc_prefix_emb(tok))
        else:
            prefix = self.enc_prefix
        return prefix.to(dtype).unsqueeze(0).expand(batch_size, -1, -1)

    # ------------------------------------------------------------------
    # Decoder past_key_values (4-tuples)
    # ------------------------------------------------------------------

    def _build_past_key_values(self, encoder_hidden, batch_size, dtype):
        """Build one 4-tuple per decoder layer:
          (prefix_self_k, prefix_self_v, cross_k, cross_v)

        prefix_self_k/v — learnable prefix for decoder self-attention
        cross_k/v       — pre-computed cross-attention KV from the extended encoder
                          output; equivalent to computing on-the-fly and avoids the
                          2-tuple cross-attention override issue
        """
        n = self.num_decoder_layers
        tok = torch.arange(self.num_virtual_tokens, device=next(self.parameters()).device)

        if self.prefix_projection:
            proj = self.dec_prefix_mlp(self.dec_prefix_emb(tok))   # (n_virt, n*2*d)
            prefix = proj.view(self.num_virtual_tokens, n, 2, self.d_model)
            prefix = prefix.permute(1, 2, 0, 3)                     # (n, 2, n_virt, d)
        else:
            prefix = self.dec_prefix                                  # (n, 2, n_virt, d)

        # (n, 2, n_virt, heads, head_dim) → (n, 2, heads, n_virt, head_dim)
        prefix = prefix.view(n, 2, self.num_virtual_tokens, self.num_dec_heads, self.head_dim)
        prefix = prefix.permute(0, 1, 3, 2, 4)
        # Expand for batch: (n, 2, B, heads, n_virt, head_dim)
        prefix = prefix.unsqueeze(2).expand(-1, -1, batch_size, -1, -1, -1)

        past_key_values = []
        for i, layer in enumerate(self.base_model.get_decoder().layers):
            pk = prefix[i, 0].to(dtype).contiguous()   # (B, heads, n_virt, head_dim)
            pv = prefix[i, 1].to(dtype).contiguous()

            # Pre-compute cross-attention KV using the encoder_attn projection weights.
            # encoder_hidden: (B, num_virt + src_len, d)
            enc_attn = layer.encoder_attn
            cross_k_raw = enc_attn.k_proj(encoder_hidden)  # (B, enc_len, d)
            cross_v_raw = enc_attn.v_proj(encoder_hidden)
            # (B, enc_len, heads, head_dim) → (B, heads, enc_len, head_dim)
            cross_k = (
                cross_k_raw
                .view(batch_size, -1, self.num_dec_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .to(dtype)
            )
            cross_v = (
                cross_v_raw
                .view(batch_size, -1, self.num_dec_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .to(dtype)
            )
            past_key_values.append((pk, pv, cross_k, cross_v))

        return past_key_values

    # ------------------------------------------------------------------
    # Public API expected by finetune.py / validate.py
    # ------------------------------------------------------------------

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def forward(self, input_ids, attention_mask, **kwargs):
        encoder_outputs, extended_mask = self._encode_with_prefix(input_ids, attention_mask)
        batch_size = input_ids.shape[0]
        dtype = encoder_outputs[0].dtype
        past_key_values = self._build_past_key_values(encoder_outputs[0], batch_size, dtype)
        return self.base_model(
            input_ids=input_ids,
            attention_mask=extended_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            **kwargs,
        )

    def generate(self, input_ids, attention_mask, **kwargs):
        encoder_outputs, extended_mask = self._encode_with_prefix(input_ids, attention_mask)
        batch_size = input_ids.shape[0]
        num_beams = kwargs.get("num_beams", 1)
        dtype = encoder_outputs[0].dtype

        # HuggingFace generate expands encoder_outputs tensors by num_beams automatically,
        # but does NOT expand past_key_values (a tuple-of-tuples, not a tensor).
        # Pre-expand past_key_values using the beam-expanded encoder hidden states so
        # dimensions match on the first decode step.
        expanded_enc_hidden = encoder_outputs[0].repeat_interleave(num_beams, dim=0)
        past_key_values = self._build_past_key_values(
            expanded_enc_hidden, batch_size * num_beams, dtype
        )

        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=extended_mask,
            encoder_outputs=encoder_outputs,  # generate will expand this by num_beams
            past_key_values=past_key_values,  # already expanded by num_beams
            **kwargs,
        )

    def save_pretrained(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        prefix_state = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("base_model.")
        }
        torch.save(prefix_state, os.path.join(save_path, "prefix_tuning_weights.pt"))
        cfg = {
            "type": "prefix_tuning",
            "base_model_name_or_path": self.base_model.config._name_or_path,
            "num_virtual_tokens": self.num_virtual_tokens,
            "encoder_hidden_size": self.encoder_hidden_size,
            "prefix_projection": self.prefix_projection,
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


def prepare_model_for_finetuning(ft_params):
    if ft_params.should_finetune:
        model = AutoModelForSeq2SeqLM.from_pretrained(ft_params.base_model)
        print("loaded pretrained model")
    else:
        model_config = AutoConfig.from_pretrained(ft_params.base_model)
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        print("loaded architecture only")
    if hasattr(model.config, "max_length"):  # this should be in a GenerationConfig
        delattr(model.config, "max_length")
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
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=ft_params.lora_r,
            lora_alpha=ft_params.lora_alpha,
            lora_dropout=ft_params.lora_dropout,
            target_modules=ft_params.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        print(f"--> LoRA ENABLED (r={ft_params.lora_r}, alpha={ft_params.lora_alpha}) <--")
        model.print_trainable_parameters()
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
    """Merges a saved LoRA adapter checkpoint into its base model and
    overwrites experiment_dir with the merged full model, so it can be
    loaded downstream with plain AutoModelForSeq2SeqLM.from_pretrained."""
    base_model = AutoModelForSeq2SeqLM.from_pretrained(ft_params.base_model)
    model = PeftModel.from_pretrained(base_model, experiment_dir)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(experiment_dir)


def load_model_for_inference(model_path, torch_dtype=None):
    """Loads a model for inference: plain checkpoint, PEFT LoRA adapter (already
    merged), or custom PrefixTuning checkpoint."""
    dtype_kwargs = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
    prefix_config_path = Path(model_path) / "prefix_tuning_config.json"
    adapter_config_path = Path(model_path) / "adapter_config.json"

    if prefix_config_path.exists():
        with open(prefix_config_path) as f:
            cfg = json.load(f)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg["base_model_name_or_path"], **dtype_kwargs
        )
        model = PrefixTuning(
            base_model,
            cfg["num_virtual_tokens"],
            cfg["encoder_hidden_size"],
            cfg["prefix_projection"],
        )
        weights = torch.load(
            Path(model_path) / "prefix_tuning_weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        # strict=False: weights contains only prefix params; base model params
        # are already loaded from the pretrained checkpoint above.
        model.load_state_dict(weights, strict=False)
        return model

    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            adapter_cfg["base_model_name_or_path"], **dtype_kwargs
        )
        return PeftModel.from_pretrained(base_model, model_path)

    return AutoModelForSeq2SeqLM.from_pretrained(model_path, **dtype_kwargs)
