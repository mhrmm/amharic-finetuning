"""Pure-PyTorch M2M100/NLLB model.

Mirrors the HuggingFace M2M100ForConditionalGeneration architecture so that
pretrained weights can be loaded directly from an HF checkpoint directory
without using transformers at runtime.

Usage
-----
    model = M2M100ForConditionalGeneration.from_pretrained(
        "/path/to/nllb-200-distilled-1.3B"
    )
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class EncoderOutput:
    """Wraps encoder hidden states so that [0] returns last_hidden_state."""
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

    def __getitem__(self, idx):
        if idx == 0:
            return self.last_hidden_state
        raise IndexError(idx)


class Seq2SeqOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class M2M100Config:
    def __init__(
        self,
        vocab_size: int = 256206,
        d_model: int = 1024,
        encoder_layers: int = 24,
        decoder_layers: int = 24,
        encoder_attention_heads: int = 16,
        decoder_attention_heads: int = 16,
        encoder_ffn_dim: int = 8192,
        decoder_ffn_dim: int = 8192,
        max_position_embeddings: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int = 1,
        eos_token_id: int = 2,
        decoder_start_token_id: int = 2,
        name_or_path: str = "",
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.name_or_path = name_or_path
        self._name_or_path = name_or_path
        self.model_type = "m2m_100"

    @classmethod
    def from_hf_dir(cls, path: str) -> "M2M100Config":
        with open(Path(path) / "config.json") as f:
            c = json.load(f)
        return cls(
            vocab_size=c["vocab_size"],
            d_model=c["d_model"],
            encoder_layers=c["encoder_layers"],
            decoder_layers=c["decoder_layers"],
            encoder_attention_heads=c["encoder_attention_heads"],
            decoder_attention_heads=c["decoder_attention_heads"],
            encoder_ffn_dim=c["encoder_ffn_dim"],
            decoder_ffn_dim=c["decoder_ffn_dim"],
            max_position_embeddings=c.get("max_position_embeddings", 1024),
            dropout=c.get("dropout", 0.1),
            pad_token_id=c.get("pad_token_id", 1),
            eos_token_id=c.get("eos_token_id", 2),
            decoder_start_token_id=c.get("decoder_start_token_id", 2),
            name_or_path=str(path),
        )

    def save(self, path: str):
        d = {
            "model_type": "m2m_100",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "encoder_attention_heads": self.encoder_attention_heads,
            "decoder_attention_heads": self.decoder_attention_heads,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "decoder_ffn_dim": self.decoder_ffn_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "dropout": self.dropout,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
        }
        with open(Path(path) / "config.json", "w") as f:
            json.dump(d, f, indent=2)


# ---------------------------------------------------------------------------
# Sinusoidal positional embeddings
# Matches HF M2M100SinusoidalPositionalEmbedding exactly so the buffer
# loaded from an HF checkpoint is identical to what we compute here.
# ---------------------------------------------------------------------------

def _sinusoidal_weights(num_rows: int, d_model: int, padding_idx: int) -> torch.Tensor:
    half = d_model // 2
    freq = math.log(10000) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32) * -freq)
    pos = torch.arange(num_rows, dtype=torch.float32).unsqueeze(1) * freq.unsqueeze(0)
    w = torch.cat([torch.sin(pos), torch.cos(pos)], dim=1)
    if d_model % 2 == 1:
        w = torch.cat([w, torch.zeros(num_rows, 1)], dim=1)
    w[padding_idx] = 0.0
    return w


class SinusoidalPositionalEmbedding(nn.Module):
    _OFFSET = 2  # HF M2M100 positions start from padding_idx + 1 = 2

    def __init__(self, max_positions: int, d_model: int, padding_idx: int = 1):
        super().__init__()
        self.padding_idx = padding_idx
        w = _sinusoidal_weights(max_positions + self._OFFSET, d_model, padding_idx)
        self.register_buffer("weights", w, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_len: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            mask = input_ids.ne(self.padding_idx).int()
            pos = (torch.cumsum(mask, dim=1) * mask + past_len * mask).long() + self.padding_idx
        else:
            bsz, seq_len = inputs_embeds.shape[:2]
            pos = (
                torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
                + self._OFFSET
            ).unsqueeze(0).expand(bsz, -1)
        return self.weights.index_select(0, pos.reshape(-1)).view(*pos.shape, -1).detach()


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _additive_mask(mask_2d: torch.Tensor, tgt_len: int, dtype: torch.dtype) -> torch.Tensor:
    """(B, src_len) binary keep-mask → (B, 1, tgt_len, src_len) additive mask."""
    inv = (1.0 - mask_2d.to(dtype))[:, None, None, :].expand(-1, 1, tgt_len, -1)
    return inv.masked_fill(inv.bool(), torch.finfo(dtype).min)


def _causal_mask(past_len: int, new_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """(1, 1, new_len, past_len+new_len) causal additive mask."""
    total = past_len + new_len
    m = torch.full((new_len, total), torch.finfo(dtype).min, dtype=dtype, device=device)
    for i in range(new_len):
        m[i, : past_len + i + 1] = 0.0
    return m.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Attention  (separate q/k/v projections, matching HF M2M100Attention)
# ---------------------------------------------------------------------------

# Per-layer KV cache: (self_k, self_v, cross_k, cross_v)
LayerCache = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class M2M100Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, tgt, _ = x.shape
        is_cross = kv is not None

        q = self._heads(self.q_proj(x)) * self.scale

        if is_cross and past is not None:
            k, v = past
        elif is_cross:
            k = self._heads(self.k_proj(kv))
            v = self._heads(self.v_proj(kv))
        elif past is not None:
            k = torch.cat([past[0], self._heads(self.k_proj(x))], dim=2)
            v = torch.cat([past[1], self._heads(self.v_proj(x))], dim=2)
        else:
            k = self._heads(self.k_proj(x))
            v = self._heads(self.v_proj(x))

        w = torch.matmul(q, k.transpose(-1, -2))
        if mask is not None:
            w = w + mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        w = self.drop(w)

        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(b, tgt, -1)
        return self.out_proj(out), (k, v)


# ---------------------------------------------------------------------------
# Encoder layer  (pre-norm)
# ---------------------------------------------------------------------------

class M2M100EncoderLayer(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()
        d = config.d_model
        self.self_attn = M2M100Attention(d, config.encoder_attention_heads, config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r = x
        x, _ = self.self_attn(self.self_attn_layer_norm(x), mask=mask)
        x = r + self.drop(x)

        r = x
        x = self.final_layer_norm(x)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return r + x


# ---------------------------------------------------------------------------
# Decoder layer  (pre-norm)
# ---------------------------------------------------------------------------

class M2M100DecoderLayer(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()
        d = config.d_model
        self.self_attn = M2M100Attention(d, config.decoder_attention_heads, config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.encoder_attn = M2M100Attention(d, config.decoder_attention_heads, config.dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        past: Optional[LayerCache] = None,
    ) -> Tuple[torch.Tensor, LayerCache]:
        r = x
        x, self_kv = self.self_attn(self.self_attn_layer_norm(x), mask=self_mask,
                                     past=past[:2] if past is not None else None)
        x = r + self.drop(x)

        r = x
        x, cross_kv = self.encoder_attn(self.encoder_attn_layer_norm(x), kv=enc, mask=cross_mask,
                                         past=past[2:] if past is not None else None)
        x = r + self.drop(x)

        r = x
        x = self.final_layer_norm(x)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return r + x, (*self_kv, *cross_kv)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class M2M100Encoder(nn.Module):
    def __init__(self, config: M2M100Config, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_scale = math.sqrt(config.d_model)
        self.embed_tokens = embed_tokens
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id
        )
        self.layers = nn.ModuleList([M2M100EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        pos = self.embed_positions(input_ids=input_ids, inputs_embeds=inputs_embeds)
        x = self.drop(inputs_embeds + pos)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = _additive_mask(attention_mask, x.size(1), x.dtype)

        for layer in self.layers:
            x = layer(x, mask=attn_mask)

        return EncoderOutput(self.layer_norm(x))


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class M2M100Decoder(nn.Module):
    def __init__(self, config: M2M100Config, embed_tokens: nn.Embedding):
        super().__init__()
        self.embed_scale = math.sqrt(config.d_model)
        self.embed_tokens = embed_tokens
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.d_model, config.pad_token_id
        )
        self.layers = nn.ModuleList([M2M100DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        enc_hidden: torch.Tensor,
        enc_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[LayerCache]] = None,
    ) -> Tuple[torch.Tensor, List[LayerCache]]:
        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0
        new_len = input_ids.size(1)

        embeds = self.embed_tokens(input_ids) * self.embed_scale
        pos = self.embed_positions(
            input_ids=input_ids if past_len == 0 else None,
            inputs_embeds=embeds if past_len > 0 else None,
            past_len=past_len,
        )
        x = self.drop(embeds + pos)

        self_mask = _causal_mask(past_len, new_len, x.dtype, x.device)
        cross_mask = _additive_mask(enc_mask, new_len, x.dtype) if enc_mask is not None else None

        new_kv: List[LayerCache] = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, enc_hidden, self_mask=self_mask, cross_mask=cross_mask,
                          past=past_key_values[i] if past_key_values is not None else None)
            new_kv.append(kv)

        return self.layer_norm(x), new_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class M2M100ForConditionalGeneration(nn.Module):
    def __init__(self, config: M2M100Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.encoder = M2M100Encoder(config, self.shared)
        self.decoder = M2M100Decoder(config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight  # tie

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_encoder(self) -> M2M100Encoder:
        return self.encoder

    def get_decoder(self) -> M2M100Decoder:
        return self.decoder

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = self.config.decoder_start_token_id
        shifted[shifted == -100] = self.config.pad_token_id
        return shifted

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_outputs=None,
        **kwargs,
    ) -> Seq2SeqOutput:
        enc = encoder_outputs[0] if encoder_outputs is not None else self.encoder(input_ids, attention_mask)[0]

        if decoder_input_ids is None:
            if labels is not None:
                decoder_input_ids = self._shift_right(labels)
            else:
                raise ValueError("Provide decoder_input_ids or labels.")

        dec, _ = self.decoder(decoder_input_ids, enc, enc_mask=attention_mask)
        logits = self.lm_head(dec)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)

        return Seq2SeqOutput(loss=loss, logits=logits)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        forced_bos_token_id: Optional[int] = None,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        encoder_outputs=None,
        **kwargs,
    ) -> torch.Tensor:
        enc = encoder_outputs[0] if encoder_outputs is not None else self.encoder(input_ids, attention_mask)[0]
        batch = enc.size(0)
        device = enc.device
        eos = self.config.eos_token_id

        if num_beams <= 1:
            return self._greedy(enc, attention_mask, forced_bos_token_id, max_new_tokens, batch, device, eos)
        return self._beam(enc, attention_mask, forced_bos_token_id, max_new_tokens, num_beams, batch, device, eos)

    def _step(self, tok, enc, enc_mask, past_kv):
        dec, new_kv = self.decoder(tok, enc, enc_mask=enc_mask, past_key_values=past_kv)
        return self.lm_head(dec[:, -1, :]), new_kv

    def _greedy(self, enc, enc_mask, forced_bos, max_new_tokens, batch, device, eos):
        seqs = torch.full((batch, 1), self.config.decoder_start_token_id, dtype=torch.long, device=device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        past_kv = None

        for step in range(max_new_tokens):
            inp = seqs if past_kv is None else seqs[:, -1:]
            logits, past_kv = self._step(inp, enc, enc_mask, past_kv)

            if step == 0 and forced_bos is not None:
                tok = torch.full((batch,), forced_bos, dtype=torch.long, device=device)
            else:
                tok = logits.argmax(dim=-1)

            tok = tok.masked_fill(done, eos)
            seqs = torch.cat([seqs, tok.unsqueeze(1)], dim=1)
            done = done | (tok == eos)
            if done.all():
                break

        return seqs

    def _beam(self, enc, enc_mask, forced_bos, max_new_tokens, K, batch, device, eos):
        vocab = self.config.vocab_size

        enc = enc.repeat_interleave(K, dim=0)
        if enc_mask is not None:
            enc_mask = enc_mask.repeat_interleave(K, dim=0)

        BK = batch * K
        seqs = torch.full((BK, 1), self.config.decoder_start_token_id, dtype=torch.long, device=device)
        scores = torch.full((BK,), float("-inf"), device=device)
        scores[torch.arange(batch, device=device) * K] = 0.0
        done = torch.zeros(BK, dtype=torch.bool, device=device)
        past_kv = None

        for step in range(max_new_tokens):
            inp = seqs if past_kv is None else seqs[:, -1:]
            logits, past_kv = self._step(inp, enc, enc_mask, past_kv)

            if step == 0 and forced_bos is not None:
                lp = torch.full((BK, vocab), float("-inf"), device=device, dtype=logits.dtype)
                lp[:, forced_bos] = 0.0
            else:
                lp = F.log_softmax(logits, dim=-1)
                lp[done] = float("-inf")
                lp[done, eos] = 0.0

            cand = (scores.unsqueeze(1) + lp).view(batch, K * vocab)
            top_scores, top_idx = cand.topk(K, dim=1)

            src = top_idx // vocab
            tok = top_idx % vocab
            off = torch.arange(batch, device=device).unsqueeze(1) * K
            g = (off + src).view(-1)

            seqs = torch.cat([seqs[g], tok.view(-1, 1)], dim=1)
            scores = top_scores.view(-1)
            done = done[g] | (tok.view(-1) == eos)
            enc = enc[g]
            if enc_mask is not None:
                enc_mask = enc_mask[g]
            past_kv = [tuple(t[g] for t in lc) for lc in past_kv]

            if done.all():
                break

        best = scores.view(batch, K).argmax(dim=1)
        return seqs[torch.arange(batch, device=device) * K + best]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.config.save(path)
        torch.save(self.state_dict(), Path(path) / "model.pt")

    @classmethod
    def from_pretrained(cls, path: str, torch_dtype=None) -> "M2M100ForConditionalGeneration":
        local_path = _resolve_path(path)
        config = M2M100Config.from_hf_dir(local_path)
        model = cls(config)

        our_pt = Path(local_path) / "model.pt"
        if our_pt.exists():
            sd = torch.load(our_pt, map_location="cpu", weights_only=True)
            model.load_state_dict(sd)
        else:
            _load_hf_weights(model, local_path)

        if torch_dtype is not None:
            model = model.to(torch_dtype)
        return model

    def print_trainable_parameters(self):
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n = sum(p.numel() for p in self.parameters())
        print(f"trainable: {t:,} / {n:,} ({100*t/n:.2f}%)")


# ---------------------------------------------------------------------------
# HF weight loading  (no transformers required)
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """Return a local directory path, downloading from the Hub if needed."""
    if Path(path).is_dir():
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path)


def _read_hf_state_dict(path: str) -> dict:
    p = Path(path)

    def _try_safetensors(fpath):
        from safetensors.torch import load_file
        return load_file(fpath, device="cpu")

    # single safetensors
    if (p / "model.safetensors").exists():
        try:
            return _try_safetensors(p / "model.safetensors")
        except ImportError:
            pass

    # sharded safetensors
    if (p / "model.safetensors.index.json").exists():
        try:
            with open(p / "model.safetensors.index.json") as f:
                idx = json.load(f)
            sd = {}
            for fn in set(idx["weight_map"].values()):
                sd.update(_try_safetensors(p / fn))
            return sd
        except ImportError:
            pass

    # single pytorch_model.bin
    if (p / "pytorch_model.bin").exists():
        return torch.load(p / "pytorch_model.bin", map_location="cpu", weights_only=True)

    # sharded pytorch_model.bin
    if (p / "pytorch_model.bin.index.json").exists():
        with open(p / "pytorch_model.bin.index.json") as f:
            idx = json.load(f)
        sd = {}
        for fn in set(idx["weight_map"].values()):
            sd.update(torch.load(p / fn, map_location="cpu", weights_only=True))
        return sd

    raise FileNotFoundError(f"No weight file found in {path}")


def _load_hf_weights(model: M2M100ForConditionalGeneration, path: str):
    hf_sd = _read_hf_state_dict(path)

    # HF keys are prefixed with "model." except lm_head
    our_sd = {}
    for k, v in hf_sd.items():
        our_sd[k[len("model."):] if k.startswith("model.") else k] = v

    missing, unexpected = model.load_state_dict(our_sd, strict=False)
    # Tied weights appear in our state_dict under multiple names but are a single
    # parameter, so "missing" from the loader's perspective is harmless for them.
    _tied = {"encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"}
    real_missing = [k for k in missing if k not in _tied]
    if real_missing:
        raise RuntimeError(f"Missing weights after loading HF checkpoint: {real_missing}")
    if unexpected:
        print(f"[nllb_pt] unexpected HF keys (ignored): {unexpected}")
