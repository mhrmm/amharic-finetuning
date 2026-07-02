import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bool_to_additive(bool_mask: torch.Tensor) -> torch.Tensor:
    """Convert a boolean padding mask (True = ignore) to an additive float mask
    compatible with PyTorch's attn_mask convention (-inf = ignore)."""
    return bool_mask.float().masked_fill(bool_mask, float("-inf"))


class Seq2SeqConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        bos_token_id: int = 1,
        name_or_path: str = "",
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.name_or_path = name_or_path

    # Aliases expected by PrefixTuning and other code
    @property
    def decoder_layers(self):
        return self.num_decoder_layers

    @property
    def decoder_attention_heads(self):
        return self.nhead

    # HuggingFace-style compat alias (read-only via property)
    @property
    def _name_or_path(self):
        return self.name_or_path

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "name_or_path": self.name_or_path,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "config.json")) as f:
            return cls.from_dict(json.load(f))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class EncoderOutput:
    """Thin container so encoder_outputs[0] returns last_hidden_state."""

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

    def __getitem__(self, idx):
        if idx == 0:
            return self.last_hidden_state
        raise IndexError(idx)


class ModelOutput:
    def __init__(self, loss=None, logits=None, encoder_last_hidden_state=None):
        self.loss = loss
        self.logits = logits
        self.encoder_last_hidden_state = encoder_last_hidden_state


class EncoderModule(nn.Module):
    """Encoder with exposed embed_tokens for PrefixTuning."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.d_model = config.d_model
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self._register_pos_encoding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

    def _register_pos_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * math.sqrt(self.d_model)
        seq_len = inputs_embeds.size(1)
        x = self.drop(inputs_embeds + self.pe[:, :seq_len, :])
        pad_mask = _bool_to_additive(attention_mask == 0) if attention_mask is not None else None
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        return EncoderOutput(out)


class DecoderModule(nn.Module):
    """Decoder wrapper exposing .layers for PrefixTuning."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.d_model = config.d_model
        self._register_pos_encoding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )

    def _register_pos_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    @property
    def layers(self):
        return self.decoder.layers

    def forward(self, input_embeds, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        tgt_emb = input_embeds  # pre-scaled embeddings from caller
        seq_len = tgt_emb.size(1)
        tgt_emb = self.drop(tgt_emb + self.pe[:, :seq_len, :])
        causal = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt_emb.device)
        return self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


class Seq2SeqTransformer(nn.Module):
    """Pure-PyTorch encoder-decoder transformer for seq2seq tasks."""

    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.config = config
        self.encoder_module = EncoderModule(config)
        self.decoder_module = DecoderModule(config)
        self.tgt_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_encoder(self) -> EncoderModule:
        return self.encoder_module

    def get_decoder(self) -> DecoderModule:
        return self.decoder_module

    def _encode(self, input_ids, attention_mask):
        return self.encoder_module(input_ids=input_ids, attention_mask=attention_mask)

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = self.config.bos_token_id
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
    ) -> ModelOutput:
        if encoder_outputs is not None:
            memory = encoder_outputs[0]
            mem_pad_mask = _bool_to_additive(attention_mask == 0) if attention_mask is not None else None
        else:
            enc_out = self._encode(input_ids, attention_mask)
            memory = enc_out.last_hidden_state
            mem_pad_mask = _bool_to_additive(attention_mask == 0) if attention_mask is not None else None

        if decoder_input_ids is None:
            if labels is not None:
                decoder_input_ids = self._shift_right(labels)
            else:
                raise ValueError("Provide decoder_input_ids or labels.")

        tgt_emb = self.tgt_embedding(decoder_input_ids) * math.sqrt(self.config.d_model)
        tgt_pad_mask = _bool_to_additive(decoder_input_ids == self.config.pad_token_id)

        dec_out = self.decoder_module.forward(
            tgt_emb, memory,
            memory_key_padding_mask=mem_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        logits = self.output_proj(dec_out)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return ModelOutput(loss=loss, logits=logits, encoder_last_hidden_state=memory)

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
        bos_id = forced_bos_token_id if forced_bos_token_id is not None else self.config.bos_token_id
        batch_size = input_ids.size(0)

        if encoder_outputs is not None:
            memory = encoder_outputs[0]
            mem_pad_mask = _bool_to_additive(attention_mask == 0) if attention_mask is not None else None
        else:
            enc_out = self._encode(input_ids, attention_mask)
            memory = enc_out.last_hidden_state
            mem_pad_mask = _bool_to_additive(attention_mask == 0) if attention_mask is not None else None

        if num_beams <= 1:
            return self._greedy_decode(memory, mem_pad_mask, bos_id, max_new_tokens)
        return self._beam_search(memory, mem_pad_mask, bos_id, max_new_tokens, num_beams, batch_size)

    def _greedy_decode(self, memory, mem_pad_mask, bos_id, max_new_tokens):
        device = memory.device
        B = memory.size(0)
        eos = self.config.eos_token_id

        seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if done.all():
                break
            tgt_emb = self.tgt_embedding(seqs) * math.sqrt(self.config.d_model)
            dec_out = self.decoder_module.forward(tgt_emb, memory, memory_key_padding_mask=mem_pad_mask)
            next_tok = self.output_proj(dec_out[:, -1, :]).argmax(dim=-1)  # (B,)
            next_tok = next_tok.masked_fill(done, eos)
            seqs = torch.cat([seqs, next_tok.unsqueeze(1)], dim=1)
            done = done | (next_tok == eos)

        return seqs

    def _beam_search(self, memory, mem_pad_mask, bos_id, max_new_tokens, num_beams, batch_size):
        device = memory.device
        vocab = self.config.vocab_size
        eos = self.config.eos_token_id

        # Expand memory: (B, src_len, d) → (B*K, src_len, d)
        memory = memory.repeat_interleave(num_beams, dim=0)
        if mem_pad_mask is not None:
            mem_pad_mask = mem_pad_mask.repeat_interleave(num_beams, dim=0)

        BK = batch_size * num_beams
        seqs = torch.full((BK, 1), bos_id, dtype=torch.long, device=device)

        # Only the first beam per sample starts active; others are -inf
        scores = torch.full((BK,), float("-inf"), device=device)
        scores[torch.arange(batch_size, device=device) * num_beams] = 0.0

        done = torch.zeros(BK, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if done.all():
                break

            tgt_emb = self.tgt_embedding(seqs) * math.sqrt(self.config.d_model)
            dec_out = self.decoder_module.forward(tgt_emb, memory, memory_key_padding_mask=mem_pad_mask)
            logits = self.output_proj(dec_out[:, -1, :])  # (BK, vocab)
            lp = F.log_softmax(logits, dim=-1)

            # Finished beams: force EOS
            lp[done] = float("-inf")
            lp[done, eos] = 0.0

            cand = (scores.unsqueeze(1) + lp).view(batch_size, num_beams * vocab)  # (B, K*V)
            top_scores, top_idx = cand.topk(num_beams, dim=1)  # (B, K)

            src_beam = top_idx // vocab  # (B, K) index within [0, num_beams)
            tok = top_idx % vocab  # (B, K)

            batch_off = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            g_src = (batch_off + src_beam).view(-1)  # (BK,)

            seqs = seqs[g_src]
            seqs = torch.cat([seqs, tok.view(-1, 1)], dim=1)
            scores = top_scores.view(-1)
            done = done[g_src] | (tok.view(-1) == eos)
            memory = memory[g_src]
            if mem_pad_mask is not None:
                mem_pad_mask = mem_pad_mask[g_src]

        best = scores.view(batch_size, num_beams).argmax(dim=1)
        g_best = torch.arange(batch_size, device=device) * num_beams + best
        return seqs[g_best]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100.0 * trainable / total if total else 0.0
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        self.config.name_or_path = str(path)
        self.config.save(path)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path, torch_dtype=None):
        config = Seq2SeqConfig.load(path)
        model = cls(config)
        state = torch.load(
            os.path.join(path, "model.pt"), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        if torch_dtype is not None:
            model = model.to(torch_dtype)
        return model

    @classmethod
    def from_config(cls, config: Seq2SeqConfig):
        return cls(config)
