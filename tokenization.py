import sys
from typing import Dict, List, Optional
import warnings
from abc import ABC, abstractmethod

import torch


class Tokenizer(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __call__(self, sents):
        pass

    @abstractmethod
    def get_special_tokens(self):
        pass

    @abstractmethod
    def batch_decode(self, token_ids):
        pass


class SentencePieceTokenizer(Tokenizer):
    """Wraps a trained SentencePiece model.

    Config format (in experiment JSON):
        {
            "type": "sentencepiece",
            "model": "path/to/spm.model",
            "max_length": 192,
            "special_tokens": {
                "<pad>":   0,
                "</s>":    1,
                "eng_Latn": 2,
                "amh_Ethi": 3
            }
        }

    The special_tokens dict maps token names to IDs that are reserved
    OUTSIDE the SentencePiece vocabulary (prepended/appended manually).
    All IDs listed here are skipped during batch_decode.

    The SentencePiece model must be trained so that its IDs do NOT overlap
    with the special_token IDs (e.g. offset the SP vocab by
    len(special_tokens), or train with --pad_id / --eos_id / --bos_id
    set to the reserved values).
    """

    def __init__(
        self,
        model_path: str,
        max_length: Optional[int] = None,
        special_tokens: Optional[Dict[str, int]] = None,
        vocab_offset: int = 0,
    ):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "sentencepiece is required for SentencePieceTokenizer. "
                "Install with: pip install sentencepiece"
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.max_length = max_length
        self.vocab_offset = vocab_offset
        self._special_tokens: Dict[str, int] = special_tokens or {}
        self._skip_ids = set(self._special_tokens.values())
        self._eos_id = self._special_tokens.get("</s>", self.sp.eos_id())

    def __len__(self):
        return self.sp.GetPieceSize()

    def __call__(self, sent: str, lang_code: Optional[str] = None) -> List[int]:
        ids = [i + self.vocab_offset for i in self.sp.EncodeAsIds(sent)]
        if lang_code is not None and lang_code in self._special_tokens:
            ids = [self._special_tokens[lang_code]] + ids
        ids = ids + [self._eos_id]
        if self.max_length is not None and len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [self._eos_id]
        return ids

    def get_special_tokens(self) -> Dict[str, int]:
        return self._special_tokens

    def batch_decode(self, token_ids) -> List[str]:
        results = []
        for seq in token_ids:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            filtered = [
                t - self.vocab_offset
                for t in seq
                if t not in self._skip_ids and t >= 0 and t - self.vocab_offset > 0
            ]
            results.append(self.sp.DecodeIds(filtered))
        return results

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.sp.IdToPiece(i - self.vocab_offset) for i in ids]


class ByteTokenizer(Tokenizer):
    def __init__(self, max_length=None):
        self.special_tokens = {"<pad>": 0, "</s>": 2, "lang_Lang": 1}
        self.max_length = max_length

    def __call__(self, sent: str, lang_code="lang_Lang"):
        tokens = [byte + len(self.special_tokens) for byte in sent.encode()]
        if self.max_length is not None and len(tokens) > self.max_length - 2:
            tokens = tokens[: self.max_length - 2]
        return [self.special_tokens[lang_code]] + tokens + [self.special_tokens["</s>"]]

    def __len__(self):
        return 256 + len(self.special_tokens)

    def get_special_tokens(self):
        return self.special_tokens

    def batch_decode(self, token_ids):
        pass
