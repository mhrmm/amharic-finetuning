import unittest
from tokenization import SentencePieceTokenizer, ByteTokenizer
from torch import tensor


class TestTokenization(unittest.TestCase):

    @unittest.skip("NllbTokenizer removed — requires HuggingFace transformers")
    def test_nllb_tokenizer1(self):
        pass

    @unittest.skip("NllbTokenizer removed — requires HuggingFace transformers")
    def test_nllb_tokenizer2(self):
        pass

    @unittest.skip("NllbTokenizer removed — requires HuggingFace transformers")
    def test_hf_tokenizer_properties(self):
        pass

    def test_byte_tokenizer_encodes(self):
        tok = ByteTokenizer(max_length=20)
        result = tok("hi", lang_code="lang_Lang")
        # [lang_Lang id] + byte ids + [</s> id]
        self.assertEqual(result[0], tok.get_special_tokens()["lang_Lang"])
        self.assertEqual(result[-1], tok.get_special_tokens()["</s>"])
        # 'hi' = bytes 104, 105 → ids 104+3, 105+3
        self.assertEqual(result[1], 104 + len(tok.get_special_tokens()))
        self.assertEqual(result[2], 105 + len(tok.get_special_tokens()))

    def test_byte_tokenizer_truncates(self):
        tok = ByteTokenizer(max_length=5)
        result = tok("hello world", lang_code="lang_Lang")
        self.assertLessEqual(len(result), 5)

    def test_byte_tokenizer_len(self):
        tok = ByteTokenizer()
        self.assertEqual(len(tok), 256 + 3)  # 256 bytes + 3 special tokens

    def test_byte_tokenizer_special_tokens(self):
        tok = ByteTokenizer()
        st = tok.get_special_tokens()
        self.assertIn("<pad>", st)
        self.assertIn("</s>", st)
        self.assertIn("lang_Lang", st)

    @unittest.skip("requires a trained SentencePiece model file")
    def test_sentencepiece_tokenizer(self):
        tok = SentencePieceTokenizer("path/to/spm.model")
        result = tok("hello", lang_code="eng_Latn")
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
