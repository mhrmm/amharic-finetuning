import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm
from myutil import load_model_for_inference
from tokenization import SentencePieceTokenizer
from validate import translate


def translate_corpus(model, tokenizer, lines, src_lang, tgt_lang, batch_size, num_beams):
    """Translates a plain-text corpus, one sentence per line, in batches."""
    translations = []
    for i in tqdm(range(0, len(lines), batch_size), desc="Translating"):
        batch = lines[i : i + batch_size]
        src_ids = [tokenizer(sent, lang_code=src_lang) for sent in batch]
        max_len = max(len(ids) for ids in src_ids)
        pad_id = list(tokenizer.get_special_tokens().values())[0]  # first special token as pad
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in src_ids]
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = (input_ids != pad_id).int()
        encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
        translations.extend(
            translate(encoded, tokenizer, model, tgt_lang, num_beams=num_beams)
        )
    return translations


def read_jsonl(path):
    with open(path) as reader:
        return [json.loads(line) for line in reader if line.strip()]


def write_jsonl(path_or_writer, records):
    for record in records:
        path_or_writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Translates a plain-text corpus (one sentence per line) "
        "with a finetuned experiment checkpoint."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dir", type=str, help="Experiment directory with a finetuned checkpoint."
    )
    source.add_argument("--model", type=str, help="Path to a saved model directory.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to a SentencePiece model file (.model).",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        default=None,
        help="JSON object mapping special token names (e.g. '<pad>', '</s>', "
        "and language codes like 'eng_Latn') to reserved ids, matching the "
        "'special_tokens' entry of the tokenizer's experiment config.",
    )
    parser.add_argument(
        "--vocab-offset",
        type=int,
        default=0,
        help="Offset added to SentencePiece ids to make room for special "
        "tokens, matching the tokenizer's experiment config.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="File to translate. Either plain text (one sentence per line) or "
        "JSONL (one object per line, translating the 'source_text' field), "
        "detected from the .jsonl extension.",
    )
    parser.add_argument(
        "--src-lang", type=str, required=True, help="Source language code (e.g. eng_Latn)."
    )
    parser.add_argument(
        "--tgt-lang", type=str, required=True, help="Target language code (e.g. amh_Ethi)."
    )
    parser.add_argument(
        "--output", type=str, help="Where to write the translations. Defaults to stdout."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    model_name_or_path = args.dir if args.dir is not None else args.model
    use_fp16 = torch.cuda.is_available()
    model = load_model_for_inference(
        model_name_or_path, torch_dtype=torch.float16 if use_fp16 else torch.float32
    )
    if torch.cuda.is_available():
        model.cuda()

    special_tokens = json.loads(args.special_tokens) if args.special_tokens else None
    tokenizer = SentencePieceTokenizer(
        args.tokenizer,
        max_length=args.max_length,
        special_tokens=special_tokens,
        vocab_offset=args.vocab_offset,
    )

    is_jsonl = args.input.endswith(".jsonl")

    if is_jsonl:
        records = read_jsonl(args.input)
        lines = [record["source_text"] for record in records]
    else:
        with open(args.input) as reader:
            lines = [line.rstrip("\n") for line in reader]

    translations = translate_corpus(
        model, tokenizer, lines, args.src_lang, args.tgt_lang, args.batch_size, args.num_beams
    )

    if is_jsonl:
        for record, mt_text in zip(records, translations):
            record["mt_text"] = mt_text
            record["mt_lang"] = args.tgt_lang
        if args.output:
            with open(args.output, "w") as writer:
                write_jsonl(writer, records)
        else:
            write_jsonl(sys.stdout, records)
    elif args.output:
        with open(args.output, "w") as writer:
            for line in translations:
                writer.write(line + "\n")
    else:
        for line in translations:
            print(line)


if __name__ == "__main__":
    main()
