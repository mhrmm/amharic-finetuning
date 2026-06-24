import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM
from tokenization import HuggingfaceTokenizer
from validate import translate


def translate_corpus(model, tokenizer, lines, src_lang, tgt_lang, batch_size, num_beams):
    """Translates a plain-text corpus, one sentence per line, in batches."""
    tokenizer.tokenizer.src_lang = src_lang
    translations = []
    for i in tqdm(range(0, len(lines), batch_size), desc="Translating"):
        batch = lines[i : i + batch_size]
        encoded = tokenizer.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.max_length,
        )
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
        "with either a finetuned experiment or an off-the-shelf HuggingFace model."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dir", type=str, help="Experiment directory with a finetuned checkpoint."
    )
    source.add_argument("--model", type=str, help="HuggingFace model name or path.")
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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16 if use_fp16 else torch.float32
    )
    if torch.cuda.is_available():
        model.cuda()

    tokenizer = HuggingfaceTokenizer(model_name_or_path, max_length=args.max_length)

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
