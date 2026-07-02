import argparse
import os
from pathlib import Path
from random import shuffle
from tqdm import tqdm


def reorganize(batch_size, root_dir, split, output_dir, tokenizer_model):
    """
    Reorganizes text files by sorting lines by token length and shuffling in batches.

    Tokenizes each English line using a SentencePiece model, sorts all lines by
    their tokenized length, chunks them into batches, shuffles the batch order,
    and reorders all split-related files accordingly.  The reorganized files are
    saved to `output_dir`.

    Parameters
    ----------
    batch_size : int
        Number of lines per shuffled chunk.
    root_dir : Path
        Path to the directory containing input files named like `<split>.*`.
    split : str
        Prefix of the files to process (e.g., "train" for "train.en", "train.fr").
    output_dir : Path
        Path to the directory where reorganized files will be written.
    tokenizer_model : str
        Path to a SentencePiece model file (.model) used for length estimation.
    """
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError(
            "sentencepiece is required. Install with: pip install sentencepiece"
        )

    os.makedirs(output_dir, exist_ok=True)
    files = list(root_dir.glob(f"{split}.*"))

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model)

    lengths = []
    with open(root_dir / f"{split}.en") as reader:
        for i, line in tqdm(enumerate(reader)):
            tokens = sp.EncodeAsIds(line.strip())
            lengths.append((len(tokens), i))

    line_nums_by_length = [line_num for _, line_num in sorted(lengths)]
    num_full_chunks = len(line_nums_by_length) // batch_size
    remainder_start = num_full_chunks * batch_size
    chunk_starts = [batch_size * k for k in range(num_full_chunks)]
    shuffle(chunk_starts)
    line_nums = []
    for start in chunk_starts:
        line_nums.extend(line_nums_by_length[start : start + batch_size])
    if remainder_start < len(line_nums_by_length):
        line_nums.extend(line_nums_by_length[remainder_start:])

    for file in files:
        lines = []
        with open(file) as reader:
            for line in tqdm(reader):
                lines.append(line.strip())
        with open(output_dir / file.name, "w") as writer:
            for num in tqdm(line_nums):
                writer.write(lines[num] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorders the sentences of a parallel corpus so that batched sentences have similar lengths."
    )
    parser.add_argument(
        "--in_dir", type=str, required=True, help="Directory with the original files."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for storing the new, reordered files.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Desired batch size.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Prefix of the files to process (e.g. 'train' for train.en, train.fr, ...).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to a SentencePiece model file (.model) used to estimate token lengths.",
    )
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    reorganize(args.batch_size, in_dir, args.split, out_dir, args.tokenizer)
