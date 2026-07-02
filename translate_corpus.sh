#!/bin/sh
#SBATCH -c 1
#SBATCH -t 10-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python scripts/translate_corpus.py --model facebook/nllb-200-distilled-600M --tokenizer /mnt/storage/models/nllb-200/sentencepiece.bpe.model --special-tokens '{"<pad>": 1, "</s>": 2, "eng_Latn": 256047, "amh_Ethi": 256009}' --vocab-offset 1 --input /mnt/storage/data/am-en/afridoc/health/validation.en --src-lang eng_Latn --tgt-lang amh_Ethi
