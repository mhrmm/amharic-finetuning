#!/bin/sh
#SBATCH -c 1
#SBATCH -t 10-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python scripts/translate_corpus.py --model facebook/nllb-200-distilled-600M --input /mnt/storage/data/am-en/afridoc/health/validation.en --src-lang eng_Latn --tgt-lang amh_Ethi