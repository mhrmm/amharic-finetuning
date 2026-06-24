#!/bin/sh
#SBATCH -c 1
#SBATCH -t 10-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --config configs/en-am.json
#python scripts/translate_corpus.py --model facebook/nllb-200-distilled-600M --input ch2.jsonl --src-lang eng_Latn --tgt-lang amh_Ethi --output ch2.mt.jsonl