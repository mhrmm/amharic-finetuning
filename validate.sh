#!/bin/sh
#SBATCH -c 1
#SBATCH -t 7-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python validate.py --model facebook/nllb-200-3.3B --config experiments/en-am-nllb-3.3B-lora-v0/experiment.json --out experiments/en-am-nllb-3.3B-lora-v0