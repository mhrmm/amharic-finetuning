#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --config configs/exp5-21/experiment5-21.multi.4096.json
python finetune.py --config configs/exp5-21/experiment5-21.bi0.4096.json
python finetune.py --config configs/exp5-21/experiment5-21.bi1.4096.json
python finetune.py --config configs/exp5-22/experiment5-22.multi.4096.json
python finetune.py --config configs/exp5-22/experiment5-22.bi0.4096.json
python finetune.py --config configs/exp5-22/experiment5-22.bi1.4096.json
python finetune.py --config configs/exp5-23/experiment5-23.multi.4096.json
python finetune.py --config configs/exp5-23/experiment5-23.bi0.4096.json
python finetune.py --config configs/exp5-23/experiment5-23.bi1.4096.json
python finetune.py --config configs/exp5-24/experiment5-24.multi.4096.json
python finetune.py --config configs/exp5-24/experiment5-24.bi0.4096.json
python finetune.py --config configs/exp5-24/experiment5-24.bi1.4096.json
python finetune.py --config configs/exp5-25/experiment5-25.multi.4096.json
python finetune.py --config configs/exp5-25/experiment5-25.bi0.4096.json
python finetune.py --config configs/exp5-25/experiment5-25.bi1.4096.json
python finetune.py --config configs/exp5-26/experiment5-26.multi.4096.json
python finetune.py --config configs/exp5-26/experiment5-26.bi0.4096.json
python finetune.py --config configs/exp5-26/experiment5-26.bi1.4096.json
python finetune.py --config configs/exp5-27/experiment5-27.multi.4096.json
python finetune.py --config configs/exp5-27/experiment5-27.bi0.4096.json
python finetune.py --config configs/exp5-27/experiment5-27.bi1.4096.json
python finetune.py --config configs/exp5-28/experiment5-28.multi.4096.json
python finetune.py --config configs/exp5-28/experiment5-28.bi0.4096.json
python finetune.py --config configs/exp5-28/experiment5-28.bi1.4096.json
python finetune.py --config configs/exp5-29/experiment5-29.multi.4096.json
python finetune.py --config configs/exp5-29/experiment5-29.bi0.4096.json
python finetune.py --config configs/exp5-29/experiment5-29.bi1.4096.json
python finetune.py --config configs/exp5-30/experiment5-30.multi.4096.json
python finetune.py --config configs/exp5-30/experiment5-30.bi0.4096.json
python finetune.py --config configs/exp5-30/experiment5-30.bi1.4096.json
python finetune.py --config configs/exp5-31/experiment5-31.multi.4096.json
python finetune.py --config configs/exp5-31/experiment5-31.bi0.4096.json
python finetune.py --config configs/exp5-31/experiment5-31.bi1.4096.json
python finetune.py --config configs/exp5-32/experiment5-32.multi.4096.json
python finetune.py --config configs/exp5-32/experiment5-32.bi0.4096.json
python finetune.py --config configs/exp5-32/experiment5-32.bi1.4096.json
python finetune.py --config configs/exp5-33/experiment5-33.multi.4096.json
python finetune.py --config configs/exp5-33/experiment5-33.bi0.4096.json
python finetune.py --config configs/exp5-33/experiment5-33.bi1.4096.json
python finetune.py --config configs/exp5-34/experiment5-34.multi.4096.json
python finetune.py --config configs/exp5-34/experiment5-34.bi0.4096.json
python finetune.py --config configs/exp5-34/experiment5-34.bi1.4096.json
python finetune.py --config configs/exp5-35/experiment5-35.multi.4096.json
python finetune.py --config configs/exp5-35/experiment5-35.bi0.4096.json
python finetune.py --config configs/exp5-35/experiment5-35.bi1.4096.json
python finetune.py --config configs/exp5-36/experiment5-36.multi.4096.json
python finetune.py --config configs/exp5-36/experiment5-36.bi0.4096.json
python finetune.py --config configs/exp5-36/experiment5-36.bi1.4096.json
python finetune.py --config configs/exp5-37/experiment5-37.multi.4096.json
python finetune.py --config configs/exp5-37/experiment5-37.bi0.4096.json
python finetune.py --config configs/exp5-37/experiment5-37.bi1.4096.json
python finetune.py --config configs/exp5-38/experiment5-38.multi.4096.json
python finetune.py --config configs/exp5-38/experiment5-38.bi0.4096.json
python finetune.py --config configs/exp5-38/experiment5-38.bi1.4096.json
python finetune.py --config configs/exp5-39/experiment5-39.multi.4096.json
python finetune.py --config configs/exp5-39/experiment5-39.bi0.4096.json
python finetune.py --config configs/exp5-39/experiment5-39.bi1.4096.json
python finetune.py --config configs/exp5-40/experiment5-40.multi.4096.json
python finetune.py --config configs/exp5-40/experiment5-40.bi0.4096.json
python finetune.py --config configs/exp5-40/experiment5-40.bi1.4096.json
python finetune.py --config configs/exp5-41/experiment5-41.multi.4096.json
python finetune.py --config configs/exp5-41/experiment5-41.bi0.4096.json
python finetune.py --config configs/exp5-41/experiment5-41.bi1.4096.json
