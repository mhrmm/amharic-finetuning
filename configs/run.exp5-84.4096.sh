#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --config configs/exp5-63/experiment5-63.multi.4096.json
python finetune.py --config configs/exp5-63/experiment5-63.bi0.4096.json
python finetune.py --config configs/exp5-63/experiment5-63.bi1.4096.json
python finetune.py --config configs/exp5-64/experiment5-64.multi.4096.json
python finetune.py --config configs/exp5-64/experiment5-64.bi0.4096.json
python finetune.py --config configs/exp5-64/experiment5-64.bi1.4096.json
python finetune.py --config configs/exp5-65/experiment5-65.multi.4096.json
python finetune.py --config configs/exp5-65/experiment5-65.bi0.4096.json
python finetune.py --config configs/exp5-65/experiment5-65.bi1.4096.json
python finetune.py --config configs/exp5-66/experiment5-66.multi.4096.json
python finetune.py --config configs/exp5-66/experiment5-66.bi0.4096.json
python finetune.py --config configs/exp5-66/experiment5-66.bi1.4096.json
python finetune.py --config configs/exp5-67/experiment5-67.multi.4096.json
python finetune.py --config configs/exp5-67/experiment5-67.bi0.4096.json
python finetune.py --config configs/exp5-67/experiment5-67.bi1.4096.json
python finetune.py --config configs/exp5-68/experiment5-68.multi.4096.json
python finetune.py --config configs/exp5-68/experiment5-68.bi0.4096.json
python finetune.py --config configs/exp5-68/experiment5-68.bi1.4096.json
python finetune.py --config configs/exp5-69/experiment5-69.multi.4096.json
python finetune.py --config configs/exp5-69/experiment5-69.bi0.4096.json
python finetune.py --config configs/exp5-69/experiment5-69.bi1.4096.json
python finetune.py --config configs/exp5-70/experiment5-70.multi.4096.json
python finetune.py --config configs/exp5-70/experiment5-70.bi0.4096.json
python finetune.py --config configs/exp5-70/experiment5-70.bi1.4096.json
python finetune.py --config configs/exp5-71/experiment5-71.multi.4096.json
python finetune.py --config configs/exp5-71/experiment5-71.bi0.4096.json
python finetune.py --config configs/exp5-71/experiment5-71.bi1.4096.json
python finetune.py --config configs/exp5-72/experiment5-72.multi.4096.json
python finetune.py --config configs/exp5-72/experiment5-72.bi0.4096.json
python finetune.py --config configs/exp5-72/experiment5-72.bi1.4096.json
python finetune.py --config configs/exp5-73/experiment5-73.multi.4096.json
python finetune.py --config configs/exp5-73/experiment5-73.bi0.4096.json
python finetune.py --config configs/exp5-73/experiment5-73.bi1.4096.json
python finetune.py --config configs/exp5-74/experiment5-74.multi.4096.json
python finetune.py --config configs/exp5-74/experiment5-74.bi0.4096.json
python finetune.py --config configs/exp5-74/experiment5-74.bi1.4096.json
python finetune.py --config configs/exp5-75/experiment5-75.multi.4096.json
python finetune.py --config configs/exp5-75/experiment5-75.bi0.4096.json
python finetune.py --config configs/exp5-75/experiment5-75.bi1.4096.json
python finetune.py --config configs/exp5-76/experiment5-76.multi.4096.json
python finetune.py --config configs/exp5-76/experiment5-76.bi0.4096.json
python finetune.py --config configs/exp5-76/experiment5-76.bi1.4096.json
python finetune.py --config configs/exp5-77/experiment5-77.multi.4096.json
python finetune.py --config configs/exp5-77/experiment5-77.bi0.4096.json
python finetune.py --config configs/exp5-77/experiment5-77.bi1.4096.json
python finetune.py --config configs/exp5-78/experiment5-78.multi.4096.json
python finetune.py --config configs/exp5-78/experiment5-78.bi0.4096.json
python finetune.py --config configs/exp5-78/experiment5-78.bi1.4096.json
python finetune.py --config configs/exp5-79/experiment5-79.multi.4096.json
python finetune.py --config configs/exp5-79/experiment5-79.bi0.4096.json
python finetune.py --config configs/exp5-79/experiment5-79.bi1.4096.json
python finetune.py --config configs/exp5-80/experiment5-80.multi.4096.json
python finetune.py --config configs/exp5-80/experiment5-80.bi0.4096.json
python finetune.py --config configs/exp5-80/experiment5-80.bi1.4096.json
python finetune.py --config configs/exp5-81/experiment5-81.multi.4096.json
python finetune.py --config configs/exp5-81/experiment5-81.bi0.4096.json
python finetune.py --config configs/exp5-81/experiment5-81.bi1.4096.json
python finetune.py --config configs/exp5-82/experiment5-82.multi.4096.json
python finetune.py --config configs/exp5-82/experiment5-82.bi0.4096.json
python finetune.py --config configs/exp5-82/experiment5-82.bi1.4096.json
python finetune.py --config configs/exp5-83/experiment5-83.multi.4096.json
python finetune.py --config configs/exp5-83/experiment5-83.bi0.4096.json
python finetune.py --config configs/exp5-83/experiment5-83.bi1.4096.json
