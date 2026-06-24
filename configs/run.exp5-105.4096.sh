#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --config configs/exp5-84/experiment5-84.multi.4096.json
python finetune.py --config configs/exp5-84/experiment5-84.bi0.4096.json
python finetune.py --config configs/exp5-84/experiment5-84.bi1.4096.json
python finetune.py --config configs/exp5-85/experiment5-85.multi.4096.json
python finetune.py --config configs/exp5-85/experiment5-85.bi0.4096.json
python finetune.py --config configs/exp5-85/experiment5-85.bi1.4096.json
python finetune.py --config configs/exp5-86/experiment5-86.multi.4096.json
python finetune.py --config configs/exp5-86/experiment5-86.bi0.4096.json
python finetune.py --config configs/exp5-86/experiment5-86.bi1.4096.json
python finetune.py --config configs/exp5-87/experiment5-87.multi.4096.json
python finetune.py --config configs/exp5-87/experiment5-87.bi0.4096.json
python finetune.py --config configs/exp5-87/experiment5-87.bi1.4096.json
python finetune.py --config configs/exp5-88/experiment5-88.multi.4096.json
python finetune.py --config configs/exp5-88/experiment5-88.bi0.4096.json
python finetune.py --config configs/exp5-88/experiment5-88.bi1.4096.json
python finetune.py --config configs/exp5-89/experiment5-89.multi.4096.json
python finetune.py --config configs/exp5-89/experiment5-89.bi0.4096.json
python finetune.py --config configs/exp5-89/experiment5-89.bi1.4096.json
python finetune.py --config configs/exp5-90/experiment5-90.multi.4096.json
python finetune.py --config configs/exp5-90/experiment5-90.bi0.4096.json
python finetune.py --config configs/exp5-90/experiment5-90.bi1.4096.json
python finetune.py --config configs/exp5-91/experiment5-91.multi.4096.json
python finetune.py --config configs/exp5-91/experiment5-91.bi0.4096.json
python finetune.py --config configs/exp5-91/experiment5-91.bi1.4096.json
python finetune.py --config configs/exp5-92/experiment5-92.multi.4096.json
python finetune.py --config configs/exp5-92/experiment5-92.bi0.4096.json
python finetune.py --config configs/exp5-92/experiment5-92.bi1.4096.json
python finetune.py --config configs/exp5-93/experiment5-93.multi.4096.json
python finetune.py --config configs/exp5-93/experiment5-93.bi0.4096.json
python finetune.py --config configs/exp5-93/experiment5-93.bi1.4096.json
python finetune.py --config configs/exp5-94/experiment5-94.multi.4096.json
python finetune.py --config configs/exp5-94/experiment5-94.bi0.4096.json
python finetune.py --config configs/exp5-94/experiment5-94.bi1.4096.json
python finetune.py --config configs/exp5-95/experiment5-95.multi.4096.json
python finetune.py --config configs/exp5-95/experiment5-95.bi0.4096.json
python finetune.py --config configs/exp5-95/experiment5-95.bi1.4096.json
python finetune.py --config configs/exp5-96/experiment5-96.multi.4096.json
python finetune.py --config configs/exp5-96/experiment5-96.bi0.4096.json
python finetune.py --config configs/exp5-96/experiment5-96.bi1.4096.json
python finetune.py --config configs/exp5-97/experiment5-97.multi.4096.json
python finetune.py --config configs/exp5-97/experiment5-97.bi0.4096.json
python finetune.py --config configs/exp5-97/experiment5-97.bi1.4096.json
python finetune.py --config configs/exp5-98/experiment5-98.multi.4096.json
python finetune.py --config configs/exp5-98/experiment5-98.bi0.4096.json
python finetune.py --config configs/exp5-98/experiment5-98.bi1.4096.json
python finetune.py --config configs/exp5-99/experiment5-99.multi.4096.json
python finetune.py --config configs/exp5-99/experiment5-99.bi0.4096.json
python finetune.py --config configs/exp5-99/experiment5-99.bi1.4096.json
python finetune.py --config configs/exp5-100/experiment5-100.multi.4096.json
python finetune.py --config configs/exp5-100/experiment5-100.bi0.4096.json
python finetune.py --config configs/exp5-100/experiment5-100.bi1.4096.json
python finetune.py --config configs/exp5-101/experiment5-101.multi.4096.json
python finetune.py --config configs/exp5-101/experiment5-101.bi0.4096.json
python finetune.py --config configs/exp5-101/experiment5-101.bi1.4096.json
python finetune.py --config configs/exp5-102/experiment5-102.multi.4096.json
python finetune.py --config configs/exp5-102/experiment5-102.bi0.4096.json
python finetune.py --config configs/exp5-102/experiment5-102.bi1.4096.json
python finetune.py --config configs/exp5-103/experiment5-103.multi.4096.json
python finetune.py --config configs/exp5-103/experiment5-103.bi0.4096.json
python finetune.py --config configs/exp5-103/experiment5-103.bi1.4096.json
python finetune.py --config configs/exp5-104/experiment5-104.multi.4096.json
python finetune.py --config configs/exp5-104/experiment5-104.bi0.4096.json
python finetune.py --config configs/exp5-104/experiment5-104.bi1.4096.json