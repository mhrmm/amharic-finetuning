#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --config configs/exp5-0/experiment5-0.multi.4096.json
python finetune.py --config configs/exp5-0/experiment5-0.bi0.4096.json
python finetune.py --config configs/exp5-0/experiment5-0.bi1.4096.json
python finetune.py --config configs/exp5-1/experiment5-1.multi.4096.json
python finetune.py --config configs/exp5-1/experiment5-1.bi0.4096.json
python finetune.py --config configs/exp5-1/experiment5-1.bi1.4096.json
python finetune.py --config configs/exp5-2/experiment5-2.multi.4096.json
python finetune.py --config configs/exp5-2/experiment5-2.bi0.4096.json
python finetune.py --config configs/exp5-2/experiment5-2.bi1.4096.json
python finetune.py --config configs/exp5-3/experiment5-3.multi.4096.json
python finetune.py --config configs/exp5-3/experiment5-3.bi0.4096.json
python finetune.py --config configs/exp5-3/experiment5-3.bi1.4096.json
python finetune.py --config configs/exp5-4/experiment5-4.multi.4096.json
python finetune.py --config configs/exp5-4/experiment5-4.bi0.4096.json
python finetune.py --config configs/exp5-4/experiment5-4.bi1.4096.json
python finetune.py --config configs/exp5-5/experiment5-5.multi.4096.json
python finetune.py --config configs/exp5-5/experiment5-5.bi0.4096.json
python finetune.py --config configs/exp5-5/experiment5-5.bi1.4096.json
python finetune.py --config configs/exp5-6/experiment5-6.multi.4096.json
python finetune.py --config configs/exp5-6/experiment5-6.bi0.4096.json
python finetune.py --config configs/exp5-6/experiment5-6.bi1.4096.json
python finetune.py --config configs/exp5-7/experiment5-7.multi.4096.json
python finetune.py --config configs/exp5-7/experiment5-7.bi0.4096.json
python finetune.py --config configs/exp5-7/experiment5-7.bi1.4096.json
python finetune.py --config configs/exp5-8/experiment5-8.multi.4096.json
python finetune.py --config configs/exp5-8/experiment5-8.bi0.4096.json
python finetune.py --config configs/exp5-8/experiment5-8.bi1.4096.json
python finetune.py --config configs/exp5-9/experiment5-9.multi.4096.json
python finetune.py --config configs/exp5-9/experiment5-9.bi0.4096.json
python finetune.py --config configs/exp5-9/experiment5-9.bi1.4096.json
python finetune.py --config configs/exp5-10/experiment5-10.multi.4096.json
python finetune.py --config configs/exp5-10/experiment5-10.bi0.4096.json
python finetune.py --config configs/exp5-10/experiment5-10.bi1.4096.json
python finetune.py --config configs/exp5-11/experiment5-11.multi.4096.json
python finetune.py --config configs/exp5-11/experiment5-11.bi0.4096.json
python finetune.py --config configs/exp5-11/experiment5-11.bi1.4096.json
python finetune.py --config configs/exp5-12/experiment5-12.multi.4096.json
python finetune.py --config configs/exp5-12/experiment5-12.bi0.4096.json
python finetune.py --config configs/exp5-12/experiment5-12.bi1.4096.json
python finetune.py --config configs/exp5-13/experiment5-13.multi.4096.json
python finetune.py --config configs/exp5-13/experiment5-13.bi0.4096.json
python finetune.py --config configs/exp5-13/experiment5-13.bi1.4096.json
python finetune.py --config configs/exp5-14/experiment5-14.multi.4096.json
python finetune.py --config configs/exp5-14/experiment5-14.bi0.4096.json
python finetune.py --config configs/exp5-14/experiment5-14.bi1.4096.json
python finetune.py --config configs/exp5-15/experiment5-15.multi.4096.json
python finetune.py --config configs/exp5-15/experiment5-15.bi0.4096.json
python finetune.py --config configs/exp5-15/experiment5-15.bi1.4096.json
python finetune.py --config configs/exp5-16/experiment5-16.multi.4096.json
python finetune.py --config configs/exp5-16/experiment5-16.bi0.4096.json
python finetune.py --config configs/exp5-16/experiment5-16.bi1.4096.json
python finetune.py --config configs/exp5-17/experiment5-17.multi.4096.json
python finetune.py --config configs/exp5-17/experiment5-17.bi0.4096.json
python finetune.py --config configs/exp5-17/experiment5-17.bi1.4096.json
python finetune.py --config configs/exp5-18/experiment5-18.multi.4096.json
python finetune.py --config configs/exp5-18/experiment5-18.bi0.4096.json
python finetune.py --config configs/exp5-18/experiment5-18.bi1.4096.json
python finetune.py --config configs/exp5-19/experiment5-19.multi.4096.json
python finetune.py --config configs/exp5-19/experiment5-19.bi0.4096.json
python finetune.py --config configs/exp5-19/experiment5-19.bi1.4096.json
python finetune.py --config configs/exp5-20/experiment5-20.multi.4096.json
python finetune.py --config configs/exp5-20/experiment5-20.bi0.4096.json
python finetune.py --config configs/exp5-20/experiment5-20.bi1.4096.json
