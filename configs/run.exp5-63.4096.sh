#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --config configs/exp5-42/experiment5-42.multi.4096.json
python finetune.py --config configs/exp5-42/experiment5-42.bi0.4096.json
python finetune.py --config configs/exp5-42/experiment5-42.bi1.4096.json
python finetune.py --config configs/exp5-43/experiment5-43.multi.4096.json
python finetune.py --config configs/exp5-43/experiment5-43.bi0.4096.json
python finetune.py --config configs/exp5-43/experiment5-43.bi1.4096.json
python finetune.py --config configs/exp5-44/experiment5-44.multi.4096.json
python finetune.py --config configs/exp5-44/experiment5-44.bi0.4096.json
python finetune.py --config configs/exp5-44/experiment5-44.bi1.4096.json
python finetune.py --config configs/exp5-45/experiment5-45.multi.4096.json
python finetune.py --config configs/exp5-45/experiment5-45.bi0.4096.json
python finetune.py --config configs/exp5-45/experiment5-45.bi1.4096.json
python finetune.py --config configs/exp5-46/experiment5-46.multi.4096.json
python finetune.py --config configs/exp5-46/experiment5-46.bi0.4096.json
python finetune.py --config configs/exp5-46/experiment5-46.bi1.4096.json
python finetune.py --config configs/exp5-47/experiment5-47.multi.4096.json
python finetune.py --config configs/exp5-47/experiment5-47.bi0.4096.json
python finetune.py --config configs/exp5-47/experiment5-47.bi1.4096.json
python finetune.py --config configs/exp5-48/experiment5-48.multi.4096.json
python finetune.py --config configs/exp5-48/experiment5-48.bi0.4096.json
python finetune.py --config configs/exp5-48/experiment5-48.bi1.4096.json
python finetune.py --config configs/exp5-49/experiment5-49.multi.4096.json
python finetune.py --config configs/exp5-49/experiment5-49.bi0.4096.json
python finetune.py --config configs/exp5-49/experiment5-49.bi1.4096.json
python finetune.py --config configs/exp5-50/experiment5-50.multi.4096.json
python finetune.py --config configs/exp5-50/experiment5-50.bi0.4096.json
python finetune.py --config configs/exp5-50/experiment5-50.bi1.4096.json
python finetune.py --config configs/exp5-51/experiment5-51.multi.4096.json
python finetune.py --config configs/exp5-51/experiment5-51.bi0.4096.json
python finetune.py --config configs/exp5-51/experiment5-51.bi1.4096.json
python finetune.py --config configs/exp5-52/experiment5-52.multi.4096.json
python finetune.py --config configs/exp5-52/experiment5-52.bi0.4096.json
python finetune.py --config configs/exp5-52/experiment5-52.bi1.4096.json
python finetune.py --config configs/exp5-53/experiment5-53.multi.4096.json
python finetune.py --config configs/exp5-53/experiment5-53.bi0.4096.json
python finetune.py --config configs/exp5-53/experiment5-53.bi1.4096.json
python finetune.py --config configs/exp5-54/experiment5-54.multi.4096.json
python finetune.py --config configs/exp5-54/experiment5-54.bi0.4096.json
python finetune.py --config configs/exp5-54/experiment5-54.bi1.4096.json
python finetune.py --config configs/exp5-55/experiment5-55.multi.4096.json
python finetune.py --config configs/exp5-55/experiment5-55.bi0.4096.json
python finetune.py --config configs/exp5-55/experiment5-55.bi1.4096.json
python finetune.py --config configs/exp5-56/experiment5-56.multi.4096.json
python finetune.py --config configs/exp5-56/experiment5-56.bi0.4096.json
python finetune.py --config configs/exp5-56/experiment5-56.bi1.4096.json
python finetune.py --config configs/exp5-57/experiment5-57.multi.4096.json
python finetune.py --config configs/exp5-57/experiment5-57.bi0.4096.json
python finetune.py --config configs/exp5-57/experiment5-57.bi1.4096.json
python finetune.py --config configs/exp5-58/experiment5-58.multi.4096.json
python finetune.py --config configs/exp5-58/experiment5-58.bi0.4096.json
python finetune.py --config configs/exp5-58/experiment5-58.bi1.4096.json
python finetune.py --config configs/exp5-59/experiment5-59.multi.4096.json
python finetune.py --config configs/exp5-59/experiment5-59.bi0.4096.json
python finetune.py --config configs/exp5-59/experiment5-59.bi1.4096.json
python finetune.py --config configs/exp5-60/experiment5-60.multi.4096.json
python finetune.py --config configs/exp5-60/experiment5-60.bi0.4096.json
python finetune.py --config configs/exp5-60/experiment5-60.bi1.4096.json
python finetune.py --config configs/exp5-61/experiment5-61.multi.4096.json
python finetune.py --config configs/exp5-61/experiment5-61.bi0.4096.json
python finetune.py --config configs/exp5-61/experiment5-61.bi1.4096.json
python finetune.py --config configs/exp5-62/experiment5-62.multi.4096.json
python finetune.py --config configs/exp5-62/experiment5-62.bi0.4096.json
python finetune.py --config configs/exp5-62/experiment5-62.bi1.4096.json
