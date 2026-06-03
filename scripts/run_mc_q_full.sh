#!/bin/bash
cd /root/autodl-tmp/pokestrategist
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pokestrategist
rm -rf runs/pokestrategist_v1_metamon_gen9ou_1550_mc_q_full_b
python -u -m pokestrategist.cli.train_rl \
  --data data/processed/pokestrategist_v1_metamon_gen9ou_1550_full.jsonl \
  --checkpoint runs/pokestrategist_v1_metamon_gen9ou_1550_opt_unified_stage3/model.pt \
  --output-dir runs/pokestrategist_v1_metamon_gen9ou_1550_mc_q_full_b \
  --epochs 4 \
  --batch-size 256 \
  --device cuda \
  --num-workers 8 \
  --eval-num-workers 0 \
  --prefetch-factor 4 \
  --learning-rate 3e-4 \
  --warmup-epochs 0 \
  --min-learning-rate-ratio 0.1 \
  > logs/pokestrategist_v1_metamon_gen9ou_1550_mc_q_full_b.log 2>&1
