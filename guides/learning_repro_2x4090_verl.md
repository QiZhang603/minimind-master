# MiniMind Learning Repro Roadmap (Linux 2x4090 + verl)

This guide is for learning-oriented reproduction, not just one-click run.

## 1. Environment setup on Linux

```bash
conda create -n minimind python=3.10 -y
conda activate minimind
pip install -r requirements.txt
pip install wandb datasets pyarrow ray verl
```

Optional but recommended for monitoring:

```bash
wandb login
```

## 2. Data preparation

Place your original MiniMind files under dataset, for example:

- dataset/rlaif.jsonl

Convert rlaif jsonl into verl parquet:

```bash
python trainer/verl_tools/convert_minimind_rlaif_to_verl.py \
  --input dataset/rlaif.jsonl \
  --train_output dataset/verl/rlaif_train.parquet \
  --val_output dataset/verl/rlaif_val.parquet \
  --val_ratio 0.02
```

## 3. Learning sequence and what to observe

### Step A: Pretrain (LM basics)

```bash
python trainer/train_pretrain.py --use_wandb --tracker wandb --wandb_project MiniMind-Pretrain
```

Focus:
- loss convergence speed
- tokens/s and stability
- gradient behavior in long training

### Step B: Full SFT (instruction alignment)

```bash
python trainer/train_full_sft.py --use_wandb --tracker wandb --wandb_project MiniMind-Full-SFT
```

Focus:
- instruction-following quality
- behavior change from pretrain checkpoint
- overfitting signs

### Step C: LoRA (parameter-efficient tuning)

```bash
python trainer/train_lora.py --use_wandb --tracker wandb --wandb_project MiniMind-LoRA
```

Focus:
- quality delta vs full SFT
- speed/memory tradeoff
- domain transfer efficiency

### Step D: Native MiniMind GRPO baseline

```bash
python trainer/train_grpo.py --use_wandb --tracker wandb --wandb_project MiniMind-GRPO
```

Focus:
- reward trends and variance
- kl_ref and policy loss coupling
- response length and exploration behavior

### Step E: verl GRPO (engineering-scale RL flow)

First export your MiniMind SFT checkpoint to Hugging Face style:

```bash
python scripts/convert_model.py
```

Then run verl GRPO:

```bash
bash trainer/verl_tools/run_grpo_verl_2x4090.sh
```

Focus:
- compare reward/loss curves with native GRPO
- compare throughput and GPU memory behavior
- inspect distributed task graph in Ray

## 4. Ray debugger and timeline

Use debug preset:

```bash
bash trainer/verl_tools/run_grpo_verl_debug.sh
```

Then:

1. Open Ray dashboard: http://<server_ip>:8265
2. Open timeline json from out/verl_grpo_debug_ckpt/ray_timeline.json in https://ui.perfetto.dev
3. Track rollout, reward, and update stages to understand GRPO data flow

## 5. Notes on experiment tracking

Training scripts now support tracker backend switching:

- tracker=swanlab (default)
- tracker=wandb

Example with proxy and entity:

```bash
python trainer/train_full_sft.py \
  --use_wandb \
  --tracker wandb \
  --wandb_project MiniMind-Full-SFT \
  --wandb_entity your_team \
  --wandb_proxy http://127.0.0.1:7890
```
