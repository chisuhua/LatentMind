# Phase 0 实施指南：脚本、配置与 SOP

> **一句话定位**：Phase 0（共享前置验证）的**可执行手册**——每个 P.0.x 的训练脚本、配置文件、监控仪表盘、错误处理 SOP、验收脚本都完整给出，配合 [logos-sadko-64m-coordination.md](../research/logos-sadko-64m-coordination.md) 使用。
> **性质**：协调文档的**配套执行手册**
> **最后更新**：2026-07-29（v1.0 新增）

---

## 0. 文档定位与环境要求

### 0.1 与协调文档的关系

| 文档 | 关系 |
|------|------|
| [logos-sadko-64m-coordination.md](../research/logos-sadko-64m-coordination.md) | "为什么做" - 协调框架 |
| **本文档** | **"怎么做" - 执行手册** |

### 0.2 硬件要求

| 资源 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3090 24GB | RTX 4090 24GB |
| CPU | 16 cores | 32 cores |
| 内存 | 64GB | 128GB |
| 存储 | 1TB SSD | 2TB NVMe |
| 网络 | 1Gbps（用于下载模型/数据）| — |

### 0.3 Python 环境

```yaml
# environment.yml
name: latentmind_phase0
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch=2.1.0
  - pytorch-cuda=12.1
  - numpy=1.26
  - tqdm=4.66
  - wandb=0.16
  - transformers=4.36
  - tokenizers=0.15
  - datasets=2.14
  - pip
  - pip:
    - flash-attn==2.3.3
    - lm-eval==0.4.0
    - trl==0.8.0
```

### 0.4 仓库结构

```
LatentMind/
├── scripts/
│   ├── phase0/
│   │   ├── p01_minimind3_baseline/
│   │   │   ├── train.sh
│   │   │   ├── eval_c4_ppl.py
│   │   │   ├── monitor.py
│   │   │   └── config.yaml
│   │   ├── p02_perception_cnn/
│   │   │   ├── train_cnn.py
│   │   │   ├── cifar10_loader.py
│   │   │   ├── alignment_loss.py
│   │   │   └── config.yaml
│   │   ├── p03_data_pipeline/
│   │   │   ├── build_dataset.py
│   │   │   └── dataloader.py
│   │   ├── p04_train_infra/
│   │   │   ├── bf16_trainer.py
│   │   │   ├── checkpoint.py
│   │   │   └── resume.py
│   │   ├── p05_eval_framework/
│   │   │   ├── unified_eval.py
│   │   │   ├── metrics/
│   │   │   │   ├── ppl.py
│   │   │   │   ├── gsm8k.py
│   │   │   │   ├── multi_path.py
│   │   │   │   └── latency.py
│   │   │   └── config.yaml
│   │   └── p06_edge_infra/
│   │       ├── radix_cache.py
│   │       ├── per_token_exit.py
│   │       └── hierarchical.py
│   └── utils/
│       ├── logging.py
│       └── error_handler.py
└── logs/
    └── phase0/
```

---

## 1. P.0.1 MiniMind3 64M 从零训练基线

### 1.1 训练脚本（`scripts/phase0/p01_minimind3_baseline/train.sh`）

```bash
#!/bin/bash
# Phase 0 P.0.1：MiniMind3 64M 从零训练基线
# 目标：C4 PPL < 3.5（合格）/ < 3.0（优秀）
set -euo pipefail

# 配置
export P0_CONFIG="scripts/phase0/p01_minimind3_baseline/config.yaml"
export P0_OUTPUT_DIR="checkpoints/phase0/p01_minimind3_baseline"
export P0_LOG_DIR="logs/phase0/p01"
export P0_DATA_SUBSET="minimind3_pretrain_4b_subset"

mkdir -p "$P0_OUTPUT_DIR" "$P0_LOG_DIR"

# 启动 W&B 监控
wandb login
wandb init --project latentmind-phase0 --name p01_minimind3_baseline

# 关键环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# 启动训练（参考 MiniMind3 train_pretrain.py）
python -m scripts.phase0.p01_minimind3_baseline.train \
    --config "$P0_CONFIG" \
    --output_dir "$P0_OUTPUT_DIR" \
    --log_dir "$P0_LOG_DIR" \
    --data_subset "$P0_DATA_SUBSET" \
    --precision bf16 \
    --gradient_checkpointing \
    2>&1 | tee "$P0_LOG_DIR/train.log"

# 训练完成后自动评估
python -m scripts.phase0.p01_minimind3_baseline.eval_c4_ppl \
    --checkpoint "$P0_OUTPUT_DIR/final.pt" \
    --output "$P0_LOG_DIR/c4_ppl.json" \
    2>&1 | tee "$P0_LOG_DIR/eval.log"

# 生成验收报告
python -m scripts.phase0.utils.generate_p0_report \
    --task p01 \
    --log_dir "$P0_LOG_DIR" \
    --output "reports/p01_validation.md"
```

### 1.2 配置 YAML（`config.yaml`）

```yaml
# scripts/phase0/p01_minimind3_baseline/config.yaml
model:
  type: minimind3_dense_64m
  vocab_size: 6400
  hidden_size: 768
  num_hidden_layers: 8
  num_attention_heads: 16
  num_key_value_heads: 8
  head_dim: 48
  intermediate_size: 2048
  max_position_embeddings: 4096
  rope_theta: 10000.0

training:
  # 从零训练（不加载任何预训练权重）
  init_from_scratch: true
  load_pretrained: false  # 🔴 必须 false

  # 数据
  data_subset: minimind3_pretrain_4b_subset
  total_tokens: 4_000_000_000
  sequence_length: 2048
  batch_size: 64
  gradient_accumulation_steps: 1

  # 优化器
  optimizer: AdamW
  learning_rate: 1.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  warmup_steps: 2000
  scheduler: cosine  # constant_after_warmup 可选

  # 精度
  precision: bf16
  gradient_checkpointing: true

  # 训练时长
  max_train_steps: 100000  # 约 2-4 hours 单卡 3090
  eval_interval: 2000
  save_interval: 5000
  log_interval: 50

# 评估
eval:
  batch_size: 16
  sequence_length: 2048
  max_eval_samples: 10000  # C4 验证集子集

# 监控
monitoring:
  wandb_project: latentmind-phase0
  track_loss_every: 50
  track_lr_every: 50
  track_grad_norm_every: 50
  track_memory_every: 500
```

### 1.3 监控仪表盘（`monitor.py`）

```python
# scripts/phase0/p01_minimind3_baseline/monitor.py
"""P.0.1 训练监控——检测异常并报警"""
import torch
import wandb
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainMonitor:
    """训练监控器——检测异常并按 SOP 报警"""
    patience_loss: int = 500      # loss 不降容忍步数
    min_loss_delta: float = 0.01  # 最小 loss 下降幅度
    nan_check_steps: int = 10    # NaN 检测间隔
    grad_norm_threshold: float = 10.0  # 梯度爆炸阈值
    memory_threshold_pct: float = 0.95  # 显存使用阈值

    def __init__(self):
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.alerts = []

    def step(self, loss: float, grad_norm: float, mem_used: int, mem_total: int):
        """每个训练 step 调用"""
        # 1. NaN 检测
        if loss != loss:  # NaN
            self.alert("NaN", f"Loss is NaN at step, immediate action required")
            return "abort"

        # 2. 梯度爆炸
        if grad_norm > self.grad_norm_threshold:
            self.alert("GRAD_EXPLOSION", f"grad_norm={grad_norm:.2f} > {self.grad_norm_threshold}")
            return "clip_grad"

        # 3. 显存不足
        mem_pct = mem_used / mem_total
        if mem_pct > self.memory_threshold_pct:
            self.alert("OOM_RISK", f"Memory {mem_pct:.1%} > {self.memory_threshold_pct:.0%}")
            return "reduce_batch"

        # 4. Loss 不降
        if loss < self.best_loss - self.min_loss_delta:
            self.best_loss = loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.patience_loss:
            self.alert("LOSS_STAGNANT",
                f"Loss not decreasing for {self.patience_loss} steps (best={self.best_loss:.4f})")
            return "investigate"

        # 5. 正常情况
        if self.steps_without_improvement % 100 == 0:
            self.alert("INFO", f"Steps without improvement: {self.steps_without_improvement}")
        return "continue"

    def alert(self, level: str, msg: str):
        """报警——记录 + W&B + 控制台"""
        alert = {"level": level, "message": msg}
        self.alerts.append(alert)
        wandb.log({"alert": level, "alert_msg": msg})
        print(f"🚨 [{level}] {msg}")
```

### 1.4 错误处理 SOP（最关键）

#### SOP-1：OOM（显存不足）

**触发条件**：训练中断，错误信息含 "OutOfMemoryError" 或 "CUDA OOM"

**标准操作**：
```
1. 检查当前显存使用：
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv

2. 按以下顺序降低显存：
   Level 1: 减小 batch_size（64 → 32 → 16）
   Level 2: 启用 gradient_accumulation_steps（4 → 8 → 16）补偿
   Level 3: 启用 gradient_checkpointing
   Level 4: 减小 sequence_length（2048 → 1024）

3. 重启训练，记录到 W&B
4. 失败 3 次后升级到 P.0 协调会议
```

#### SOP-2：NaN / Loss 爆炸

**触发条件**：训练中 loss 突然变为 NaN / Inf，或梯度范数 > 10

**标准操作**：
```
1. 立即停止训练（Ctrl+C）
2. 检查最近的 loss 曲线（W&B）
3. 加载上一个 checkpoint（save_interval 之前）
4. 应用修复：
   Fix A: 启用 gradient clipping max_norm=1.0
   Fix B: 降低 learning_rate（1e-4 → 5e-5 → 1e-5）
   Fix C: 检查数据（可能有损坏样本）
5. 从 checkpoint 恢复训练
6. 失败 3 次后升级
```

#### SOP-3：Loss 不下降

**触发条件**：连续 500 step loss 没有显著改善（delta < 0.01）

**标准操作**：
```
1. 检查 learning rate（可能过小/过大）
2. 检查 warmup 步数（可能不足）
3. 检查数据（可能有重复/错位）
4. 修复方案：
   Fix A: 调整学习率（×0.5 或 ×2）
   Fix B: 增加 warmup_steps
   Fix C: 重新运行数据 pipeline 验证
5. 失败后升级
```

#### SOP-4：训练崩溃（设备错误/系统重启）

**触发条件**：GPU 设备错误、系统重启、断电

**标准操作**：
```
1. 重启后检查最后一个 checkpoint
2. 运行 resume.py（断点恢复）
3. 验证 loss 曲线连续性
4. 如果 checkpoint 损坏 → 退回最近一个 save_interval 的 checkpoint
5. 如果全部损坏 → 重新训练（记录事故）
```

### 1.5 验收脚本（`eval_c4_ppl.py`）

```python
# scripts/phase0/p01_minimind3_baseline/eval_c4_ppl.py
"""P.0.1 验收：C4 PPL 测量"""
import torch
import json
import argparse
from pathlib import Path

def evaluate_c4_ppl(checkpoint_path: str, output_path: str):
    """评估 C4 验证集 PPL"""
    # 加载模型
    model = load_minimind3_from_checkpoint(checkpoint_path)
    model.eval()

    # 加载 C4 验证集子集
    test_loader = build_c4_eval_loader(
        max_samples=10000,
        seq_length=2048,
        batch_size=16,
    )

    # 计算 PPL
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].cuda()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            num_tokens = (input_ids != 0).sum().item()  # 排除 padding
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, current PPL: {torch.exp(total_loss / total_tokens).item():.4f}")

    final_ppl = torch.exp(total_loss / total_tokens).item()

    # 验收判定
    validation = {
        "final_ppl": final_ppl,
        "total_tokens_evaluated": total_tokens,
        "checkpoint": checkpoint_path,
        "verdict": (
            "EXCELLENT" if final_ppl < 3.0 else
            "PASS" if final_ppl < 3.5 else
            "FAIL"
        ),
        "thresholds": {
            "excellent": 3.0,
            "pass": 3.5,
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(validation, f, indent=2)

    print(f"\n✅ P.0.1 验收结果：{validation['verdict']}")
    print(f"   C4 PPL: {final_ppl:.4f}")
    return validation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate_c4_ppl(args.checkpoint, args.output)
```

---

## 2. P.0.2 3 层 CNN 感知层

### 2.1 模型定义（参考 SenseNova NEO-Unify）

```python
# scripts/phase0/p02_perception_cnn/perception_cnn.py
"""3 层 CNN 感知层——像素 → 768 维潜空间"""
import torch
import torch.nn as nn

class PerceptionCNN(nn.Module):
    """3 层 CNN 感知层

    输入：图像 (B, C, H, W)
    输出：latent (B, L, 768)  其中 L = (H/16) * (W/16)

    设计参考商汤 SenseNova NEO-Unify
    - Layer 1: Conv 3×3, stride=2 → (H/2, W/2)
    - Layer 2: Conv 3×3, stride=2 → (H/4, W/4)
    - Layer 3: Conv 3×3, stride=2 → (H/8, W/8)
    - 投影：Conv 1×1 → 768 维
    """
    def __init__(self, hidden_size=768, in_channels=3):
        super().__init__()
        self.hidden_size = hidden_size

        # 3 层 CNN（每层 2× 下采样）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # 投影到目标隐藏维度
        self.proj = nn.Linear(256, hidden_size)

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W) pixel tensor
        Returns:
            latent: (B, L, hidden_size) flattened patches
        """
        x = self.conv1(image)   # (B, 64, H/2, W/2)
        x = self.conv2(x)       # (B, 128, H/4, W/4)
        x = self.conv3(x)       # (B, 256, H/8, W/8)
        x = x.flatten(2).transpose(1, 2)  # (B, L, 256)
        x = self.proj(x)        # (B, L, 768)
        return x

    def get_num_patches(self, image_size):
        """计算 patch 数量"""
        return (image_size // 8) ** 2
```

### 2.2 联合损失函数（`alignment_loss.py`）

```python
# scripts/phase0/p02_perception_cnn/alignment_loss.py
"""联合损失：图像分类 + 文本-图像 latent 对齐"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentLoss(nn.Module):
    """P.0.2 损失函数

    联合训练两个目标：
    1. 图像分类（确保 CNN 提取有效特征）
    2. 对齐损失（确保感知层输出与文本 768 维对齐）
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, image_latent, text_latent, labels):
        """
        Args:
            image_latent: (B, L_img, 768) 图像感知层输出
            text_latent: (B, L_txt, 768) 文本 Embedding 输出
            labels: (B,) 分类标签
        """
        # 1. 分类损失（pool image latent）
        img_pooled = image_latent.mean(dim=1)  # (B, 768)
        logits = self.classifier(img_pooled)
        cls_loss = F.cross_entropy(logits, labels)

        # 2. 对齐损失（让图像和文本的潜空间分布一致）
        # 使用 mean-pooled 表示计算余弦相似度
        img_mean = image_latent.mean(dim=1)
        img_mean = F.normalize(img_mean, dim=-1)

        txt_pooled = text_latent.mean(dim=1)
        txt_mean = F.normalize(txt_pooled, dim=-1)

        # 信息论对齐损失（最大化互信息）
        similarity = torch.matmul(img_mean, txt_mean.T)  # (B, B)
        target = torch.arange(similarity.size(0)).cuda()
        align_loss = (F.cross_entropy(similarity, target) +
                      F.cross_entropy(similarity.T, target)) / 2

        return cls_loss + align_loss
```

### 2.3 训练配置（`config.yaml`）

```yaml
# scripts/phase0/p02_perception_cnn/config.yaml
model:
  type: PerceptionCNN
  hidden_size: 768
  in_channels: 3

training:
  init_from_scratch: true
  load_pretrained: false

  dataset: cifar10_mini
  num_classes: 10
  image_size: 32

  batch_size: 128
  sequence_length: 32

  optimizer: AdamW
  learning_rate: 5.0e-4  # CNN 通常需要更大 LR
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps: 500
  scheduler: cosine

  precision: bf16
  max_train_steps: 10000  # 约 1-2 hours
  log_interval: 100
  eval_interval: 1000

evaluation:
  cifar10_target_accuracy: 0.7
  alignment_target_cosine: 0.9
```

### 2.4 错误处理 SOP

**SOP-CNN-1：模型不收敛**

```
触发：CIFAR-10 准确率 < 70%
操作：
1. 检查数据 pipeline（图像是否正确归一化）
2. 检查学习率（CNN 通常需要 1e-4 到 1e-3）
3. 检查数据增强（添加 RandomCrop + Flip）
4. 启用 warmup 1000 steps
5. 失败后升级
```

**SOP-CNN-2：对齐损失不降**

```
触发：alignment_loss > 0.1（cosine sim < 0.9）
操作：
1. 检查文本 latent 是否正确归一化
2. 调整对齐损失权重
3. 使用更长的文本序列（确保 text_latent 不是 OOV）
4. 失败后升级
```

---

## 3. P.0.3 数据 Pipeline

### 3.1 DataLoader 实现（`dataloader.py`）

```python
# scripts/phase0/p03_data_pipeline/dataloader.py
"""Phase 0 共享 DataLoader"""
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional

class Phase0Dataset(Dataset):
    """多模态数据集——图像 + 文本（未来扩展）"""
    def __init__(self, tokenizer, data_subset: str, seq_length: int = 2048,
                 image_size: int = 224, include_images: bool = False):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.image_size = image_size
        self.include_images = include_images

        # 从 data_subset 加载数据
        self.examples = load_data_subset(data_subset)

        # 内存压力测试
        self._validate_memory()

    def _validate_memory(self):
        """启动时验证数据集不会导致 OOM"""
        sample = self[0]
        sample_size_gb = sum([
            sample['input_ids'].element_size() * sample['input_ids'].nelement(),
        ]) / 1024**3

        max_batch_gb = sample_size_gb * 64  # 默认 batch=64
        if max_batch_gb > 20:  # 24GB 显存，保留 4GB 给模型
            raise RuntimeError(
                f"❌ OOM 风险：单 batch {max_batch_gb:.1f}GB > 20GB。"
                f"请减小 batch_size 或启用 gradient checkpointing。"
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]['text']

        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.seq_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'text': text,  # 用于调试
        }
```

### 3.2 错误处理 SOP

**SOP-DATA-1：DataLoader 卡死**

```
触发：DataLoader 超过 30s 无数据输出
操作：
1. 检查 num_workers（推荐 4-8）
2. 检查共享内存 /dev/shm 是否足够
3. 减少 batch_size 降低压力
4. 检查是否有死锁（persistent_workers=False）
```

**SOP-DATA-2：token 错位**

```
触发：检测到相同文本生成不同 token ID
操作：
1. 检查 tokenizer 缓存（重新加载）
2. 检查数据预处理是否清理了空白
3. 升级到数据源验证
```

---

## 4. P.0.4 训练基础设施

### 4.1 BF16 训练脚本（`bf16_trainer.py`）

```python
# scripts/phase0/p04_train_infra/bf16_trainer.py
"""BF16 训练核心循环"""
import torch
import torch.amp as amp

class BF16Trainer:
    """Phase 0 标准 BF16 训练器

    关键功能：
    1. BF16 混合精度
    2. Gradient checkpointing（默认开启）
    3. NaN 检测
    4. Automatic checkpoint saving
    """
    def __init__(self, model, optimizer, scheduler, scaler=None,
                 grad_clip_norm=1.0, log_interval=50):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler  # FP16 需要，BF16 可选
        self.grad_clip_norm = grad_clip_norm
        self.log_interval = log_interval
        self.global_step = 0

    def train_step(self, batch):
        """单个训练 step"""
        self.model.train()
        input_ids = batch['input_ids'].cuda()

        # BF16 前向
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss

        # 反向
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_norm
        )

        # NaN 检测
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Loss is {loss.item()} at step {self.global_step}")
            return {"loss": None, "grad_norm": None, "status": "nan"}

        # 优化器 step
        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "lr": self.scheduler.get_last_lr()[0],
            "status": "ok",
        }
```

### 4.2 Checkpoint Save/Load（`checkpoint.py`）

```python
# scripts/phase0/p04_train_infra/checkpoint.py
"""Checkpoint 管理"""
import torch
from pathlib import Path
from typing import Optional

def save_checkpoint(model, optimizer, scheduler, step: int,
                    output_dir: str, keep_last_n: int = 3):
    """保存 checkpoint——保留最近 N 个"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_path / f"checkpoint_step_{step}.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, ckpt_path)

    # 清理旧 checkpoint
    ckpts = sorted(output_path.glob("checkpoint_step_*.pt"))
    for old_ckpt in ckpts[:-keep_last_n]:
        old_ckpt.unlink()

    print(f"✅ Saved checkpoint: {ckpt_path}")
    return ckpt_path

def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None):
    """加载 checkpoint——用于断点恢复"""
    ckpt = torch.load(checkpoint_path)

    # 验证兼容性
    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    step = ckpt.get('step', 0)
    print(f"✅ Loaded checkpoint from step {step}")
    return step
```

### 4.3 错误处理 SOP

**SOP-INFRA-1：BF16 NaN**

```
触发：训练 loss 变为 NaN/Inf
操作：
1. 启用 BF16 → FP32 切换（bf16_trainer.py 加 fallback）
2. 减小学习率（×0.5）
3. 添加 gradient clipping max_norm=1.0
4. 加载上一个 checkpoint
```

**SOP-INFRA-2：Checkpoint 损坏**

```
触发：torch.load() 抛出 RuntimeError
操作：
1. 尝试 `weights_only=False` 加载
2. 退回上一个 save_interval 的 checkpoint
3. 检查磁盘健康（fsck）
```

---

## 5. P.0.5 评估框架（最关键——决定跨路线对比有效性）

### 5.1 统一评估接口（`unified_eval.py`）

```python
# scripts/phase0/p05_eval_framework/unified_eval.py
"""Phase 0 统一评估接口——所有路线必须使用同一脚本"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class EvalResult:
    """统一的评估结果格式——所有路线必须返回这个"""
    task_name: str
    metric_name: str
    score: float
    std: Optional[float] = None
    num_samples: int = 0
    elapsed_sec: float = 0.0
    metadata: Dict = None

    def to_dict(self):
        return asdict(self)


class UnifiedEvaluator:
    """统一评估器——禁止私自评估"""

    def __init__(self, model_path: str, model_type: str,
                 eval_config_path: str = "scripts/phase0/p05_eval_framework/config.yaml"):
        self.model = self._load_model(model_path, model_type)
        self.config = self._load_config(eval_config_path)

    def evaluate_all(self) -> Dict[str, EvalResult]:
        """评估所有任务——返回统一格式"""
        results = {}

        # 1. C4 PPL
        results['c4_ppl'] = self.eval_c4_ppl()

        # 2. GSM8K mini
        results['gsm8k'] = self.eval_gsm8k_mini()

        # 3. MultiPL-E mini（代码生成）
        results['multipl_e'] = self.eval_multipl_e_mini()

        # 4. Multi-path accuracy（决策）
        results['multi_path_acc'] = self.eval_multi_path()

        # 5. 单 token 延迟（端侧）
        results['latency_per_token_ms'] = self.eval_latency()

        # 6. 显存峰值
        results['memory_peak_mb'] = self.eval_memory_peak()

        return results

    def eval_c4_ppl(self) -> EvalResult:
        """C4 PPL 评估"""
        # 统一实现，禁止路线私自修改
        ...

    def eval_gsm8k_mini(self) -> EvalResult:
        """GSM8K mini（200 样本）评估"""
        ...

    def eval_latency(self) -> EvalResult:
        """单 token 延迟 profile"""
        ...

    def save_results(self, output_path: str):
        """保存结果——统一格式"""
        results = self.evaluate_all()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
        print(f"✅ Saved eval results to {output_path}")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", required=True,
                       choices=["minimind3_base", "logos_64m", "sadko_64m"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evaluator = UnifiedEvaluator(args.model_path, args.model_type)
    results = evaluator.save_results(args.output)

    print("\n📊 评估结果：")
    for task, result in results.items():
        print(f"  {task}: {result.score:.4f}")
```

### 5.2 单 Token 延迟 Profile（`latency.py`）

```python
# scripts/phase0/p05_eval_framework/metrics/latency.py
"""单 token 延迟 + 显存峰值 测量"""
import torch
import time
import statistics

class LatencyProfiler:
    """延迟 + 显存评估——决定端侧可行性"""

    def __init__(self, model):
        self.model = model.cuda().eval()

    def measure(self, seq_length=2048, batch_size=1, num_warmup=10, num_trials=100):
        """测量单 token 延迟分布"""
        # Warmup
        input_ids = torch.randint(0, 6400, (batch_size, seq_length)).cuda()
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(input_ids)

        torch.cuda.synchronize()

        # 测量延迟
        latencies = []
        memory_peaks = []

        for trial in range(num_trials):
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(input_ids)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000

            peak_mb = torch.cuda.max_memory_allocated() / 1024**2

            latencies.append(latency_ms)
            memory_peaks.append(peak_mb)

        return {
            'latency_avg_ms': statistics.mean(latencies),
            'latency_p50_ms': statistics.median(latencies),
            'latency_p99_ms': sorted(latencies)[int(0.99 * len(latencies))],
            'latency_std_ms': statistics.stdev(latencies),
            'memory_avg_mb': statistics.mean(memory_peaks),
            'memory_peak_mb': max(memory_peaks),
            'seq_length': seq_length,
            'batch_size': batch_size,
        }
```

### 5.3 配置（`config.yaml`）

```yaml
# scripts/phase0/p05_eval_framework/config.yaml
evaluation:
  unified_dataset_path: data/phase0_eval/
  cache_eval_results: true

  tasks:
    c4_ppl:
      dataset: c4_validation
      num_samples: 10000
      sequence_length: 2048
      batch_size: 16
      pass_threshold: 3.5

    gsm8k_mini:
      dataset: gsm8k_test_mini  # 200 样本子集
      pass_threshold: 0.5  # 准确率

    multipl_e_mini:
      dataset: multipl_e_mini
      pass_threshold: 0.4

    multi_path:
      dataset: nuscenes_mini_decisions
      pass_threshold: 0.6

    latency:
      sequence_length: 2048
      num_trials: 100
      pass_threshold_ms: 100  # K=2 单 token < 100ms

    memory_peak:
      pass_threshold_mb: 20480  # < 20GB
```

### 5.4 错误处理 SOP

**SOP-EVAL-1：评估结果不一致**

```
触发：同路线两次评估结果差异 > 5%
操作：
1. 固定 random seed
2. 检查硬件温度（GPU throttle）
3. 重新跑三次取平均
4. 失败后升级到协调会议
```

**SOP-EVAL-2：跨路线评估结果不可比**

```
触发：发现某路线用了非统一脚本
操作：
1. 立即停止该路线的"评估结论"
2. 用 unified_eval.py 重跑该路线
3. 更新报告——之前的"评估"标记为 ❌ 不可信
```

---

## 6. P.0.6 端侧并行化基础设施（Logos 专用）

### 6.1 Radix Cache 实现（`radix_cache.py`）

```python
# scripts/phase0/p06_edge_infra/radix_cache.py
"""Radix Tree 共享前缀缓存——Phase 0 Logos 基础设施"""
from typing import Dict, List, Optional, Tuple
import torch


class RadixTreeNode:
    """Radix Tree 节点——存储(hidden_state, prefix_hash) → 下一层"""
    def __init__(self):
        self.hidden_state: Optional[torch.Tensor] = None
        self.children: Dict[int, 'RadixTreeNode'] = {}
        self.ref_count: int = 0


class RadixCache:
    """Radix Cache——多路径推理共享前缀

    使用场景：Radix Cache 多路径并行（B.4 依赖此基础设施）
    """
    def __init__(self, base_hidden: torch.Tensor):
        """初始化——共享前缀"""
        self.root = RadixTreeNode()
        self.root.hidden_state = base_hidden.detach().clone()
        self.root.ref_count += 1
        self.hit_count = 0
        self.miss_count = 0

    def get(self, prefix_hash: Tuple) -> Optional[torch.Tensor]:
        """检索缓存的前缀 hidden state"""
        node = self.root
        for h in prefix_hash:
            if h not in node.children:
                self.miss_count += 1
                return None
            node = node.children[h]
        self.hit_count += 1
        return node.hidden_state

    def put(self, prefix_hash: Tuple, hidden_state: torch.Tensor):
        """存储前缀 hidden state"""
        node = self.root
        for h in prefix_hash:
            if h not in node.children:
                node.children[h] = RadixTreeNode()
            node = node.children[h]
        node.hidden_state = hidden_state.detach().clone()

    def stats(self) -> dict:
        """获取缓存统计"""
        total = self.hit_count + self.miss_count
        return {
            'hit_rate': self.hit_count / total if total > 0 else 0,
            'total_lookups': total,
            'hits': self.hit_count,
            'misses': self.miss_count,
        }
```

### 6.2 Per-Token Early Exit（`per_token_exit.py`）

```python
# scripts/phase0/p06_edge_infra/per_token_exit.py
"""Per-Token 早退机制——Phase 0 Logos 基础设施"""
import torch

class PerTokenEarlyExit:
    """每个 token 独立判断是否提前退出

    算法（来自 per-token-convergence.md）：
    - 90% token 在 6 步内收敛
    - 10% token 需要 8 步
    """
    def __init__(self, max_k: int = 8, epsilon: float = 1e-4):
        self.max_k = max_k
        self.epsilon = epsilon

    def should_exit(self, hidden_state: torch.Tensor, prev_hidden: torch.Tensor) -> bool:
        """判断 token 是否已收敛"""
        delta = (hidden_state - prev_hidden).norm(dim=-1)  # [B, L]
        return (delta < self.epsilon).all().item()

    def forward(self, block_fn, x: torch.Tensor, init_hidden: torch.Tensor):
        """带早退的前向传播"""
        h = init_hidden.clone()
        exit_step = torch.zeros(x.size(0), x.size(1), dtype=torch.long)  # [B, L]

        for k in range(self.max_k):
            h_new = block_fn(h, x)
            exit_mask = self.should_exit(h_new, h)
            exit_step += (~exit_mask).long()  # 累积步数

            if exit_mask:
                h = h_new
                break
            h = h_new

        return h, exit_step
```

### 6.3 Hierarchical Reasoning（`hierarchical.py`）

```python
# scripts/phase0/p06_edge_infra/hierarchical.py
"""层次化推理——主 + 子并行"""
import torch

class HierarchicalReasoner:
    """主推理在关键节点暂停，派生子推理完成局部任务

    使用场景：可分解推理（数学先计算再验证）
    """
    def __init__(self, main_block, sub_block, K_main: int = 2, K_sub: int = 1):
        self.main_block = main_block
        self.sub_block = sub_block
        self.K_main = K_main
        self.K_sub = K_sub

    def forward(self, x: torch.Tensor, init_h: torch.Tensor):
        h = init_h.clone()

        # 主推理第一阶段
        for k in range(self.K_main):
            h = self.main_block(h, x)

        # 子推理并行
        sub_results = []
        for l in range(self.K_sub):
            sub_h = self.sub_block(h, x)
            sub_results.append(sub_h)

        # 主推理第二阶段（融合子结果）
        sub_context = torch.stack(sub_results).mean(dim=0)
        h = self.main_block(h, x, sub_context=sub_context)

        return h
```

### 6.4 验收脚本

```python
# scripts/phase0/p06_edge_infra/test_p06.py
"""P.0.6 验收——所有端侧基础设施必须通过"""
import torch
from radix_cache import RadixCache
from per_token_exit import PerTokenEarlyExit
from hierarchical import HierarchicalReasoner

def test_radix_cache():
    """Radix Cache 必须能跑通"""
    base = torch.randn(2, 16, 768)
    cache = RadixCache(base)

    # 存储前缀
    cache.put((1, 2, 3), torch.randn(2, 16, 768))
    cache.put((1, 2, 4), torch.randn(2, 16, 768))

    # 检索
    h = cache.get((1, 2, 3))
    assert h is not None

    # 命中率测试
    for _ in range(100):
        cache.get((1, 2, 3))

    stats = cache.stats()
    assert stats['hit_rate'] > 0.5, f"❌ 命中率 {stats['hit_rate']} < 50%"

    print(f"✅ Radix Cache 命中率：{stats['hit_rate']:.1%}")

def test_per_token_exit():
    """Per-Token 早退必须能跑通"""
    def block_fn(h, x):
        # 模拟快速收敛
        return h * 0.5 + x[:, :h.size(1)] * 0.5

    exit_module = PerTokenEarlyExit(max_k=8, epsilon=1e-4)
    x = torch.randn(2, 16, 768)
    init_h = torch.randn(2, 16, 768)
    h, exit_step = exit_module.forward(block_fn, x, init_h)

    avg_exit = exit_step.float().mean().item()
    assert avg_exit <= 3.5, f"❌ 平均退出步数 {avg_exit} > 3.5"

    print(f"✅ Per-Token 早退平均步数：{avg_exit:.1f}")

def test_hierarchical():
    """Hierarchical 推理必须能跑通"""
    main_block = lambda h, x, **kw: h + x[:, :h.size(1)] * 0.1
    sub_block = lambda h, x: h + x[:, :h.size(1)] * 0.05

    reasoner = HierarchicalReasoner(main_block, sub_block, K_main=2, K_sub=1)
    x = torch.randn(2, 16, 768)
    init_h = torch.randn(2, 16, 768)
    h = reasoner.forward(x, init_h)

    print(f"✅ Hierarchical 推理完成，输出 shape：{h.shape}")

if __name__ == "__main__":
    print("=== P.0.6 端侧并行化基础设施验收 ===")
    test_radix_cache()
    test_per_token_exit()
    test_hierarchical()
    print("\n✅ P.0.6 全部通过")
```

### 6.5 错误处理 SOP

**SOP-EDGE-1：Radix Cache 命中率低**

```
触发：命中率 < 50%
操作：
1. 检查 prefix_hash 生成策略——是否使用确定性 hash
2. 检查 shared prefix 长度（应该足够长）
3. 增加缓存容量
```

**SOP-EDGE-2：早退不收敛**

```
触发：平均退出步数 > 3.5
操作：
1. 调整 epsilon（×10 或 ÷10）
2. 检查 block_fn 是否数值稳定
3. 增加 warmup 步骤
```

---

## 7. Phase 0 验收报告模板

### 7.1 报告结构

```markdown
# Phase 0 验收报告

| 属性 | 值 |
|------|------|
| 报告日期 | YYYY-MM-DD |
| 执行人 | <name> |
| 硬件 | RTX 3090 / 4090 |
| 总耗时 | <hours> |

## 6 项 P.0.x 验收结果

### P.0.1 MiniMind3 64M 从零训练基线

| 指标 | 实际值 | 合格线 | 优秀线 | 判定 |
|------|--------|--------|--------|------|
| C4 PPL | X.XX | < 3.5 | < 3.0 | ✅/❌ |
| 训练时长 | X.X h | < 4h | < 3h | ✅/❌ |
| 显存峰值 | X.X GB | < 20GB | < 16GB | ✅/❌ |
| 无 NaN/崩溃 | TRUE/FALSE | TRUE | TRUE | ✅/❌ |

**P.0.1 结论**：✅ PASS / ⚠️ PASS_WITH_CAVEATS / ❌ FAIL

### P.0.2 3 层 CNN 感知层

| 指标 | 实际值 | 合格线 | 优秀线 | 判定 |
|------|--------|--------|--------|------|
| CIFAR-10 准确率 | X.XX | > 70% | > 80% | ✅/❌ |
| 对齐 cosine sim | X.XX | > 0.9 | > 0.95 | ✅/❌ |

**P.0.2 结论**：✅ / ⚠️ / ❌

### P.0.3 - P.0.6（类似）

## 综合结论

- [ ] ✅ **全部通过**——可启动 Phase 1（v1.0/v2.0/v3.0）
- [ ] ⚠️ **部分通过**——列出未通过项和修复计划
- [ ] ❌ **重大失败**——列出失败项 + 建议（重新设计？延期？）

## 附件

- 训练日志：logs/phase0/p0X/*.log
- 评估结果：reports/phase0/p0X_*.json
- W&B 链接：https://wandb.ai/latentmind-phase0
- Checkpoint 路径：checkpoints/phase0/p0X/
```

### 7.2 报告生成脚本

```python
# scripts/phase0/utils/generate_p0_report.py
"""Phase 0 验收报告自动生成"""
import json
from pathlib import Path
from datetime import datetime

def generate_report(task_id: str, log_dir: str, output_path: str):
    """生成 P.0.x 验收报告"""
    # 读取各种输出
    train_log = (Path(log_dir) / "train.log").read_text() if (Path(log_dir) / "train.log").exists() else ""
    eval_json = json.loads((Path(log_dir) / "c4_ppl.json").read_text()) if (Path(log_dir) / "c4_ppl.json").exists() else {}

    # 提取关键指标
    # ... (实际提取逻辑)

    # 生成 markdown 报告
    report = generate_markdown_report(task_id, train_log, eval_json)
    Path(output_path).write_text(report)
```

---

## 8. 紧急升级与协调 SOP

### 8.1 失败回退决策树

```
Phase 0 P.0.X 失败
├─ 单项 P.0.x 失败
│  ├─ SOP 修复尝试 1 → 成功 → 继续
│  ├─ SOP 修复尝试 2 → 成功 → 继续
│  ├─ SOP 修复尝试 3 → 失败 → ESCALATE
│  └─ ESCALATE：进入 8.2
│
└─ 多项 P.0.x 同时失败
   └─ ESCALATE：进入 8.2
```

### 8.2 升级路径（Escalation Path）

| 级别 | 升级时机 | 联系人 | 决策权 |
|------|----------|--------|--------|
| L1 | 单次 SOP 修复失败 | 路线负责人 | 继续尝试 / 调用 L2 |
| L2 | SOP 修复 3 次失败 | 协调员 + 双路线负责人 | 重试 / 调整范围 / 调用 L3 |
| L3 | 双路线重大失败 | 项目负责人 + 主要贡献者 | 重新设计 / 延期 1 月 / 调用 L4 |
| L4 | 涉及路线重新设计 | 全体参与者 | 决定新方向 / 暂停项目 |

### 8.3 通信渠道

| 类型 | 渠道 | 频率 |
|------|------|------|
| 日常状态 | Slack #latentmind-dev | 实时 |
| 周会 | Zoom 周会 | 周一 10:00 |
| 紧急事故 | Slack #latentmind-emergency | 立即 |
| Phase 0 进度 | GitHub Project board | 每日更新 |

### 8.4 Phase 0 完成检查清单

最后一周由协调员执行：

- [ ] P.0.1 验收报告发布
- [ ] P.0.2 - P.0.6 全部验收通过
- [ ] Phase 0 综合报告发布
- [ ] Phase 1A（Logos）启动准备就绪
- [ ] Phase 1B（SADKO）启动准备就绪
- [ ] W&B 项目整理归档

---

## 9. 相关文档

| 文档 | 关系 |
|------|------|
| [logos-sadko-64m-coordination.md §2](../research/logos-sadko-64m-coordination.md) | Phase 0 共享前置的"为什么" |
| **本文档** | Phase 0 的"怎么做" |
| [logos-64m-validation-plan.md §0+](../research/logos-64m-validation-plan.md) | Logos 64M 对 Phase 0 的依赖 |
| [sadko-64m-validation-plan.md §〇+](../research/sadko-64m-validation-plan.md) | SADKO 64M 对 Phase 0 的依赖 |
| [phase-0-recovery-sop.md](./phase-0-recovery-sop.md) | **配套：失败回退决策树详细 SOP** |

---

**最后更新**：2026-07-29
**作者**：来自工作流（Phase 0 实施指南）
**版本**：v1.0