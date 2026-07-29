# Logos-Native-64M v1.0 架构设计：双时间尺度对比

> **一句话定位**：在 MiniMind3 64M 上对比四种"双时间尺度"实现——HRM H/L、SADKO Split-GQA、混合风格、标准 Transformer 基线，找出 Logos 主线的最优基座架构。
> **上游文档**：[logos-64m-validation-plan.md §2](./logos-64m-validation-plan.md#2-v10双时间尺度对比基座适配)
> **下游文档**：[logos-v2-architecture.md](./logos-v2-architecture.md)（基于 v1.0 推荐的基座）
> **最后更新**：2026-07-29

---

## 1. 设计目标

**找到 Logos 64M v1.0 的最优基座架构**——四种"双时间尺度"实现方案中，A.3 混合风格应最优。

**验收标准**：
- A.3 PPL 优于 A.4 基线 ≥ 5%（合格）/ ≥ 10%（优秀）
- A.3 PPL 优于 A.1 和 A.2（**核心**）
- 训练稳定（无 NaN、无 mode collapse）

---

## 2. MiniMind3 64M 基座规格

```yaml
# minimind3_64m_base.yaml
model_type: minimind
vocab_size: 6400
hidden_size: 768
num_hidden_layers: 8
num_attention_heads: 16
num_key_value_heads: 8
head_dim: 48
intermediate_size: 2048
hidden_act: silu
rms_norm_eps: 1.0e-5
rope_theta: 10000.0
max_position_embeddings: 4096
tie_word_embeddings: true
```

**关键约束**：
- 8 层 → 必须二分为 H block + L block（如 4H + 4L）
- 8 KV heads → Split-GQA 最大 4:4 分裂
- head_dim=48 → 异构 RoPE 需适配

---

## 3. 四种"双时间尺度"实现方案

### 3.1 A.1 HRM 风格（独立 H/L block）

**核心思想**：把 8 层分为 4 H block + 4 L block（交替）。

```text
SADKO-HRM-64M v1.0 A.1（HRM 风格）：
├── Tokenizer: vocab=6400
├── Embedding: dim=768
├── Transformer Blocks × 8（交替 H/L）：
│   ├── Layer 0: H_block (4 heads)
│   ├── Layer 1: L_block (4 heads)
│   ├── Layer 2: H_block
│   ├── Layer 3: L_block
│   ├── Layer 4: H_block
│   ├── Layer 5: L_block
│   ├── Layer 6: H_block
│   └── Layer 7: L_block
├── 循环结构：
│   for h in range(2):     # H cycles
│       for l in range(2): # L cycles（每 H 周期）
│           z_L = L_block(z_L + z_H)
│       z_H = H_block(z_H + z_L)
└── 总循环步 = 2H × 2L = 4 + 2 H updates = 6 步
```

**优势**：块级时间尺度分离最清晰（不同 block 独立优化）
**劣势**：增加深度，推理延迟较高（但 L block 可并行化）

### 3.2 A.2 Split-GQA 风格（SADKO 借鉴）

**核心思想**：保持 8 层标准 block，但每个 block 内部 KV heads 分裂。

```text
SADKO-HRM-64M v1.0 A.2（Split-GQA 风格）：
├── Tokenizer: vocab=6400
├── Embedding: dim=768
├── Transformer Blocks × 8：
│   └── 每个 Block 内部：
│       ├── SplitGQAAttention:
│       │   ├── Static Heads: 4 (base=500_000, 长程)
│       │   └── Dynamic Heads: 4 (base=10_000, 短程)
│       └── SwiGLU FFN
├── 循环结构：
│   for k in range(K):
│       z = block(z)  # 标准单 block 循环
└── K 次循环（与 A.1 不同的循环结构）
```

**详细实现**（借鉴 SADKO [sadko-v1-architecture.md §5](./sadko-v1-architecture.md)）：

```python
class SplitGQAAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        # 标准 Q 投影（16 heads）
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        
        # Static + Dynamic KV 投影（4 + 4 = 8 KV heads）
        self.k_static_proj = nn.Linear(hidden_size, 4 * head_dim)
        self.v_static_proj = nn.Linear(hidden_size, 4 * head_dim)
        self.k_dynamic_proj = nn.Linear(hidden_size, 4 * head_dim)
        self.v_dynamic_proj = nn.Linear(hidden_size, 4 * head_dim)
        
        # O 投影
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
        # 异构 RoPE
        self.rope_static = RoPE(head_dim, base=500_000)
        self.rope_dynamic = RoPE(head_dim, base=10_000)
    
    def forward(self, x):
        q = self.q_proj(x).view(B, L, 16, 48)  # 16 query heads
        
        # Static KV（长程，base=500k）
        k_s = self.k_static_proj(x).view(B, L, 4, 48)
        v_s = self.v_static_proj(x).view(B, L, 4, 48)
        k_s = self.rope_static(k_s)
        
        # Dynamic KV（短程，base=10k）
        k_d = self.k_dynamic_proj(x).view(B, L, 4, 48)
        v_d = self.v_dynamic_proj(x).view(B, L, 4, 48)
        k_d = self.rope_dynamic(k_d)
        
        # 拼接后注意力
        k = torch.cat([k_s, k_d], dim=2)  # [B, L, 8, 48]
        v = torch.cat([v_s, v_d], dim=2)
        
        attn_out = attention(q, k, v)
        return self.o_proj(attn_out)
```

**优势**：单 block 内部时间尺度分离，参数量小
**劣势**：单 block 容量受限（与 HRM H/L block 容量相比）

### 3.3 A.3 混合风格（应最优）

**核心思想**：H/L block 独立 + 每个 block 内部 Split-GQA。

```text
SADKO-HRM-64M v1.0 A.3（混合风格）：
├── Tokenizer: vocab=6400
├── Embedding: dim=768
├── Transformer Blocks × 8（交替 H/L）：
│   ├── H_block（每 2 层出现 1 次，共 4 个）：
│   │   ├── SplitGQAAttention:
│   │   │   ├── Static Heads: 3（更多静态，base=500k）
│   │   │   └── Dynamic Heads: 1（少量动态，base=10k）
│   │   └── SwiGLU FFN
│   └── L_block（每 2 层出现 1 次，共 4 个）：
│       ├── SplitGQAAttention:
│       │   ├── Static Heads: 1（少量静态，base=500k）
│       │   └── Dynamic Heads: 3（更多动态，base=10k）
│       └── SwiGLU FFN
├── 循环结构（同 A.1）：2H × 2L = 6 步
```

**核心思想**：
- H block **更多 Static heads**（长程主导，适合全局逻辑）
- L block **更多 Dynamic heads**（短程主导，适合局部细节）
- **双层时间尺度**：block 维度 + head 维度

**优势**：双重时间尺度，表达力最强
**劣势**：实现复杂，需精细调优 static/dynamic head 比例

### 3.4 A.4 基线（标准 Transformer）

**核心思想**：保持 MiniMind3 64M 原架构，仅做循环。

```text
SADKO-HRM-64M v1.0 A.4（基线）：
├── 标准 Transformer Block × 8
└── for k in range(K): z = block(z)
```

**目的**：作为对照，验证"双时间尺度"是否真有价值。

---

## 4. 配置定义

### 4.1 共享配置（所有方案）

```yaml
# logos_v1_config.yaml
training:
  lr: 1e-4
  warmup_steps: 2000
  optimizer: AdamW
  batch_size: 64
  sequence_length: 2048
  total_tokens: 4B  # 与 MiniMind3 原始 Pretrain 对齐

quantization:
  perception_layer: INT8
  hrm_backbone: FP16  # 不用 Q4，破坏 MagicNorm
  decoder: INT8
  kv_cache: INT8（端侧）

constraints:
  hardware: 单卡 RTX 3090 24GB
  memory_budget: < 8GB（含 KV cache）
  latency_target: K=2 < 100ms
```

### 4.2 各方案特定配置

**A.1 HRM 风格**：
- num_h_blocks: 4
- num_l_blocks: 4
- H_cycles: 2
- L_cycles: 2
- 总循环步: 6

**A.2 Split-GQA 风格**：
- num_kv_heads_static: 4
- num_kv_heads_dynamic: 4
- rope_base_static: 500_000
- rope_base_dynamic: 10_000
- K: 6（与 A.1 循环步对齐）

**A.3 混合风格**：
- H_block static_heads: 3
- H_block dynamic_heads: 1
- L_block static_heads: 1
- L_block dynamic_heads: 3
- H_cycles × L_cycles: 2 × 2 = 6 步

**A.4 基线**：
- 标准 Transformer
- K: 6

---

## 5. 权重初始化策略

### 5.1 从 MiniMind3 预训练权重初始化

**A.1 HRM 风格**：
- 从 MiniMind3 加载 8 层权重
- 前 4 层 → H_block
- 后 4 层 → L_block
- H_block 初始 bias 略大（偏向全局）

**A.2 Split-GQA 风格**：
- 从 MiniMind3 加载标准 K/V 投影（8 heads）
- **劈裂**为 Static 4 + Dynamic 4（按 head index 0-3 为 Static，4-7 为 Dynamic）
- K_static ← K[:, :4, :], K_dynamic ← K[:, 4:, :]

**A.3 混合风格**：
- 先按 A.2 加载（block 内劈裂）
- 再按 A.1 重组（H/L block 角色分工）

**A.4 基线**：
- 直接加载 MiniMind3 权重

### 5.2 门控初始化

```python
# H block 初始化：偏置 = +0.5（偏向全局）
nn.init.constant_(h_block.gate.bias, 0.5)

# L block 初始化：偏置 = -0.5（偏向局部）
nn.init.constant_(l_block.gate.bias, -0.5)
```

---

## 6. 关键代码实现

### 6.1 H_block / L_block 定义

```python
class LogosHBlock(nn.Module):
    """H module: global, slow, memory-oriented"""
    def __init__(self, hidden_size, num_heads, num_kv_static, num_kv_dynamic):
        super().__init__()
        self.ln_1 = RMSNorm(hidden_size)
        self.attn = SplitGQAAttention(
            hidden_size, num_heads,
            num_kv_static, num_kv_dynamic,
            rope_base_static=500_000,
            rope_base_dynamic=10_000,
        )
        self.ln_2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate=2048)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class LogosLBlock(nn.Module):
    """L module: local, fast, compute-oriented"""
    def __init__(self, hidden_size, num_heads, num_kv_static, num_kv_dynamic):
        super().__init__()
        self.ln_1 = RMSNorm(hidden_size)
        self.attn = SplitGQAAttention(
            hidden_size, num_heads,
            num_kv_static, num_kv_dynamic,
            rope_base_static=500_000,
            rope_base_dynamic=10_000,
        )
        self.ln_2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate=2048)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

### 6.2 混合风格 (A.3) 完整模型

```python
class LogosMixedModel(nn.Module):
    """混合风格: H/L blocks + Split-GQA"""
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 4 H blocks（更多 static heads）
        self.h_blocks = nn.ModuleList([
            LogosHBlock(
                config.hidden_size,
                num_heads=16,
                num_kv_static=3,  # H 偏静态
                num_kv_dynamic=1,
            ) for _ in range(4)
        ])
        
        # 4 L blocks（更多 dynamic heads）
        self.l_blocks = nn.ModuleList([
            LogosLBlock(
                config.hidden_size,
                num_heads=16,
                num_kv_static=1,  # L 偏动态
                num_kv_dynamic=3,
            ) for _ in range(4)
        ])
        
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids, h_cycles=2, l_cycles=2):
        z = self.embed(input_ids)
        z_H = z  # H 状态从输入初始化
        z_L = z  # L 状态从输入初始化
        
        for h in range(h_cycles):
            for l in range(l_cycles):
                # L block 在 H 状态下更新
                # 交替使用不同的 L block
                z_L = self.l_blocks[l](z_L + z_H)
            # H block 在 L 状态更新后更新
            z_H = self.h_blocks[h](z_H + z_L)
        
        return self.lm_head(self.norm(z_H))
```

### 6.3 与 SADKO v1-architecture 的对应关系

| SADKO 实现 | Logos 借鉴方式 |
|-----------|------------|
| [sadko-v1-architecture.md §5 SplitGQAAttention](./sadko-v1-architecture.md) | 直接借鉴，作为 A.2/A.3 的基础组件 |
| [sadko-v1-architecture.md §5 DualPathFFN](./sadko-v1-architecture.md) | H/L block 内的 FFN 可借鉴 |
| [sadko-v1-architecture.md §6 权重初始化](./sadko-v1-architecture.md) | K/V 投影劈裂初始化策略 |

---

## 7. 实验设计

### 7.1 训练配置

```python
# 共享训练配置
training_config = {
    'data': 'minimind3_pretrain_4b',
    'lr': 1e-4,
    'warmup': 2000,
    'batch_size': 64,
    'seq_length': 2048,
    'total_tokens': 4_000_000_000,
    'eval_interval': 10_000,
    'log_interval': 100,
}
```

### 7.2 评估指标

| 指标 | 目的 |
|------|------|
| **PPL**（C4 验证集）| 语言建模质量 |
| **GSM8K mini**（200 样本）| 推理能力 |
| **有效秩**（effective rank）| 隐藏状态多样性 |
| **门控值分布** | H/L block 分化是否自然 |
| **MAGIC NORM 检查** | 梯度稳定性 |

### 7.3 必做消融

按优先级：

1. **基础 PPL 对比**：A.1 / A.2 / A.3 / A.4 在相同 4B tokens 训练后的 PPL
2. **GSM8K mini 对比**：四方案在推理任务上的精度
3. **门控分化监控**：A.1 和 A.3 的 H/L block 是否自然分化
4. **有效秩演化**：每方案训练过程中有效秩变化
5. **消融：移除 H block**（A.3 → A.4）：验证 H block 贡献
6. **消融：移除 Static heads**（A.3 → A.1）：验证 Split-GQA 贡献

---

## 8. 验收标准

### 8.1 PPL 验收

| 验收项 | 合格线 | 优秀线 |
|-------|:---:|:---:|
| A.1 HRM 风格 vs A.4 基线 | 持平或更好 | 优于基线 5% |
| A.2 Split-GQA vs A.4 基线 | 持平或更好 | 优于基线 5% |
| **A.3 混合 vs A.1** | **更好** | **优于 A.1 ≥ 3%** |
| **A.3 混合 vs A.2** | **更好** | **优于 A.2 ≥ 3%** |
| **A.3 混合 vs A.4** | **更好** | **优于基线 ≥ 10%** |

### 8.2 训练稳定性

| 验收项 | 标准 |
|-------|------|
| 无 NaN | 必须 |
| 无 mode collapse | 必须 |
| Loss 收敛 | 必须 |
| 梯度范数 < 10 | 必须 |

### 8.3 失败模式

| 失败现象 | 可能原因 | 修复 |
|---------|---------|------|
| A.3 不优于 A.1 | Split-GQA 增加噪声 | 调整 static/dynamic head 比例 |
| A.3 PPL 崩溃 | 门控初始化不当 | 调整 H/L gate bias |
| 训练不收敛 | MagicNorm 与 Split-GQA 不兼容 | 加 STARS 谱正则化 |

---

## 9. 决策传递

### 9.1 v1.0 出结果后

| v1.0 结论 | v2.0 基线 |
|----------|---------|
| A.3 混合最优 | 用 A.3 作为 v2.0 基线 |
| A.1 HRM 风格最优 | 用 A.1 作为 v2.0 基线（更简单）|
| A.2 Split-GQA 最优 | 用 A.2 作为 v2.0 基线（参数效率最高）|
| A.4 基线最优 | 双时间尺度无价值，v2.0 用 A.4 |

### 9.2 失败回退

| v1.0 失败 | 回退 |
|----------|------|
| A.3 严重崩溃 | 用 A.4 基线进入 v2.0 |
| 全部 PPL 高于基线 | 用 A.4 基线（说明双时间尺度在此规模无价值）|
| 训练不稳定 | 启用 STARS 谱正则化修复（[stars.md](../references/stars.md)）|

---

## 10. 与 v2.0 的衔接

v1.0 完成后，最优架构（A.3 期望）作为 v2.0 的基线：
- v2.0 B.1-B.5 都在 A.3 上测试不同循环策略
- 推荐组合：**A.3 + K=2 + 早退 + Radix Cache**

---

## 11. 相关文档

| 文档 | 关系 |
|------|------|
| [logos-64m-validation-plan.md §2](./logos-64m-validation-plan.md#2-v10双时间尺度对比基座适配) | 本文档的父级 |
| [sadko-v1-architecture.md](./sadko-v1-architecture.md) | SADKO Split-GQA 详细实现（本文档 A.2/A.3 借鉴）|
| [logos-whitepaper.md §2.2](./logos-whitepaper.md#22-模块-b分层递归潜空间引擎-750m) | H/L block 的高层架构 |
| [logos-k-strategy.md](./logos-k-strategy.md) | K 值策略（v2.0 详细论证）|
| [docs/references/hrm-text.md](../references/hrm-text.md) | HRM-Text 原论文笔记 |
| [AGENTS.md §7.5](../../AGENTS.md#75-决策时间表) | Logos 决策时间表 |

---

**最后更新**：2026-07-29
**作者**：来自工作流（Logos v1.0 架构设计）
**版本**：v1.0