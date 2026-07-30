# SADKO-Native-64M v1.0 架构设计：基座适配层（左脑改造）

> **一句话定位**：v1.0 施工图纸——在 MiniMind3 64M 上完成 Split-GQA + 异构 RoPE + Dual-Path FFN + Cross-Attention 骨架预埋，确保 PPL 不退化
> **上游文档**：[64m-validation-plan.md](./64m-validation-plan.md) §2（验证计划中的 v1.0 定位）
> **下游文档**：[v2-architecture.md](./v2-architecture.md)
> **最后更新**：2026-07-29

---

## 1. 设计目标

在**不破坏 MiniMind3 原始语言能力**的前提下，完成左脑结构性改造：

- 物理分离 Static/Dynamic KV Heads（Split-GQA）
- 异构 RoPE 频率分流
- 锚点层 Dual-Path Gated FFN
- Cross-Attention 骨架预埋（gate=0，不参与计算）

**验收标准**：PPL 退化 ≤ 0.1（C4 验证集 ≤ 2.75~2.80，原始 ~2.70），Static/Dynamic 注意力熵显著分离（Dynamic − Static > 0.3）。

---

## 2. MiniMind3 64M 基座规格（共同基础）

### 2.1 原始架构参数

```yaml
# minimind3_64m_base.yaml
model_type: minimind
vocab_size: 6400
hidden_size: 768
num_hidden_layers: 8
num_attention_heads: 16
num_key_value_heads: 8
head_dim: 48                    # 768 / 16 = 48 (非标准64，所有RoPE/Attn需适配)
intermediate_size: 2048         # SwiGLU FFN 中间维度
hidden_act: silu
rms_norm_eps: 1.0e-5
rope_theta: 10000.0             # 原始统一 RoPE base
max_position_embeddings: 4096
tie_word_embeddings: true
```

### 2.2 原始模块结构

```text
MiniMind3ForCausalLM:
├── model.embed_tokens: Embedding(6400, 768)
├── model.layers.0-7: MiniMind3DecoderLayer
│   ├── input_layernorm: RMSNorm(768)
│   ├── self_attn: MiniMind3Attention
│   │   ├── q_proj: Linear(768, 768)       # 16 heads × 48 dim
│   │   ├── k_proj: Linear(768, 384)       # 8 KV heads × 48 dim
│   │   ├── v_proj: Linear(768, 384)       # 8 KV heads × 48 dim
│   │   └── o_proj: Linear(768, 768)
│   ├── post_attention_layernorm: RMSNorm(768)
│   └── mlp: MiniMind3MLP (SwiGLU)
│       ├── gate_proj: Linear(768, 2048)
│       ├── up_proj: Linear(768, 2048)
│       └── down_proj: Linear(2048, 768)
├── model.norm: RMSNorm(768)
└── lm_head: Linear(768, 6400)  # 与 embed_tokens 共享权重
```

### 2.3 环境与硬件约束

```text
硬件: 单卡 RTX 3090 24GB 或 RTX 4060 8GB
框架: PyTorch 2.1+ / transformers 4.36+
精度: bf16 (3090) 或 fp16 (4060)
训练框架: 复用 MiniMind3 原始 train_pretrain.py 管线
```

---

## 3. 架构总览

```text
SADKO-Native-64M v1.0
═══════════════════════════════════════════════════════════════
Tokenizer:  BPE + ByteLevel, vocab=6400 (保留)
Embedding:  Token Embedding, dim=768 (保留)
            + Memory Embedding, dim=768 (零初始化, 暂不激活)

Layer 0:  [标准层] RMSNorm → GQA(16Q/8KV) → RMSNorm → SwiGLU
Layer 1:  [标准层] RMSNorm → GQA(16Q/8KV) → RMSNorm → SwiGLU
Layer 2:  [锚点层] RMSNorm → SplitGQA(4S+4D, 异构RoPE)
                     → CrossAttn(gate=0) → RMSNorm → DualPathFFN
Layer 3:  [标准层] GQA(16Q/8KV) + CrossAttn骨架(gate≈0) → SwiGLU
Layer 4:  [锚点层] 同 Layer 2
Layer 5:  [标准层] 同 Layer 3
Layer 6:  [锚点层] 同 Layer 2
Layer 7:  [标准层] RMSNorm → GQA(16Q/8KV) → RMSNorm → SwiGLU

LM Head:  768 → 6400 (保留, 共享权重)
═══════════════════════════════════════════════════════════════
锚点层: [2, 4, 6]    Cross-Attention 骨架层: [3, 5]
新增参数: ~15.5M (DualPathFFN × 3 ~14.2M + CA骨架 ~0.89M + 门控 ~0.45M + Memory Emb)
总参数: ~79.5M (基座 64M + 改造 15.5M)
```

### 关键设计决策

| 决策 | 选择 | 理由 |
| :--- | :--- | :--- |
| 锚点层 | L2, L4, L6（偶数层） | 与白皮书对齐；均匀分布确保记忆信号逐层渗透 |
| Cross-Attn 骨架层 | L3, L5（锚点层后一层） | v2.0 激活时读取 MemPool |
| Split-GQA 比例 | 4:4（1:1） | 64M 仅 8 KV heads，1:1 是最小可行分割 |
| 异构 RoPE | **Static=500k（低频长程），Dynamic=10k（高频短程）** | 对齐 NTK 标准实践：更大 base = 更慢旋转 = 长程平坦注意力；10k 为短程锐利旋转。命名"低频/高频"对应 base 大小方向 |
| Cross-Attention gate | 初始化为 -10（sigmoid ≈ 4.5e-5） | 确保 v1.0 前向传播与原始 MiniMind3 完全一致 |
| Dual-Path gate | 初始化为 0.5（sigmoid(0)） | 无先验偏好，让训练自然学习分流 |
| 层类型 | **全部标准 GQA**（不引入 SWA/GLA） | 控制变量，消融隔离原则 |

---

## 4. 配置定义

### 4.1 YAML 配置

```yaml
# sadko_v1_config.yaml
model_type: sadko_native
vocab_size: 6400
hidden_size: 768
num_hidden_layers: 8
num_attention_heads: 16
num_key_value_heads: 8
head_dim: 48
intermediate_size: 2048
hidden_act: silu
rms_norm_eps: 1.0e-5
max_position_embeddings: 4096
tie_word_embeddings: true

# ===== SADKO v1.0 新增配置 =====
sadko:
  # Split-GQA
  n_static_kv_heads: 4          # 静态记忆 KV heads (base=500k, 长程/低频)
  n_dynamic_kv_heads: 4         # 动态推理 KV heads (base=10k, 短程/高频)
  rope_static_base: 500000      # 静态低频 RoPE (对齐 NTK 长程外推实践)
  rope_dynamic_base: 10000      # 动态高频 RoPE (保留原 MiniMind3 短程行为)

  # Dual-Path FFN (仅锚点层启用)
  anchor_layers: [2, 4, 6]      # 偶数层 (0-indexed)
  semantic_tag_dim: 16          # 语义标签嵌入维度
  gate_hidden_size: 192         # 门控网络隐藏层 (hidden_size // 4)
  gate_init_bias: 0.0           # sigmoid(0) = 0.5, 无先验偏好

  # Cross-Attention 骨架 (v1.0 阶段 gate=0，不参与计算)
  cross_attn_layers: [3, 5]     # 锚点层后一层插入
  cross_attn_dim: 192           # 潜空间维度 (v1.0 不激活)
  cross_attn_heads: 6           # 192 / 32 = 6 heads
  cross_attn_gate_init: -10.0   # sigmoid(-10) ≈ 0，确保零影响

  # Memory Embedding (v1.0 零初始化，不激活)
  memory_embedding_dim: 768
```

### 4.2 Python dataclass 配置（等价形式）

```python
# config_v1.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class SADKOv1Config:
    # === 继承 MiniMind3 基座配置 ===
    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 16       # Q heads
    num_key_value_heads: int = 8        # KV heads (GQA)
    head_dim: int = 48                  # hidden_size / num_attention_heads
    intermediate_size: int = 2048       # SwiGLU
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0         # 原始 RoPE base

    # === SADKO v1.0 新增配置 ===
    anchor_layers: List[int] = field(default_factory=lambda: [2, 4, 6])

    # Split-GQA
    num_static_kv_heads: int = 4        # 8 KV heads 的前 4 个
    num_dynamic_kv_heads: int = 4       # 8 KV heads 的后 4 个
    rope_base_static: float = 500000.0  # Static: 低频长程（对齐 NTK 实践）
    rope_base_dynamic: float = 10000.0  # Dynamic: 高频短程（原 MiniMind3 行为）

    # Dual-Path FFN
    dual_path_intermediate_size: int = 2048  # 每条路径的中间维度
    gate_hidden_size: int = 192              # 门控网络隐藏层 (hidden_size // 4)
    gate_init_bias: float = 0.0              # sigmoid(0) = 0.5, 无先验偏好

    # Cross-Attention 骨架 (v1.0 暂不激活)
    cross_attn_heads: int = 8
    cross_attn_head_dim: int = 48
    cross_attn_gate_init: float = -10.0      # sigmoid(-10) ≈ 0, 确保零激活

    # Memory Embedding
    memory_embedding_dim: int = 768

    @property
    def is_anchor_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.anchor_layers
```

---

## 5. 核心模块实现

### 5.1 HeterogeneousRotaryEmbedding

```python
# rope.py
import torch
import torch.nn as nn
import math

class HeterogeneousRotaryEmbedding(nn.Module):
    """异构 RoPE: Static 长程（base=500k）+ Dynamic 短程（base=10k）

    设计依据：对齐 NTK-aware 长上下文外推实践——更大 base 意味着更慢的频率旋转，
    使注意力对位置差异的响应更平坦（长程），更小 base 则是锐利的局部衰减（短程）。
    "Static/Dynamic" 命名反映语义角色，不反映频率高低。
    """

    def __init__(self, dim: int, base_static: float = 500000.0,
                 base_dynamic: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base_static = base_static
        self.base_dynamic = base_dynamic

        # 预计算 Static 频率
        inv_freq_s = 1.0 / (base_static ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq_static', inv_freq_s)

        # 预计算 Dynamic 频率
        inv_freq_d = 1.0 / (base_dynamic ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq_dynamic', inv_freq_d)

    def _compute_rotary_emb(self, inv_freq, position_ids):
        """position_ids: [B, L] → freqs: [B, L, dim]"""
        freqs = torch.einsum('bl,d->bld', position_ids.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, L, dim]
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        return cos_emb, sin_emb

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rotary(self, x, cos, sin):
        """x: [B, n_heads, L, head_dim]"""
        cos = cos.unsqueeze(1)   # [B, 1, L, head_dim]
        sin = sin.unsqueeze(1)
        return x * cos + self._rotate_half(x) * sin

    def forward(self, k_static, k_dynamic, q_static, q_dynamic, position_ids):
        """
        k_static:  [B, n_static_kv, L, head_dim]
        k_dynamic: [B, n_dynamic_kv, L, head_dim]
        q_static:  [B, n_static_q, L, head_dim]
        q_dynamic: [B, n_dynamic_q, L, head_dim]
        position_ids: [B, L]
        """
        cos_s, sin_s = self._compute_rotary_emb(self.inv_freq_static, position_ids)
        cos_d, sin_d = self._compute_rotary_emb(self.inv_freq_dynamic, position_ids)

        k_s = self._apply_rotary(k_static, cos_s, sin_s)
        q_s = self._apply_rotary(q_static, cos_s, sin_s)
        k_d = self._apply_rotary(k_dynamic, cos_d, sin_d)
        q_d = self._apply_rotary(q_dynamic, cos_d, sin_d)

        return k_s, k_d, q_s, q_d
```

> **备选实现**：可直接复用 `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding` 实例化两个不同 base 的 RoPE（`LlamaRotaryEmbedding(dim=48, base=10000)` 与 `base=500000`），配合 `apply_rotary_pos_emb` 分别应用到 Static/Dynamic 组，减少自研代码量。

### 5.2 SADKOSplitGQA

```python
# split_gqa.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SADKOSplitGQA(nn.Module):
    """
    Split-GQA with Heterogeneous RoPE.

    Q heads 分组策略 (GQA ratio = 2:1):
      - Q[0:8]  → Static KV[0:4]   (每2个Q共享1个Static KV)
      - Q[8:16] → Dynamic KV[0:4]  (每2个Q共享1个Dynamic KV)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_q_heads = config.num_attention_heads       # 16
        self.n_static_kv = config.num_static_kv_heads     # 4
        self.n_dynamic_kv = config.num_dynamic_kv_heads   # 4
        self.head_dim = config.head_dim                    # 48
        self.gqa_ratio = self.n_q_heads // (self.n_static_kv + self.n_dynamic_kv)  # 2

        # 投影层：物理分割
        self.q_proj = nn.Linear(config.hidden_size, self.n_q_heads * self.head_dim, bias=False)
        self.k_static_proj = nn.Linear(config.hidden_size, self.n_static_kv * self.head_dim, bias=False)
        self.v_static_proj = nn.Linear(config.hidden_size, self.n_static_kv * self.head_dim, bias=False)
        self.k_dynamic_proj = nn.Linear(config.hidden_size, self.n_dynamic_kv * self.head_dim, bias=False)
        self.v_dynamic_proj = nn.Linear(config.hidden_size, self.n_dynamic_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_q_heads * self.head_dim, config.hidden_size, bias=False)

        # 异构 RoPE
        self.rope = HeterogeneousRotaryEmbedding(
            dim=self.head_dim,
            base_static=config.rope_base_static,
            base_dynamic=config.rope_base_dynamic
        )

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """GQA: 将 KV heads 重复以匹配 Q heads. x: [B, n_kv, L, D]"""
        if n_rep == 1:
            return x
        B, n_kv, L, D = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv, n_rep, L, D)
        return x.reshape(B, n_kv * n_rep, L, D)

    def forward(self, x, position_ids, attention_mask=None, kv_cache=None):
        B, L, _ = x.shape

        # 1. 投影
        q = self.q_proj(x).view(B, L, self.n_q_heads, self.head_dim).transpose(1, 2)
        k_s = self.k_static_proj(x).view(B, L, self.n_static_kv, self.head_dim).transpose(1, 2)
        v_s = self.v_static_proj(x).view(B, L, self.n_static_kv, self.head_dim).transpose(1, 2)
        k_d = self.k_dynamic_proj(x).view(B, L, self.n_dynamic_kv, self.head_dim).transpose(1, 2)
        v_d = self.v_dynamic_proj(x).view(B, L, self.n_dynamic_kv, self.head_dim).transpose(1, 2)

        # 2. Q 分组
        q_s = q[:, :self.n_q_heads // 2, :, :]    # [B, 8, L, 48]
        q_d = q[:, self.n_q_heads // 2:, :, :]     # [B, 8, L, 48]

        # 3. 异构 RoPE
        k_s, k_d, q_s, q_d = self.rope(k_s, k_d, q_s, q_d, position_ids)

        # 4. KV Cache 更新
        if kv_cache is not None:
            k_s = torch.cat([kv_cache['k_static'], k_s], dim=2)
            v_s = torch.cat([kv_cache['v_static'], v_s], dim=2)
            k_d = torch.cat([kv_cache['k_dynamic'], k_d], dim=2)
            v_d = torch.cat([kv_cache['v_dynamic'], v_d], dim=2)

        new_kv_cache = {
            'k_static': k_s, 'v_static': v_s,
            'k_dynamic': k_d, 'v_dynamic': v_d
        }

        # 5. GQA 重复 KV
        k_s = self._repeat_kv(k_s, self.gqa_ratio)  # [B, 8, L, 48]
        v_s = self._repeat_kv(v_s, self.gqa_ratio)
        k_d = self._repeat_kv(k_d, self.gqa_ratio)
        v_d = self._repeat_kv(v_d, self.gqa_ratio)

        # 6. 分别计算 Attention
        scale = 1.0 / math.sqrt(self.head_dim)

        attn_s = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale
        attn_d = torch.matmul(q_d, k_d.transpose(-2, -1)) * scale

        # Causal Mask (标准单向)
        if L > 1:
            causal_mask = torch.triu(
                torch.ones(L, k_s.size(2), device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_s = attn_s.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_d = attn_d.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attention_mask is not None:
            attn_s = attn_s + attention_mask
            attn_d = attn_d + attention_mask

        attn_s = F.softmax(attn_s, dim=-1)
        attn_d = F.softmax(attn_d, dim=-1)

        out_s = torch.matmul(attn_s, v_s)  # [B, 8, L, 48]
        out_d = torch.matmul(attn_d, v_d)  # [B, 8, L, 48]

        # 7. 合并输出
        out = torch.cat([out_s, out_d], dim=1)  # [B, 16, L, 48]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.o_proj(out)

        return out, new_kv_cache, {'attn_s': attn_s, 'attn_d': attn_d}
```

### 5.3 SADKODualPathFFN

```python
# dual_path_ffn.py
class SwiGLUFFN(nn.Module):
    """标准 SwiGLU FFN (复用 MiniMind3 实现)"""
    def __init__(self, hidden_size=768, intermediate_size=2048):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SADKODualPathFFN(nn.Module):
    """
    双路径门控 FFN: Memory Path + Reason Path.
    门控输入: hidden_state (768) + semantic_tag (16) = 784
    gate 初始化为 0.5 混合 (sigmoid(0) = 0.5).
    g → 1: 偏记忆; g → 0: 偏推理
    """
    def __init__(self, config):
        super().__init__()
        self.memory_ffn = SwiGLUFFN(config.hidden_size, config.intermediate_size)
        self.reason_ffn = SwiGLUFFN(config.hidden_size, config.intermediate_size)

        gate_input_dim = config.hidden_size + config.sadko.semantic_tag_dim  # 784
        self.gate_proj = nn.Sequential(
            nn.Linear(gate_input_dim, config.hidden_size // 4),  # 784 → 192
            nn.SiLU(),
            nn.Linear(config.hidden_size // 4, 1),               # 192 → 1
            nn.Sigmoid()
        )
        # ★ 初始化: gate 输出 0.5 (无先验偏好)
        nn.init.constant_(self.gate_proj[-2].bias, config.sadko.gate_init_bias)

    def forward(self, x, semantic_tag_emb=None):
        mem_out = self.memory_ffn(x)
        rea_out = self.reason_ffn(x)

        if semantic_tag_emb is not None:
            gate_input = torch.cat([x, semantic_tag_emb], dim=-1)  # [B, L, 784]
            g = self.gate_proj(gate_input)                          # [B, L, 1]
            out = g * mem_out + (1 - g) * rea_out
        else:
            # 无 tag 时默认 0.5 混合
            g = torch.full((*x.shape[:-1], 1), 0.5, device=x.device)
            out = 0.5 * mem_out + 0.5 * rea_out

        return out, g  # 返回 gate 值用于监控
```

### 5.4 SADKOCrossAttention（骨架，v1.0 零激活）

```python
# cross_attn_skeleton.py
class SADKOCrossAttention(nn.Module):
    """
    Cross-Attention 骨架 (v1.0 阶段 gate≈0，不参与计算)
    v2.0 时接入 MemPool，v3.0 时接入 ELF 压缩记忆
    """
    def __init__(self, config):
        super().__init__()
        self.q_dim = config.hidden_size               # 768
        self.kv_dim = config.sadko.cross_attn_dim     # 192
        self.num_heads = config.sadko.cross_attn_heads  # 6
        self.head_dim = self.kv_dim // self.num_heads   # 32

        self.q_proj = nn.Linear(self.q_dim, self.kv_dim, bias=False)
        self.k_proj = nn.Linear(self.kv_dim, self.kv_dim, bias=False)  # 占位
        self.v_proj = nn.Linear(self.kv_dim, self.kv_dim, bias=False)  # 占位
        self.o_proj = nn.Linear(self.kv_dim, self.q_dim, bias=False)

        # ★ 门控: 初始化为 -10.0，sigmoid(-10) ≈ 4.5e-5 ≈ 0
        self.gate = nn.Parameter(torch.tensor(config.sadko.cross_attn_gate_init))

    def forward(self, hidden_states, memory_kv=None):
        gate = torch.sigmoid(self.gate)

        if memory_kv is None or gate < 1e-4:
            # v1.0: 无记忆输入或门控≈0，直接返回零张量
            return torch.zeros_like(hidden_states)

        # v2.0/v3.0: 实际 Cross-Attention 计算（见 v2 文档 §5.3）
        B, L, _ = hidden_states.shape
        M = memory_kv.shape[1]

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory_kv).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory_kv).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, self.kv_dim)
        out = self.o_proj(out)

        return gate * out
```

### 5.5 SADKOBlock 与模型组装

```python
# block.py
class SADKOBlock(nn.Module):
    """统一 Decoder Block，根据层索引决定使用标准/改造模块"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        is_anchor = layer_idx in config.sadko.anchor_layers      # [2, 4, 6]
        is_cross = layer_idx in config.sadko.cross_attn_layers   # [3, 5]

        # Attention
        if is_anchor:
            self.self_attn = SADKOSplitGQA(config)
        else:
            self.self_attn = MiniMind3Attention(config)  # 原始 GQA

        # Cross-Attention (仅 L3, L5)
        self.cross_attn = SADKOCrossAttention(config) if is_cross else None

        # FFN
        if is_anchor:
            self.mlp = SADKODualPathFFN(config)
        else:
            self.mlp = SwiGLUFFN(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, position_ids, attention_mask=None,
                memory_kv=None, semantic_tag_emb=None):
        # Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        # Cross-Attention (v1.0 阶段返回零)
        if self.cross_attn is not None:
            hidden_states = hidden_states + self.cross_attn(hidden_states, memory_kv)

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, SADKODualPathFFN):
            ffn_out, gate_val = self.mlp(hidden_states, semantic_tag_emb)
        else:
            ffn_out = self.mlp(hidden_states)
            gate_val = None
        hidden_states = residual + ffn_out

        return hidden_states


class SADKONativeForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.memory_embed = nn.Embedding(16, config.hidden_size)  # 语义标签
        nn.init.zeros_(self.memory_embed.weight)  # 零初始化

        self.layers = nn.ModuleList([
            SADKOBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # 共享权重

    def forward(self, input_ids, position_ids=None, attention_mask=None,
                memory_kv=None, semantic_tags=None):
        hidden_states = self.embed_tokens(input_ids)

        # 语义标签嵌入 (v1.0 阶段全零)
        semantic_tag_emb = None
        if semantic_tags is not None:
            semantic_tag_emb = self.memory_embed(semantic_tags)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_ids, attention_mask,
                memory_kv, semantic_tag_emb
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits
```

---

## 6. 权重初始化策略（从 MiniMind3 迁移）

**核心原则：改造层的前向传播在初始化时与原始模型完全一致。**

```python
def init_sadko_v1_from_minimind(minimind_model, sadko_model):
    """从 MiniMind3 预训练权重初始化 SADKO v1.0"""
    # 1. 复制所有未改造层 (L0, L1, L7) 的完整权重
    for i in [0, 1, 7]:
        copy_layer_weights(minimind_model.layers[i], sadko_model.layers[i])

    # L3, L5: 复制 Self-Attn + FFN，Cross-Attn 骨架随机初始化 (gate=-10 零影响)
    for i in [3, 5]:
        copy_attn_ffn_weights(minimind_model.layers[i], sadko_model.layers[i])

    # 2. 改造层 (L2, L4, L6): Split-GQA 权重拆分
    for i in [2, 4, 6]:
        src = minimind_model.layers[i].self_attn
        dst = sadko_model.layers[i].self_attn

        # Q 投影: 直接复制
        dst.q_proj.weight.data = src.q_proj.weight.data.clone()

        # K 投影: 原始 [384, 768] → 拆分为 k_static [192, 768] + k_dynamic [192, 768]
        k_weight = src.k_proj.weight.data  # [384, 768]
        dst.k_static_proj.weight.data = k_weight[:192].clone()   # 前4个head
        dst.k_dynamic_proj.weight.data = k_weight[192:].clone()  # 后4个head

        # V 投影: 同理
        v_weight = src.v_proj.weight.data
        dst.v_static_proj.weight.data = v_weight[:192].clone()
        dst.v_dynamic_proj.weight.data = v_weight[192:].clone()

        # O 投影: 直接复制
        dst.o_proj.weight.data = src.o_proj.weight.data.clone()

        # Dual-Path FFN: memory_ffn 继承原始权重, reason_ffn 复制一份
        dst.mlp.memory_ffn.load_state_dict(minimind_model.layers[i].mlp.state_dict())
        dst.mlp.reason_ffn.load_state_dict(minimind_model.layers[i].mlp.state_dict())
        # gate 已在 __init__ 中初始化为 0.5 输出

    # 3. Cross-Attention: 随机初始化，gate=-10.0 确保零影响 (已在 __init__ 完成)
    # 4. Memory Embedding: 零初始化 (已在 __init__ 完成)
```

---

## 7. 训练配置

```yaml
# train_v1.yaml
data:
  train_file: pretrain_t2t_mini.jsonl    # MiniMind3 原始 1.2GB 语料
  max_length: 2048
  batch_size: 8
  gradient_accumulation: 4

training:
  epochs: 2
  learning_rate: 3.0e-4                  # 主干
  new_module_lr: 1.0e-4                  # 新增模块 (Dual-Path gate, CA骨架)
  weight_decay: 0.1
  warmup_ratio: 0.05
  lr_scheduler: cosine
  precision: bf16
  max_grad_norm: 1.0
  init_from: minimind3_64m_pretrained.pt

logging:
  eval_steps: 500
  save_steps: 1000
  output_dir: ./checkpoints/sadko_v1
```

### 参数分组策略

```python
def create_optimizer_v1(model, config):
    """新增模块使用更低学习率"""
    new_module_keywords = ['memory_ffn', 'reason_ffn', 'gate_proj', 'cross_attn', 'memory_embedding']

    new_params, base_params = [], []
    for name, param in model.named_parameters():
        if any(kw in name for kw in new_module_keywords):
            new_params.append(param)
        else:
            base_params.append(param)

    return AdamW([
        {'params': base_params, 'lr': 3e-4},
        {'params': new_params, 'lr': 1e-4}
    ], weight_decay=0.01)
```

---

## 8. 验收实验方案

| 实验 | 指标 | 通过标准 | 失败处理 |
| :--- | :--- | :--- | :--- |
| **PPL 保持** | C4 验证集 PPL | ≤ 2.75（原始 ~2.70；超过则灰区 2.75~3.0 触发重训） | 检查初始化/梯度问题 |
| **熵分离** | Static vs Dynamic KV Heads 注意力熵 | **Static 熵 − Dynamic 熵 > 0.3**（Static 长程更平坦） | 检查 RoPE base 是否正确加载；当前分配为 Static=500k/Dynamic=10k |
| **速度开销** | 训练 tokens/s | 下降 ≤ 15%（原始 ~12000 t/s → ≥10200 t/s） | 检查 Split-GQA 实现效率 |
| **门控行为** | Dual-Path gate 值分布 | std > 0.05（非全 0.5，出现自然分化） | 延长训练或调整 gate 初始化 |
| **消融：去 RoPE** | 双 base 改为一致（10k）后 PPL | 退化 > 0.1（证明异构 RoPE 有效） | — |

### 8.1 PPL 保持验证

```python
# eval_v1_ppl.py
def eval_ppl(model, eval_dataset):
    """在 C4 验证集上计算 PPL"""
    model.eval()
    total_loss, total_tokens = 0, 0
    for batch in eval_dataset:
        with torch.no_grad():
            logits = model(batch["input_ids"], batch["position_ids"])
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 6400),
                batch["input_ids"][:, 1:].reshape(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += batch["input_ids"].numel()
    return math.exp(total_loss / total_tokens)

# 通过标准: PPL ≤ 2.75 (原始 MiniMind3 ≈ 2.7)
```

### 8.2 Static/Dynamic 注意力熵分离

**预期方向**（基于交换后的 RoPE base）：Static (base=500k, 慢旋转, 长程平坦注意力) 倾向于更均匀分布 → **熵更高**；Dynamic (base=10k, 快旋转, 短程锐利) 倾向于聚焦局部 → **熵更低**。因此通过标准是 **Static 熵 - Dynamic 熵 > 0.3**（Static 更"散"，Dynamic 更"锐"）。

```python
# eval_v1_entropy.py
def compute_attention_entropy(attn_weights):
    """H = -Σ p·log(p), attn_weights: [B, n_heads, L, L]"""
    entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1)
    return entropy.mean(dim=(0, 2))  # [n_heads]

def evaluate_entropy_separation(model, eval_dataset):
    """验证 Static heads 熵 > Dynamic heads 熵 (Static 长程更平坦，Dynamic 短程更聚焦)"""
    all_entropies = {'static': [], 'dynamic': []}
    for batch in eval_dataset:
        outputs = model(batch['input_ids'], output_attn=True)
        for layer_info in outputs.layer_attn_info:
            if 'attn_s' in layer_info:
                ent_s = compute_attention_entropy(layer_info['attn_s'])
                ent_d = compute_attention_entropy(layer_info['attn_d'])
                all_entropies['static'].append(ent_s.mean().item())
                all_entropies['dynamic'].append(ent_d.mean().item())

    avg_static = np.mean(all_entropies['static'])
    avg_dynamic = np.mean(all_entropies['dynamic'])
    print(f"Static: {avg_static:.4f}, Dynamic: {avg_dynamic:.4f}, Sep: {avg_static - avg_dynamic:.4f}")
    # 通过标准: Static 熵 - Dynamic 熵 > 0.3
    return avg_static - avg_dynamic > 0.3
```

### 8.3 Dual-Path Gate 值分布

```python
# eval_v1_gate.py
def evaluate_gate_distribution(model, eval_dataset):
    """监控门控值分布，确认出现自然分化"""
    all_gates = {layer: [] for layer in [2, 4, 6]}
    for batch in eval_dataset:
        outputs = model(batch['input_ids'], output_gates=True)
        for layer_idx, gate_val in outputs.layer_gates.items():
            if gate_val is not None:
                all_gates[layer_idx].append(gate_val.mean().item())

    for layer_idx, gates in all_gates.items():
        print(f"Layer {layer_idx}: mean={np.mean(gates):.3f}, std={np.std(gates):.3f}")
        # 通过标准: std > 0.05 (非全0.5, 出现分化)
```

---

## 9. v1.0 产出清单与通过条件

```text
v1.0/
├── config_v1.py / sadko_v1_config.yaml
├── rope.py                   # HeterogeneousRotaryEmbedding
├── split_gqa.py              # SADKOSplitGQA
├── dual_path_ffn.py          # SADKODualPathFFN + SwiGLUFFN
├── cross_attn_skeleton.py    # SADKOCrossAttention (骨架)
├── block.py                  # SADKOBlock
├── model_v1.py               # SADKONativeForCausalLM
├── train_v1.py               # 训练入口
├── eval_v1.py                # 验收评估
├── checkpoints/
│   └── sadko_v1_base.pt      # 训练后权重 (v2.0 起点)
└── reports/
    ├── v1_ppl_report.md
    ├── v1_entropy_report.md
    └── v1_validation.md
```

### v1.0 → v2.0 的通过条件

```text
IF PPL ≤ 2.75 AND (Static 熵 - Dynamic 熵) > 0.3 AND gate_std > 0.05:
    → 保存 sadko_v1_base.pt, 进入 v2.0
ELSE IF PPL > 3.0:
    → 检查初始化/梯度问题, 修复后重训
ELSE:
    → 延长训练至 3 epochs, 再次评估
```
