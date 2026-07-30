# SADKO-Native-64M v2.0 架构设计：记忆压缩层（潜空间读写验证）

> **一句话定位**：v2.0 施工图纸——激活 Cross-Attention 并接入 Shared MemPool（MLP 压缩器），验证"远程 KV 压缩到 192 维潜空间并正确读取"的物理可行性
> **上游文档**：[v1-architecture.md](./v1-architecture.md)（v1.0 基座）· [64m-validation-plan.md](./64m-validation-plan.md) §3
> **下游文档**：[v3-architecture.md](./v3-architecture.md)
> **最后更新**：2026-07-29

---

## 1. 设计目标

在 v1.0 基础上，验证**"将远程 KV 压缩到 192 维潜空间并通过 Cross-Attention 正确读取"**的物理可行性。

**核心改造**：

- 激活 Cross-Attention（L3, L5，gate 可学习，sigmoid(-2) ≈ 0.12）
- 实现 Shared MemPool（MLP 压缩 + 内容索引 + 位置偏置）
- 层专属低维读取投影（192→192，**绝不膨胀到 768**）
- 压缩触发器（hot_window=4096, chunk_size=64）
- **位置编码上限扩展到 16K**（v1.0 的 4096 不支持远程记忆实验；NTK-aware 动态缩放）

**隔离原则**：此阶段用 **MLP 压缩器代替 Flow Matching**，隔离验证"压缩-存储-检索-读取"管线，与"压缩质量"两个变量。

**验收标准**：远程检索 Recall@64 > 70%（精确查询），长程 QA 超越纯 hot_window 基线 >15%，16K 上下文峰值显存 < 原始全量 KV 的 60%。

---

## 2. 架构总览

```text
SADKO-Native-64M v2.0 (增量部分标记 ★)
═══════════════════════════════════════════════════════════════
Tokenizer:  BPE + ByteLevel, vocab=6400
Embedding:  Token Embedding (768) + Memory Embedding (768)

Layer 0-1:  [标准层] GQA → SwiGLU (同 v1.0)
Layer 2:    [锚点层] SplitGQA → DualPathFFN (同 v1.0)
Layer 3:    [标准层] GQA → ★CrossAttn(激活, 读取 MemPool) → SwiGLU
Layer 4:    [锚点层] 同 Layer 2
Layer 5:    [标准层] GQA → ★CrossAttn(激活, 读取 MemPool) → SwiGLU
Layer 6-7:  [标准层] (同 v1.0)

LM Head:    768 → 6400

★ Shared MemPool:
    Compressor:  MLP(768→384→192) × 2 (K/V)
    Indexer:     MLP(192→64) + L2_normalize
    PosBias:     Embedding(16 buckets → 1)
    Store:       mem_k[N×192], mem_v[N×192], index[N×64]

★ Cross-Attention Read Heads (L3, L5):
    W_q_L3: 768→192, W_k_L3: 192→192, W_v_L3: 192→192, W_o_L3: 192→768
    W_q_L5: 768→192, W_k_L5: 192→192, W_v_L5: 192→192, W_o_L5: 192→768
═══════════════════════════════════════════════════════════════
新增参数 (vs v1.0): ~2.3M
    ├── Compressor: 0.771M
    ├── Read Projections (L3+L5): 0.295M
    ├── Position Embedding: 0.01M
    └── Cross-Attention Q/O proj: ~1.2M
总参数: ~67.5M
```

### 关键设计决策

| 决策 | 选择 | 理由 |
| :--- | :--- | :--- |
| 压缩器 | MLP（非 Flow Matching） | 隔离变量：先验证"压缩-读取"管线，再验证"压缩质量" |
| 潜空间维度 | 192 | 768/4，压缩比 4:1，64M 容量下的平衡点 |
| 压缩数据源 | **L3 + L5 KV 拼接**（跨 Cross-Attention 读取层，64×768 匹配 kv_input_dim） | L3/L5 是 v2 实际读取记忆的层，它们的 KV 包含跨段全局信号；拼接避免均值池化的维度信息损失 |
| 检索方式 | Cosine 相似度 + Position Bias | 语义 + 顺序双信号 |
| Top-K | 64 chunks | 64×64=4096 tokens 远程上下文 |
| 门控 | 标量 sigmoid（可学习，init=-2.0） | 最简门控，验证"是否需要动态切换" |
| Cross-Attention 空间 | 严格 192 维 | 拒绝膨胀到 768，保持低维正交 |
| 层专属投影 | L3/L5 各自独立的 W_k, W_v | 同一份记忆，不同层解读不同切面 |

---

## 3. 配置定义

```yaml
# sadko_v2_config.yaml
# 继承 v1.0 所有配置，新增/修改以下字段:

sadko:
  # 继承 v1.0
  anchor_layers: [2, 4, 6]
  cross_attn_layers: [3, 5]           # 压缩数据源层
  cross_attn_dim: 192
  cross_attn_heads: 6                # 192 / 32 = 6 heads

  # ===== v2.0 新增: Shared MemPool =====
  mempool:
    latent_dim: 192                  # 潜空间维度
    index_dim: 64                    # 检索索引维度
    hot_window: 4096                 # 近期精确窗口 (锁定)
    chunk_size: 64                   # 压缩粒度
    topk_chunks: 64                  # 检索 Top-K
    max_mem_chunks: 1024             # MemPool 最大容量 (64×1024=65536 tokens)
    compressor_hidden: 384           # MLP 压缩器中间层
    compress_source: "concat_l3_l5"  # 拼接 L3+L5 (而非均值池化)

  # 位置偏置分桶边界
  pos_bucket_boundaries: [0, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536,
                          131072, 262144, 524288, 1048576, 2097152, 4194304, inf]

  # ===== v2.0 修改: Cross-Attention 门控 =====
  cross_attn_gate_init: -2.0         # sigmoid(-2) ≈ 0.12，开始激活

# ===== v2.0 起提升位置编码上限 (用于 8K-16K 长上下文训练) =====
max_position_embeddings: 16384     # 扩展至 16K（v1.0 的 4096 不支持远程记忆实验）
rope_theta: 10000.0                # 保持原 Dynamic base
# 长上下文 RoPE 外推方案：采用 NTK-aware 动态缩放，base 按 (factor * 10000) 调整
# factor = max_position / 4096 = 4 → v2 实际 effective base = 40000
```

```python
# config_v2.py (dataclass 等价形式)
@dataclass
class SADKOv2Config(SADKOv1Config):
    # MemPool
    latent_dim: int = 192
    index_dim: int = 64
    hot_window: int = 4096
    chunk_size: int = 64
    topk_chunks: int = 64
    max_mem_chunks: int = 1024
    compressor_hidden: int = 384
    num_pos_buckets: int = 16

    # Cross-Attention 读取层
    ca_read_layers: list = field(default_factory=lambda: [3, 5])
    ca_n_heads: int = 6
    ca_head_dim: int = 32
    ca_gate_init: float = -2.0

    stage: str = "v2"
```

---

## 4. 核心模块实现

### 4.1 MLP Compressor

```python
# compressor.py
class MLPCompressor(nn.Module):
    """将 768 维 KV（L3+L5 拼接后）压缩到 192 维潜空间

    输入约定：k_repr / v_repr 形状为 [B, 768]，由 memory_write 拼
    接 L3/L5 两层 chunk-均值后的 KV 得到（每层 384 维）。
    """

    def __init__(self, config):
        super().__init__()
        kv_dim = config.hidden_size                        # 768 = 384×2
        latent_dim = config.sadko.mempool.latent_dim       # 192
        hidden = config.sadko.mempool.compressor_hidden    # 384

        self.compress_k = nn.Sequential(
            nn.Linear(kv_dim, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim)
        )
        self.compress_v = nn.Sequential(
            nn.Linear(kv_dim, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim)
        )
        self.index_proj = nn.Linear(latent_dim, config.sadko.mempool.index_dim)  # 192→64

    def forward(self, k_repr, v_repr):
        """
        k_repr, v_repr: [B, 768] = L3_chunk_mean(384) ⊕ L5_chunk_mean(384)
        Returns: mem_k [B,192], mem_v [B,192], index [B,64] (L2 normalized)
        """
        assert k_repr.shape[-1] == self.compress_k[0].in_features, \
            f"Compressor expects 768-dim input, got {k_repr.shape[-1]}"
        mem_k = self.compress_k(k_repr)
        mem_v = self.compress_v(v_repr)
        index = F.normalize(self.index_proj(mem_k), p=2, dim=-1)
        return mem_k, mem_v, index

    def compress(self, kv_layer_a, kv_layer_b):
        """跨层拼接 + 压缩的便捷入口

        输入约定：kv_layer_a 和 kv_layer_b 各自为 [B, n_kv, chunk_size, head_dim]
        先在 chunk 内部 token 维做均值池化得到 [B, n_kv*head_dim] = [B, 384]，
        再按 K/V 通道拼接 L3+L5 两层得 [B, 768]。
        """
        k_a = kv_layer_a.mean(dim=2).reshape(kv_layer_a.size(0), -1)  # [B, 384]
        k_b = kv_layer_b.mean(dim=2).reshape(kv_layer_b.size(0), -1)  # [B, 384]
        k_repr = torch.cat([k_a, k_b], dim=-1)                        # [B, 768]
        v_repr = k_repr                                               # K/V 同源
        return self.forward(k_repr, v_repr)
```

### 4.2 Shared MemPool（运行时存储）

```python
# mempool.py
class SharedMemPool:
    """
    共享记忆存储。每个 Session 仅维护一份物理存储。
    注意：这不是 nn.Module，是运行时数据结构。
    """

    BOUNDARIES = [0, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536,
                  131072, 262144, 524288, 1048576, 2097152, 4194304, float('inf')]

    def __init__(self, config):
        self.latent_dim = config.sadko.mempool.latent_dim
        self.index_dim = config.sadko.mempool.index_dim
        self.max_chunks = config.sadko.mempool.max_mem_chunks
        self.topk = config.sadko.mempool.topk_chunks
        self.device = 'cuda'

        self.mem_k = torch.zeros(self.max_chunks, self.latent_dim, device=self.device)
        self.mem_v = torch.zeros(self.max_chunks, self.latent_dim, device=self.device)
        self.index = torch.zeros(self.max_chunks, self.index_dim, device=self.device)
        self.chunk_end = torch.zeros(self.max_chunks, dtype=torch.long, device=self.device)
        self.pos_bucket = torch.zeros(self.max_chunks, dtype=torch.long, device=self.device)
        self.valid_mask = torch.zeros(self.max_chunks, dtype=torch.bool, device=self.device)
        self.current_size = 0

    def append(self, mem_k, mem_v, index, chunk_end_pos, current_pos):
        """追加一个 chunk 到 MemPool (满时 FIFO 淘汰最早一半)"""
        idx = self.current_size
        if idx >= self.max_chunks:
            half = self.max_chunks // 2
            for buf in [self.mem_k, self.mem_v, self.index, self.chunk_end,
                        self.pos_bucket, self.valid_mask]:
                buf[:half] = buf[half:2 * half]
            self.current_size = half
            idx = self.current_size

        self.mem_k[idx] = mem_k
        self.mem_v[idx] = mem_v
        self.index[idx] = index
        self.chunk_end[idx] = chunk_end_pos
        self.pos_bucket[idx] = self._bucket_distance(current_pos - chunk_end_pos)
        self.valid_mask[idx] = True
        self.current_size += 1

    def _bucket_distance(self, distance):
        for i, bound in enumerate(self.BOUNDARIES):
            if distance <= bound:
                return i
        return len(self.BOUNDARIES) - 1

    def get_valid_slice(self):
        n = self.current_size
        return {
            'mem_k': self.mem_k[:n], 'mem_v': self.mem_v[:n],
            'index': self.index[:n], 'chunk_end': self.chunk_end[:n],
            'pos_bucket': self.pos_bucket[:n], 'valid_mask': self.valid_mask[:n]
        }
```

### 4.3 Cross-Attention Read Head（层专属低维读取，v2.0 激活版）

```python
# ca_read_head.py
class CrossAttentionReadHead(nn.Module):
    """
    层专属低维 Cross-Attention.
    核心原则: 严格在 192 维空间计算, 绝不膨胀到 768 维.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.latent_dim = config.sadko.mempool.latent_dim  # 192
        self.n_heads = config.sadko.cross_attn_heads        # 6
        self.head_dim = self.latent_dim // self.n_heads     # 32
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(config.hidden_size, self.latent_dim, bias=False)   # 768→192
        # ★ 层专属 K/V 投影: 192 → 192 (解读同一份记忆的不同切面)
        self.k_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.v_proj = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.o_proj = nn.Linear(self.latent_dim, config.hidden_size, bias=False)   # 192→768

        self.gate = nn.Parameter(torch.tensor(config.sadko.cross_attn_gate_init))  # -2.0
        self.pos_embedding = nn.Embedding(config.sadko.mempool.num_pos_buckets, 1)
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, hidden_states, mempool_data, current_pos):
        """
        hidden_states: [B, L, 768]
        mempool_data: SharedMemPool.get_valid_slice()
        Returns: (output [B, L, 768], gate_val scalar)
        """
        n = mempool_data['mem_k'].size(0)
        gate_val = torch.sigmoid(self.gate)
        if n == 0 or gate_val < 1e-3:
            return torch.zeros_like(hidden_states), gate_val

        B, L, _ = hidden_states.shape

        # 1. Query 投影: 768 → 192
        q = self.q_proj(self.norm(hidden_states))
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, 6, L, 32]

        # 2. 层专属 K/V 投影: 192 → 192
        k = self.k_proj(mempool_data['mem_k']).view(n, self.n_heads, self.head_dim)
        v = self.v_proj(mempool_data['mem_v']).view(n, self.n_heads, self.head_dim)
        k = k.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 6, N, 32]
        v = v.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)

        # 3. Attention Score (无因果 Mask — 记忆读取是全局的)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, 6, L, N]

        # 4. 位置偏置
        pos_bias = self.pos_embedding(mempool_data['pos_bucket']).squeeze(-1)  # [N]
        attn_scores = attn_scores + pos_bias.view(1, 1, 1, -1)

        # 5. Valid Mask (含因果约束: 只检索当前位置之前的 chunks)
        causal_valid = (mempool_data['chunk_end'] < current_pos) & mempool_data['valid_mask']
        attn_scores = attn_scores.masked_fill(~causal_valid.view(1, 1, 1, -1), float('-inf'))

        # 6. Softmax + Output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, 6, L, 32]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.latent_dim)

        # 7. 输出投影 + 门控
        out = self.o_proj(attn_output)
        return gate_val * out, gate_val
```

### 4.4 压缩触发器（Memory Write Pipeline）

```python
# memory_write.py
class MemoryWritePipeline:
    """当 token 离开 hot_window 时，触发跨层融合压缩。"""

    def __init__(self, config, compressor: MLPCompressor, mempool: SharedMemPool):
        self.config = config
        self.compressor = compressor
        self.mempool = mempool
        self.hot_window = config.sadko.mempool.hot_window
        self.chunk_size = config.sadko.mempool.chunk_size

    def check_and_compress(self, kv_caches, current_pos):
        """
        kv_caches: dict[layer_idx] -> {'k_static': [B,n_kv,L,D], 'v_static': ..., ...}
        触发条件: current_pos > hot_window + chunk_size
        压缩数据源: L3 + L5 两层 KV 拼接（L3/L5 是 Cross-Attention 读取层）
        """
        compress_boundary = current_pos - self.hot_window
        if compress_boundary < self.chunk_size:
            return

        source_layers = self.config.sadko.cross_attn_layers   # [3, 5]
        if not all(l in kv_caches for l in source_layers):
            return

        # Step 1: 取出最早的 chunk [B, n_kv, chunk_size, head_dim]
        kv_l3 = kv_caches[source_layers[0]]
        kv_l5 = kv_caches[source_layers[1]]

        k_l3 = kv_l3['k_static'][:, :, :self.chunk_size, :]   # [B, n_kv, 64, 48]
        k_l5 = kv_l5['k_static'][:, :, :self.chunk_size, :]
        v_l3 = kv_l3['v_static'][:, :, :self.chunk_size, :]
        v_l5 = kv_l5['v_static'][:, :, :self.chunk_size, :]

        # Step 2: 在 chunk 内部 token 维做均值池化 → [B, n_kv*head_dim] = [B, 384]
        k_l3_repr = k_l3.mean(dim=2).reshape(k_l3.size(0), -1)  # [B, 384]
        k_l5_repr = k_l5.mean(dim=2).reshape(k_l5.size(0), -1)
        v_l3_repr = v_l3.mean(dim=2).reshape(v_l3.size(0), -1)
        v_l5_repr = v_l5.mean(dim=2).reshape(v_l5.size(0), -1)

        # Step 3: 拼接两层 → [B, 768]
        k_repr = torch.cat([k_l3_repr, k_l5_repr], dim=-1)
        v_repr = torch.cat([v_l3_repr, v_l5_repr], dim=-1)

        with torch.no_grad():
            mem_k, mem_v, index = self.compressor(k_repr, v_repr)

        self.mempool.append(
            mem_k, mem_v, index,
            chunk_end_pos=self.chunk_size, current_pos=current_pos
        )

        # 从 KV Cache 中移除已压缩的 chunk (保持 hot_window)
        for layer_idx in kv_caches:
            kv = kv_caches[layer_idx]
            for key in ['k_static', 'v_static', 'k_dynamic', 'v_dynamic']:
                kv[key] = kv[key][:, :, self.chunk_size:, :]
```

---

## 5. 训练配置

```yaml
# train_v2.yaml
data:
  long_text_file: long_corpus_8k_16k.jsonl   # 8K-16K 长文本 (~200MB, 书籍/论文)
  qa_file: remote_recall_qa_50k.jsonl         # 远程事实召回 QA 对 (~50K 条)
  max_length: 8192
  batch_size: 2                               # 长文本显存受限
  gradient_accumulation: 16

training:
  stage_2a:                                   # 长上下文预训练
    epochs: 1
    learning_rate: 1.0e-4                     # 主干
    mempool_lr: 5.0e-5                        # MemPool 模块
    ca_lr: 5.0e-5
    precision: bf16
    init_checkpoint: ./checkpoints/sadko_v1_base.pt

  stage_2b:                                   # 记忆检索微调
    epochs: 2-3
    learning_rate: 5.0e-5
    mempool_lr: 2.0e-5
    metrics: [Recall@64, EM, F1]
```

---

## 6. 验收实验方案

| 实验 | 指标 | 通过标准 | 失败处理 |
| :--- | :--- | :--- | :--- |
| **压缩重构** | 压缩后 KV 重构 → PPL | PPL 增加 < 10% | 增大潜空间维度或压缩器容量 |
| **检索准确率** | Recall@64 (精确查询) | > 70% | 检查 Indexer 训练 |
| **模糊检索** | Recall@64 (语义模糊查询) | > 40% | 引入内容寻址（v3.0 前置） |
| **长程 QA** | 8K+ 文本 QA 准确率 | > 纯 hot_window 基线 +15% | 检查 Position Bias 有效性 |
| **门控行为** | gate 值在不同任务下的分布 | 背诵时 gate↑，推理时 gate↓ | 若始终≈0.5，v3.0 需重新设计门控 |
| **显存** | 16K 上下文峰值显存 | < 原始全量 KV 的 60%（原始 ~2.1GB → 目标 <1.26GB） | 验证压缩收益 |

### 6.1 压缩重构质量

```python
def eval_compression_reconstruction(model, compressor, eval_data):
    """压缩 KV → 重构 → 测量 PPL 增加"""
    original_ppl = eval_ppl(model, eval_data)
    # 压缩 → 重构 → 注入回模型计算 PPL
    reconstructed_ppl = eval_ppl_with_compressed_kv(model, compressor, eval_data)
    ppl_increase = (reconstructed_ppl - original_ppl) / original_ppl
    assert ppl_increase < 0.10, f"PPL 退化 {ppl_increase:.1%} 超过阈值 10%"
```

### 6.2 检索准确率

```python
def eval_retrieval_recall(model, mempool, qa_data):
    """远程事实召回: 答案在 hot_window 之外 (5000+ 位置)"""
    hits, total = 0, 0
    for qa in qa_data:
        q_idx = F.normalize(model.mempool.compressor.index_proj(qa['query_z']), p=2, dim=-1)
        scores = q_idx @ mempool.index[:mempool.current_size].T
        top_indices = scores.topk(64).indices
        if qa['gold_chunk_idx'] in top_indices:
            hits += 1
        total += 1
    recall = hits / total
    assert recall > 0.70, f"Recall@64 = {recall:.1%} < 70%"
```

### 6.3 长程 QA 与显存

```python
def eval_long_qa(model, qa_dataset):
    baseline_acc = eval_with_hot_window_only(model, qa_dataset)
    sadko_acc = eval_with_mempool(model, qa_dataset)
    assert sadko_acc > baseline_acc + 0.15, \
        f"提升 {sadko_acc - baseline_acc:.1%} < 15%"

# 显存验证: nvidia-smi 监控 16K 上下文峰值
# 原始 16K KV: ~2.1GB (8层 × 8heads × 48dim × 16K × 2(K+V) × 2bytes)
# v2.0 目标: < 1.26GB (60%)
```

---

## 7. v2.0 产出清单与通过条件

```text
v2.0/
├── config_v2.py / sadko_v2_config.yaml
├── compressor.py             # MLPCompressor
├── mempool.py                # SharedMemPool
├── ca_read_head.py           # CrossAttentionReadHead
├── memory_write.py           # MemoryWritePipeline
├── model_v2.py               # SADKONativeV2
├── train_v2.py
├── eval_v2.py
├── checkpoints/
│   ├── stage_2a.pt
│   └── sadko_v2_mem.pt       # 含 MemPool 的权重 (v3.0 起点)
└── reports/
    ├── v2_compression_report.md
    ├── v2_retrieval_report.md
    ├── v2_validation.md
    └── v2_falsified.md       # 已证伪/已验证机制清单
```

### v2.0 → v3.0 的通过条件

```text
IF Recall@64 > 70% AND PPL 增加 < 10% AND 长程QA提升 > 15%:
    → 保存 sadko_v2_mem.pt, 进入 v3.0
ELSE:
    → 调整潜空间维度/压缩器容量, 不进入 v3.0
```
