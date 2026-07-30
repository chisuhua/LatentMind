# SADKO-Native-64M v3.0 架构设计：灵魂注入层（完整机制验证）

> **一句话定位**：v3.0 施工图纸——替换 MLP 压缩器为 ELF-Lite（Flow Matching）、引入 FSQ 离散化与内容寻址、执行扩散对齐与四大核心实验，产出《已验证/已证伪机制清单》
> **上游文档**：[v2-architecture.md](./v2-architecture.md)（v2.0 管线）· [whitepaper.md](./whitepaper.md) §4（四大实验理论依据）
> **最后更新**：2026-07-29

---

## 1. 设计目标

在 v2.0 验证"压缩-读取管线可行"的基础上：

- **替换** MLP Compressor → 右脑 ELF-Lite（双向 Transformer + Flow Matching）
- **新增** FSQ 离散化 + 内容寻址 Router
- **替换** 标量门控 → 动态门控网络
- **新增** 左脑校验模块（置信度阈值过滤）
- **执行** Phase 1-5 完整训练流水线 + 四大核心实验（消融隔离）

**验收标准**：四大实验通过，零样本跨域联想涌现；产出《已验证/已证伪机制清单》作为 300M 迁移的唯一宪法。

---

## 2. 架构总览

```text
SADKO-Native-64M v3.0 (全架构)
═══════════════════════════════════════════════════════════════

[左脑 AR-Native] (继承 v1.0 Split-GQA + Dual-Path FFN, v2.0 CA 读取)
  Layer 0-1: 标准层
  Layer 2:    锚点层 (Split-GQA + Dual-Path FFN)
  Layer 3:    ★ Cross-Attn (FSQ Router 检索 → 层专属读取)
  Layer 4:    锚点层
  Layer 5:    ★ Cross-Attn (FSQ Router 检索 → 层专属读取)
  Layer 6-7:  标准层
  ★ 新增: 动态门控网络 (替代标量 gate)
  ★ 新增: 左脑校验模块 (置信度过滤)

[右脑 ELF-Lite] (★ 替换 MLP Compressor, ~5M)
  TransformerEncoder × 2 (双向注意力, 无因果 Mask)
  Flow Matching Head (速度场预测)
  输入: chunk KV [64, 768] → 输出: z [192]

[FSQ 量化器] (★ 新增)
  levels = [8, 8, 8, 8] 或 [8, 8, 4]
  codebook_size = 256
  z_continuous [192] → z_quantized [192] + code_index [int]
  Phase 3 后码本完全冻结

[内容寻址 Router] (★ 替换 Indexer)
  输入: z_block [192] → 投影 [64]
  相似度: Cosine(query, FSQ_codebook_embeddings[256×64])
  + Position Bias (保留 v2.0)

[扩散对齐桥梁] (★ 新增训练阶段)
  Teacher: ELF-Lite (冻结)
  Student: 左脑 Memory Embedding + 顶层
  Loss: 0.3×KL + 0.7×CE
  训练后: Teacher 丢弃, 推理零开销
═══════════════════════════════════════════════════════════════
参数变化 (vs v2.0):
    ├── 移除 MLP Compressor: -0.771M
    ├── 新增 ELF-Lite: +5.0M
    ├── 新增 FSQ: +0.05M
    ├── 新增 code_embed (FSQ→KV 嵌入表): +0.196M
    ├── 新增动态门控: +0.15M
    ├── 新增左脑校验: +0.01M
    ├── 新增实验 E 简化 MoE (2 experts × 锚点层 3): +9.4M
    └── 净增: ~14.0M → 总参数 ~93M
    注：v1.0 已含 Dual-Path FFN 改造（基座 64M + ~15.5M = ~80M），
       见 v1-arch §3 与 open-issues.md B-01
```

---

## 3. 配置定义

```yaml
# sadko_v3_config.yaml
# 继承 v2.0 所有配置，替换/新增以下字段:

sadko:
  anchor_layers: [2, 4, 6]
  cross_attn_layers: [3, 5]
  cross_attn_dim: 192
  cross_attn_heads: 6
  mempool:
    latent_dim: 192
    index_dim: 64
    hot_window: 4096
    chunk_size: 64
    topk_chunks: 64

  # ===== v3.0 新增: 右脑 ELF-Lite =====
  elf:
    encoder_layers: 2
    encoder_heads: 4
    encoder_dim: 192            # ELF 内部计算维度
    flow_matching_steps: 4      # ODE 积分步数
    kv_input_dim: 768           # 输入 KV 维度
    elf_chunk_len: 64           # 输入 chunk 长度

  # ===== v3.0 新增: FSQ 量化器 =====
  fsq:
    codebook_size: 256          # 64M 验证用
    levels: [8, 8, 8, 8]        # 4维标量量化 (备选: [8,8,4])
    latent_dim: 192
    freeze_after_phase3: true   # Phase 3 后冻结

  # ===== v3.0 修改: 动态门控 =====
  gate:
    type: dynamic               # v2.0 为 scalar, v3.0 为 dynamic MLP
    input_dim: 784              # hidden(768) + semantic_tag(16)
    hidden_dim: 192

  # ===== v3.0 新增: 左脑校验 =====
  verifier:
    type: confidence_threshold
    entropy_threshold: 0.7      # 初始化，可学习
    max_retrieval: 3            # 最大二次检索次数

  # ===== v3.0 新增: 扩散对齐 =====
  diffusion:
    kl_weight: 0.3
    ce_weight: 0.7
    steps: 3000                 # 2000-3000
    teacher_freeze: true
```

```python
# config_v3.py (dataclass 等价形式)
@dataclass
class SADKOv3Config(SADKOv2Config):
    stage: str = "v3"
    # ELF-Lite
    elf_num_layers: int = 2
    elf_num_heads: int = 4
    elf_hidden_size: int = 384
    elf_chunk_len: int = 64
    # FSQ
    fsq_levels: list = field(default_factory=lambda: [8, 8, 8, 8])
    fsq_codebook_size: int = 256
    fsq_freeze_after_phase3: bool = True
    # Router
    router_proj_dim: int = 64
    # 动态门控
    gate_input_dim: int = 784
    gate_hidden_dim: int = 192
    # 左脑校验
    verifier_entropy_threshold: float = 0.7
    verifier_max_retry: int = 1
    # 扩散对齐
    diffusion_kl_weight: float = 0.3
    diffusion_ce_weight: float = 0.7
    diffusion_steps: int = 3000
    diffusion_teacher_freeze: bool = True
```

---

## 4. 核心模块实现

### 4.1 右脑 ELF-Lite

```python
# elf_lite.py
class ELFLite(nn.Module):
    """
    右脑语义压缩引擎: 双向注意力 + Flow Matching.
    无因果 Mask, 看到完整 chunk 后全局压缩. 参数量 ~5M.
    """

    def __init__(self, config):
        super().__init__()
        self.kv_dim = config.sadko.elf.kv_input_dim      # 768
        self.latent_dim = config.sadko.elf.encoder_dim    # 192
        self.chunk_len = config.sadko.elf.elf_chunk_len   # 64

        # 输入投影: KV 768 → 192 (备选: 经 384 中间维)
        self.input_proj = nn.Linear(self.kv_dim, self.latent_dim)
        self.pos_emb = nn.Embedding(self.chunk_len, self.latent_dim)

        # ★ 双向 Transformer Encoder (无因果 Mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=config.sadko.elf.encoder_heads,          # 4
            dim_feedforward=self.latent_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.sadko.elf.encoder_layers)  # 2

        # 池化: [64, 192] → [192] (加权池化或均值)
        self.pool = nn.Linear(self.latent_dim, 1, bias=False)
        self.output_proj = nn.Linear(self.latent_dim, self.latent_dim)

        # Flow Matching Head: 速度场预测
        self.time_embed = SinusoidalPositionEmbedding(self.latent_dim)
        self.velocity_pred = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def encode(self, kv_chunk):
        """
        kv_chunk: [B, chunk_len, kv_dim] (来自左脑锚点层的 KV)
        Returns: z_continuous [B, latent_dim]
        """
        x = self.input_proj(kv_chunk)  # [B, 64, 192]
        positions = torch.arange(self.chunk_len, device=x.device)
        x = x + self.pos_emb(positions).unsqueeze(0)
        encoded = self.encoder(x)      # [B, 64, 192] 双向注意力

        # 加权池化
        weights = F.softmax(self.pool(encoded).squeeze(-1), dim=-1)  # [B, 64]
        pooled = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)  # [B, 192]
        return self.output_proj(pooled)

    def flow_matching_loss(self, z_target):
        """
        Flow Matching: 学习从噪声到数据的速度场.
        z_target: [B, latent_dim] (encode 输出的目标潜向量)
        """
        B = z_target.size(0)
        t = torch.rand(B, 1, device=z_target.device)      # 均匀采样 [0,1]
        noise = torch.randn_like(z_target)

        # 插值路径: x_t = (1-t)*noise + t*z
        x_t = (1 - t) * noise + t * z_target
        v_target = z_target - noise                        # 目标速度

        t_emb = self.time_embed(t.squeeze(-1))
        v_pred = self.velocity_pred(torch.cat([x_t, t_emb], dim=-1))
        return F.mse_loss(v_pred, v_target)
```

### 4.2 FSQ 量化器

```python
# fsq.py
class FSQQuantizer(nn.Module):
    """
    Finite Scalar Quantization (有限标量量化).
    避免 VQ-VAE 的码本坍缩, 保留流形几何特性, 支持 Slerp 插值.
    levels=[8,8,8,8] → 理论 4096 组合, 实际使用 256 码本.
    """

    def __init__(self, config):
        super().__init__()
        self.levels = config.sadko.fsq.levels              # [8, 8, 8, 8]
        self.codebook_size = config.sadko.fsq.codebook_size  # 256
        self.latent_dim = config.sadko.fsq.latent_dim      # 192
        self.n_quantize_dims = len(self.levels)             # 4

        # 量化投影: 192 → 4; 反量化: 4 → 192
        self.proj_down = nn.Linear(self.latent_dim, self.n_quantize_dims)
        self.proj_up = nn.Linear(self.n_quantize_dims, self.latent_dim)

        # 码本嵌入 (用于内容寻址 Router)
        self.codebook_embeddings = nn.Embedding(
            self.codebook_size, config.sadko.mempool.index_dim)  # 256×64

        self.frozen = False

    def _quantize_scalar(self, x, levels):
        """标量量化: x ∈ [-1,1] → levels 个离散级别"""
        half_l = (levels - 1) / 2
        return (x * half_l).round() / half_l

    def forward(self, z_continuous):
        """
        z_continuous: [B, 192]
        Returns: z_quantized [B, 192], codes [B]
        """
        z_proj = torch.tanh(self.proj_down(z_continuous))  # [B, 4], 限制到 [-1,1]

        # 逐维标量量化
        z_quant_list, code_parts = [], []
        for i, level in enumerate(self.levels):
            q = self._quantize_scalar(z_proj[:, i:i + 1], level)
            z_quant_list.append(q)
            half_l = (level - 1) / 2
            code_parts.append(((q.squeeze(-1) * half_l) + half_l).long())

        z_quant = torch.cat(z_quant_list, dim=-1)          # [B, 4]

        # 组合为单一码本索引
        codes = code_parts[0]
        for i in range(1, len(code_parts)):
            codes = codes * self.levels[i] + code_parts[i]
        codes = codes.clamp(0, self.codebook_size - 1)

        # 反量化投影 + 直通估计器 (STE)
        z_quantized = self.proj_up(z_quant)                # [B, 192]
        z_quantized = z_continuous + (z_quantized - z_continuous).detach()

        return z_quantized, codes

    def get_codebook_embeddings(self):
        """返回码本嵌入矩阵 [256, 64], 供 Router 使用"""
        indices = torch.arange(self.codebook_size,
                               device=self.codebook_embeddings.weight.device)
        return self.codebook_embeddings(indices)

    def freeze_codebook(self):
        """Phase 3 后调用"""
        self.frozen = True
        for p in self.parameters():
            p.requires_grad = False
```

### 4.3 FSQ 码字 → KV 嵌入表（code_embed）

```python
# code_embed.py
class FSQCodeEmbedding(nn.Module):
    """独立的 FSQ 码字 → Memory KV 投影表

    注意：v1.0 的 memory_embed(Embedding(16, 768)) 是语义标签表，仅 16 条目；
    此处 code_embed 是 v3.0 专用 FSQ 码字（256 条目）→ KV 嵌入。
    两者职责分离，不能混用。
    """

    def __init__(self, config):
        super().__init__()
        self.codebook_size = config.sadko.fsq.codebook_size   # 256
        self.hidden_size = config.hidden_size                 # 768

        # 直接嵌入表
        self.embed = nn.Embedding(self.codebook_size, self.hidden_size)

        # 初始化: 用小标准差, 避免初始信号过强
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, codes):
        """
        codes: [B] 或 [B, M] (long tensor, 值域 [0, codebook_size))
        Returns: [B, hidden_size] 或 [B, M, hidden_size]
        """
        return self.embed(codes)

    def extra_repr(self):
        return f"codebook_size={self.codebook_size}, hidden_size={self.hidden_size}"
```

### 4.4 内容寻址 Router

```python
# router.py
class ContentAddressingRouter(nn.Module):
    """
    基于 FSQ 码本嵌入的内容寻址.
    替代 v2.0 的纯 Cosine 索引器.
    """

    def __init__(self, config):
        super().__init__()
        self.query_proj = nn.Linear(config.sadko.mempool.latent_dim,
                                    config.sadko.mempool.index_dim)  # 192→64
        self.pos_embedding = nn.Embedding(config.sadko.mempool.num_pos_buckets, 1)

    def forward(self, z_block, fsq_codebook_embeddings, mempool_data):
        """
        z_block: [B, 192] 当前查询的潜向量
        fsq_codebook_embeddings: [256, 64]
        mempool_data: SharedMemPool.get_valid_slice() (含 fsq_codes)
        Returns: top_indices [B, topk], scores [B, N]
        """
        # 1. Query 投影
        query = F.normalize(self.query_proj(z_block), p=2, dim=-1)  # [B, 64]

        # 2. MemPool 条目经 FSQ 码字映射到码本嵌入空间
        store_embeddings = fsq_codebook_embeddings[mempool_data['fsq_codes']]  # [N, 64]
        store_embeddings = F.normalize(store_embeddings, p=2, dim=-1)

        # 3. Cosine 相似度
        scores = query @ store_embeddings.T                # [B, N]

        # 4. 位置偏置 (保留 v2.0)
        pos_bias = self.pos_embedding(mempool_data['pos_bucket']).squeeze(-1)
        scores = scores + pos_bias.unsqueeze(0)

        # 5. Valid Mask
        scores = scores.masked_fill(~mempool_data['valid_mask'].unsqueeze(0), float('-inf'))

        # 6. Top-K
        topk = min(64, scores.size(-1))
        top_indices = scores.topk(topk, dim=-1).indices
        return top_indices, scores
```

### 4.5 动态门控网络

```python
# dynamic_gate.py
class DynamicGateNetwork(nn.Module):
    """
    替代 v2.0 的标量门控.
    输入: hidden_state (768) + semantic_tag (16) = 784
    g → 1: 偏左脑 (背诵); g → 0: 偏右脑 (推理)
    """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.sadko.gate.input_dim, config.sadko.gate.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.sadko.gate.hidden_dim, 1)
        )
        nn.init.constant_(self.net[-1].bias, 0.0)  # 中性初始化

    def forward(self, hidden_state, semantic_tag=None):
        """
        hidden_state: [B, L, 768]
        semantic_tag: [B, L, 16] or None (推理时无 tag 用零向量)
        Returns: g [B, L, 1] ∈ [0,1]
        """
        if semantic_tag is None:
            semantic_tag = torch.zeros(hidden_state.size(0), hidden_state.size(1), 16,
                                       device=hidden_state.device)
        gate_input = torch.cat([hidden_state, semantic_tag], dim=-1)
        return torch.sigmoid(self.net(gate_input))
```

### 4.6 左脑校验模块

```python
# verifier.py
class LeftBrainVerifier(nn.Module):
    """
    生成前置信度阈值过滤.
    当 LM Head 输出熵超过阈值时, 触发二次检索 MemPool.
    """

    def __init__(self, config):
        super().__init__()
        self.threshold = nn.Parameter(
            torch.tensor(config.sadko.verifier.entropy_threshold))
        self.max_retry = config.sadko.verifier.max_retrieval

    def compute_confidence(self, logits):
        """
        logits: [B, vocab_size] (V=6400)
        Returns: norm_entropy [B] ∈ [0,1], is_confident [B]
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        max_entropy = math.log(logits.size(-1))  # ≈ 8.76
        norm_entropy = entropy / max_entropy
        is_confident = norm_entropy < torch.sigmoid(self.threshold)
        return norm_entropy, is_confident

    def forward(self, logits, memory_context=None):
        """
        Returns: final_logits, retry_count
        低置信度 → 触发二次检索 (在推理循环中处理)
        """
        norm_entropy, is_confident = self.compute_confidence(logits)
        if is_confident.all():
            return logits, 0
        return logits, (~is_confident).sum().item()
```

### 4.7 扩散对齐训练接口

```python
# diffusion_align.py
def diffusion_alignment_step(elf_teacher, ar_student, kb_batch, fsq, config):
    """
    Phase 4: 扩散桥梁对齐.
    Teacher (ELF, 冻结) 提供连续语义监督 → Student (AR) 学习拟合.
    """
    # 1. Teacher: 生成连续语义分布
    with torch.no_grad():
        z_teacher = elf_teacher.encode(kb_batch["kv_chunk"])  # [B, 192]
        p_teacher = F.softmax(z_teacher / 0.1, dim=-1)        # 温度缩放

    # 2. Student: 通过 FSQ 码字 + code_embed 读取（不是 v1.0 的 memory_embed）
    z_quantized, codes = fsq(z_teacher)
    memory_kv = ar_student.code_embed(codes)                  # [B, 768]
    outputs = ar_student(kb_batch["input_ids"], memory_kv=memory_kv)
    p_student = F.softmax(outputs.memory_hidden / 0.1, dim=-1)

    # 3. 对齐损失: 0.3×KL + 0.7×CE
    loss_kl = F.kl_div(p_student.log(), p_teacher, reduction='batchmean')
    loss_ce = outputs.lm_loss
    return config.diffusion_kl_weight * loss_kl + config.diffusion_ce_weight * loss_ce
```

---

## 5. Phase 1-5 完整训练流水线

```yaml
# train_v3_pipeline.yaml
phase_1_elf_pretrain:
  data: kb_corpus.jsonl                    # 知识库文本 → MiniMind Prefill → KV Cache
  model: ELFLite
  loss: flow_matching + 0.5×reconstruction
  epochs: 5
  lr: 1.0e-4
  hardware: 单卡 3090, ~4h
  checkpoint: elf_lite.pt
  acceptance: 重构 KV 后 PPL 增加 < 5%

phase_2_fsq:
  data: ELF 编码后的连续潜向量
  model: FSQQuantizer
  loss: quantization_error (MSE)
  epochs: 3
  lr: 5.0e-4
  duration: ~2h
  checkpoint: fsq_codebook.json
  acceptance: 码字插值语义合理率 ≥ 60%
  # 验证方法: 取 "猫" 和 "狗" 的码字, 5 等分插值, 解码后评估语义连续性

phase_3_mechanical_alignment:
  data: kb_corpus.jsonl
  model: 左脑 Memory Embedding + Cross-Attn + 顶层 2 层
  freeze: 左脑底层 6 层
  loss: 0.5×MSE(左脑KV, 右脑KV) + 0.5×CE
  epochs: 2
  lr: 1.0e-4
  duration: ~3h
  checkpoint: memory_aligned.pt
  acceptance: 给定 FSQ 码字，左脑复述 EM > 80%
  action: 完成后调用 fsq.freeze_codebook()

phase_4_diffusion_alignment:               # ← 注入灵魂
  data: kb_corpus.jsonl (仅知识库)
  teacher: elf_lite.pt (冻结)
  student: 左脑顶层 2 层 + Memory Embedding
  loss: 0.3×KL + 0.7×CE
  steps: 2000-3000
  lr: 5.0e-5
  duration: ~2h
  checkpoint: sadko_v3_aligned.pt (phase4_soul.pt)
  acceptance: 语义理解提升 > 8%, 零样本联想涌现

phase_5_full_sft:
  data: retrieval_qa_pairs.jsonl (带检索的 QA 对 + 混合任务)
  model: 全参数解冻
  loss: CE + gate_regularization
  epochs: 2
  lr: 3.0e-4
  duration: ~4h
  checkpoint: sadko_v3_final.pt
  acceptance: SADKO-MiniMind > 纯 MiniMind SFT
```

### Phase 3.5 机械对齐训练要点

```python
# train_phase3_5.py
def train_mechanical_alignment(config, ar_model, elf_model, fsq_model):
    # 冻结左脑底层 6 层
    for name, param in ar_model.named_parameters():
        if any(f'layers.{i}' in name for i in range(6)):
            param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, ar_model.parameters()), lr=1e-4)

    for batch in kb_dataloader:
        # 1. 右脑编码 (冻结)
        with torch.no_grad():
            z = elf_model.encode(batch['kv_chunk'])
            z_q, codes = fsq_model(z)
        # 2. 左脑通过 Memory Embedding 读取
        memory_kv = ar_model.memory_embed(codes)
        outputs = ar_model(batch['input_ids'], memory_kv=memory_kv)
        # 3. Loss
        loss = 0.5 * F.mse_loss(outputs.memory_kv_pred, z_q) + 0.5 * outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert eval_exact_match(ar_model, test_data) > 0.80
    fsq_model.freeze_codebook()  # Phase 3 后码本完全冻结
```

---

## 6. 五大核心实验（消融隔离，严格 A→B→C→D→E 顺序）

| 实验 | 对照组 | 实验组 | 数据 | 通过标准 | 证伪标准 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. 内容寻址** | 纯符号路由（地址标签硬匹配） | FSQ 码本 Cosine 内容寻址 | 500 模糊查询 + 500 精确查询 | 模糊 Recall 提升 **>20%** | 提升 <10% 或精确 EM 下降 >5% |
| **B. 拓扑缓存** | 线性 Top-K 独立检索 | FSQ 码字排序 + 滑动窗口连续检索 | 200 跨段落关联查询 | 关联 F1 显著提升 | F1 无提升或延迟增加 >20% |
| **C. 动态门控** | 固定 0.5:0.5 混合 | 可学习动态门控网络 | 50% 纯背诵 + 50% 推理/摘要 | 背诵 g→1(>0.6), 推理 g→0(<0.4) | 门控始终≈0.5 或震荡 |
| **D. 左脑校验** | 无校验直接输出 | 置信度阈值过滤 + 二次检索 | 含 20% 故意注入干扰项 | 过滤率 **>70%**, 误杀 **<10%** | 过滤率 <50% 或误杀 >20% |
| **E. 简化版 Latent MoE** | 全 expert 并行推理 | 基于 FSQ 码字 hash 路由 (2 experts) | 500 多域混合推理 | 路由一致性 >70% 且 per-expert EM 不退化 >3% | 路由随机或单 expert 主导 >90% |

### 6.1 实验 A：内容寻址

```python
# experiment_a.py
def run_experiment_a(config, model, fsq):
    fuzzy_queries = load_dataset('test/fuzzy_queries_500.jsonl')
    exact_queries = load_dataset('test/exact_queries_500.jsonl')

    # 对照组: 纯符号路由
    model.use_content_addressing = False
    fuzzy_recall_ctrl = evaluate_recall(model, fuzzy_queries, k=5)
    exact_em_ctrl = evaluate_em(model, exact_queries)

    # 实验组: FSQ 内容寻址
    model.use_content_addressing = True
    fuzzy_recall_exp = evaluate_recall(model, fuzzy_queries, k=5)
    exact_em_exp = evaluate_em(model, exact_queries)

    fuzzy_improvement = fuzzy_recall_exp - fuzzy_recall_ctrl
    exact_degradation = exact_em_ctrl - exact_em_exp

    passed = fuzzy_improvement > 0.10 and exact_degradation < 0.05
    return ExperimentResult(
        name="A. 内容寻址", passed=passed,
        metrics={'fuzzy_recall_improvement': fuzzy_improvement,
                 'exact_em_degradation': exact_degradation},
        verdict="通过" if passed else "证伪: 内容寻址在64M无效, 需重新设计融合方式")
```

### 6.2 实验 B：拓扑缓存

```python
# experiment_b.py
def run_experiment_b(config, model, fsq):
    assoc_queries = load_dataset('test/association_queries_200.jsonl')

    model.use_topological_cache = False
    f1_ctrl = evaluate_f1(model, assoc_queries)
    latency_ctrl = measure_latency(model, assoc_queries)

    model.use_topological_cache = True
    f1_exp = evaluate_f1(model, assoc_queries)
    latency_exp = measure_latency(model, assoc_queries)

    f1_improvement = f1_exp - f1_ctrl
    latency_increase = (latency_exp - latency_ctrl) / latency_ctrl

    passed = f1_improvement > 0.05 and latency_increase < 0.20
    return ExperimentResult(
        name="B. 拓扑缓存", passed=passed,
        metrics={'f1_improvement': f1_improvement, 'latency_increase': latency_increase},
        verdict="通过" if passed else "证伪: 退回线性缓存")
```

### 6.3 实验 C：动态门控

```python
# experiment_c.py
def run_experiment_c(config, model):
    recite_data = load_dataset('test/recite_500.jsonl')
    reason_data = load_dataset('test/reason_500.jsonl')

    model.use_dynamic_gate = False
    recite_em_ctrl = evaluate_em(model, recite_data)
    reason_acc_ctrl = evaluate_accuracy(model, reason_data)

    model.use_dynamic_gate = True
    recite_em_exp = evaluate_em(model, recite_data)
    reason_acc_exp = evaluate_accuracy(model, reason_data)

    gate_recite = collect_gate_values(model, recite_data)
    gate_reason = collect_gate_values(model, reason_data)

    passed = (recite_em_exp >= recite_em_ctrl and
              reason_acc_exp > reason_acc_ctrl and
              np.mean(gate_recite) > 0.6 and
              np.mean(gate_reason) < 0.4)
    return ExperimentResult(
        name="C. 动态门控", passed=passed,
        metrics={'recite_em': recite_em_exp, 'reason_acc': reason_acc_exp,
                 'gate_recite_mean': np.mean(gate_recite),
                 'gate_reason_mean': np.mean(gate_reason)},
        verdict="通过" if passed else "证伪: 门控始终≈0.5或震荡, 固定混合比例")
```

### 6.4 实验 D：左脑校验

```python
# experiment_d.py
def run_experiment_d(config, model, verifier):
    test_data = load_dataset('test/with_distractors_500.jsonl')  # 20% 干扰

    model.use_verifier = False
    error_rate_ctrl = evaluate_error_rate(model, test_data)

    model.use_verifier = True
    filter_rate = evaluate_distractor_filter_rate(model, test_data)
    false_positive = evaluate_false_positive_rate(model, test_data)
    avg_retry = evaluate_avg_retry_count(model, test_data)

    passed = filter_rate > 0.50 and false_positive < 0.20
    return ExperimentResult(
        name="D. 左脑校验", passed=passed,
        metrics={'filter_rate': filter_rate, 'false_positive': false_positive,
                 'avg_retry': avg_retry},
        verdict="通过" if passed else "证伪: 简化为启发式校验")
```

### 6.5 实验 E：简化版 Latent MoE（新增）

```python
# experiment_e.py
def run_experiment_e(config, model):
    """
    对照组：所有 token 走全部 2 个 experts（密集并行）
    实验组：基于 FSQ 码字 hash 路由 (router_idx = codes % n_experts)
    数据：500 多域混合推理（涵盖数学、代码、对话、知识问答）
    目标：验证"语义路由可学性"——同一语义 cluster 的 token 是否被一致路由
    """
    multi_domain = load_dataset('test/multi_domain_500.jsonl')

    # 对照组
    model.use_moe = False
    dense_em = evaluate_per_domain_em(model, multi_domain)

    # 实验组
    model.use_moe = True
    routed_em = evaluate_per_domain_em(model, multi_domain)
    route_consistency = evaluate_route_consistency(model, multi_domain)
    expert_load = evaluate_expert_load(model, multi_domain)

    # 判定
    passed = (
        route_consistency > 0.70 and
        all(em >= baseline - 0.03 for baseline, em in zip(dense_em, routed_em)) and
        max(expert_load) < 0.90
    )

    return ExperimentResult(
        name="E. 简化版 Latent MoE", passed=passed,
        metrics={
            'route_consistency': route_consistency,
            'expert_load': expert_load,
            'per_domain_em_delta': [r - d for r, d in zip(routed_em, dense_em)]
        },
        verdict="通过" if passed else "证伪: 路由随机或单 expert 主导 >90%, 64M 简化版 MoE 不可行"
    )
```

> **注**：本实验是完整 Latent MoE 的 64M 最小验证，仅 2 experts + hash 路由；300M 阶段再扩展为基于右脑全局语义 $z$ 的连续路由与多 expert 协同。

### 6.6 实验执行编排

```python
# run_all_experiments.py
def run_all_experiments(config):
    """严格按照 A → B → C → D → E 顺序, 消融隔离"""

    # 前置检查: Phase 3.5 Rote 基线 (严格 复述任务, 与训练 EM>80% 不同)
    rote_baseline_em = evaluate_exact_match(model, rote_data)
    assert rote_baseline_em > 0.95, f"Rote 基线 EM = {rote_baseline_em:.1%} < 95%, 不可信"

    # 注: Phase 3.5 训练 EM>80% 是一般 SFT 验收; Rote EM>95% 是更严格的"查字典"基线
    # 只有 Rote 通过才能启动四大实验 (避免在机械对齐未稳定时误判机制)

    results = []
    for exp_fn in [run_experiment_a, run_experiment_b, run_experiment_c,
                   run_experiment_d, run_experiment_e]:
        result = exp_fn(config, model)
        results.append(result)
        if not result.passed:
            print(f"⚠️ {result.name} 未通过, 记录原因后继续下一实验")

    generate_report(results, output_path='reports/v3_experiment_report.md')

    n_passed = sum(1 for r in results if r.passed)
    if n_passed == 5:
        print("✅ 64M 全架构验证通过 → 迁移至 300M")
    else:
        print(f"⚠️ {n_passed}/5 实验通过, 记录《已证伪机制清单》, 300M 仅保留已验证机制")
    return results
```

---

## 7. 训练约束（64M 专属，不可违反）

```python
V3_TRAINING_CONSTRAINTS = {
    "fsq_freeze": True,              # Phase 3 后 FSQ 码本完全冻结
    "max_lr_right_brain": 1e-4,      # 右脑组件 LR 上限 (64M 对梯度噪声极敏感)
    "max_lr_backbone": 3e-4,         # 主干 LR 上限 (防破坏预训练语言能力)
    "max_steps_per_experiment": 2000,  # 每实验最大步数 (强制早停)
    "early_stop_patience": 1000,     # 1000 步无改善即终止
    "ablation_isolation": True,      # 每次仅开启一个组件, 严禁联合验证
    "shared_dataset": True,          # 四实验同一份混合数据集, 避免混淆变量
}
```

---

## 8. 验收决策树

```text
Phase 3.5 Rote 基线: EM > 95% ?
  ├── No → 停止，修复机械对齐
  └── Yes → 启动五大实验 (消融隔离，每次仅一个)

实验 A (内容寻址):
  ├── PASS (模糊 Recall ↑>20%, 精确 EM 降<5%) → 进入 B
  └── FAIL → 调整融合方式 / 放弃内容寻址，记录原因，进入 B

实验 B (拓扑缓存):
  ├── PASS (关联 F1 显著提升, 延迟<20%) → 进入 C
  └── FAIL → 退回线性缓存，记录原因，进入 C

实验 C (动态门控):
  ├── PASS (背诵 g→1>0.6, 推理 g→0<0.4) → 进入 D
  └── FAIL → 固定混合比例，记录原因，进入 D

实验 D (左脑校验):
  ├── PASS (过滤率>70%, 误杀<10%) → 进入 E
  └── FAIL → 简化为启发式校验，记录原因，进入 E

实验 E (简化版 Latent MoE):
  ├── PASS (路由一致性>70%, per-expert EM 不退化>3%) → ✅ 全架构验证通过
  └── FAIL → 64M 阶段放弃 MoE, 300M 再设计

最终产出:
  ├── verified_mechanisms.md    (已验证清单)
  ├── falsified_mechanisms.md   (已证伪清单，同等重要)
  └── migration_to_300m.md      (300M 迁移指南)
```

> **Phase 命名澄清**：
> - Phase 3 = FSQ 离散化
> - Phase 3.5 = 机械对齐（训练验收 EM > 80%）
> - 启动门槛用 **Rote 测试集 EM > 95%**（更严格的"查字典"基线，与训练 EM > 80% 角色不同）

---

## 9. v3.0 产出清单

```text
v3.0/
├── config_v3.py / sadko_v3_config.yaml
├── elf_lite.py               # 右脑 ELF-Lite (Flow Matching)
├── fsq.py                    # FSQ 量化器
├── router.py                 # 内容寻址 Router
├── dynamic_gate.py           # 动态门控网络
├── verifier.py               # 左脑校验模块
├── diffusion_align.py        # 扩散对齐训练
├── model_v3.py               # SADKONativeV3
├── train_phase1_elf.py       # Phase 1: ELF 预训练
├── train_phase2_fsq.py       # Phase 2: FSQ 离散化
├── train_phase3_5_align.py   # Phase 3.5: 机械对齐
├── train_phase4_diffusion.py # Phase 4: 扩散对齐
├── train_phase5_sft.py       # Phase 5: 全链路 SFT
├── experiment_a.py ~ experiment_e.py  # 实验 A 内容寻址 / B 拓扑缓存 / C 动态门控 / D 左脑校验 / E 简化 MoE
├── run_all_experiments.py    # 实验编排
├── checkpoints/
│   ├── elf_lite.pt
│   ├── fsq_quantizer.pt / fsq_codebook.json
│   ├── memory_aligned.pt
│   ├── phase4_soul.pt        # ← 灵魂注入后的权重
│   └── sadko_v3_final.pt     # Phase 5 SFT 后
└── reports/
    ├── v3_phase_report.md
    ├── v3_experiment_report.md
    ├── experiment_A_result.md ~ experiment_D_result.md
    ├── verified_mechanisms.md    # ✅ 已验证机制清单
    ├── falsified_mechanisms.md   # ❌ 已证伪机制清单 (同等重要)
    └── migration_to_300m.md     # 300M 迁移指南
```
