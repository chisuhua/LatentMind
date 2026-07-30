# SADKO-Native-64M 三阶段验证计划

> **一句话定位**：SADKO 白皮书的 64M 工程落地方案——基于 MiniMind3 的 v1.0 基座适配 → v2.0 记忆压缩 → v3.0 灵魂注入三阶段串行验证路线图
> **上游文档**：[whitepaper.md](./whitepaper.md)（架构设计与四大实验的理论依据）
> **最后更新**：2026-07-29

**一句话总结**：v1.0 证明"左脑能被改造"，v2.0 证明"记忆能被压缩和读取"，v3.0 证明"SADKO 的灵魂存在"。三个阶段严格串行，每阶段的验收报告是下一阶段的唯一启动许可证。

---

## 〇+、共享前置验证（Phase 0，~5 周，与 Logos 共用）

> 📌 **2026-07-29 v1.5 新增**：SADKO 64M 训练**也依赖** Phase 0 共享前置验证——这是双轨协调的关键发现。
>
> 详见双轨协调文档：[logos-sadko-64m-coordination.md §2](../logos/sadko-64m-coordination.md#2-phase-0共享前置验证-5-周)
>
> ⚠️ **关键提示**：Phase 0 不通过，禁止启动 SADKO 64M v1.0 / v2.0 / v3.0 的任何架构改造实验。

### 为什么 SADKO 也需要 Phase 0

SADKO 文档中已有"〇、MiniMind3 64M Dense 基座调研"——这是**架构调研**而非**训练验证**。我们犯了混淆：调研只是"看"，没"训"。

Phase 0 补充了关键的训练验证，**两者并存**：
- 〇（调研）：MiniMind3 架构是什么、参数多少、约束是什么（**已完成**）
- 〇+（Phase 0 训练验证）：MiniMind3 能从零训练到合理 PPL 吗、感知层可训练吗、评估框架 OK 吗（**待完成**）

### Phase 0 必做项（来自协调文档）

| # | 任务 | 周期 | 必需性 | SADKO 影响 |
|---|------|------|:---:|----------|
| P.0.1 | MiniMind3 64M Dense 从零训练基线 | 1 周 | 🔴 必须 | 决定基座可行性 |
| P.0.2 | 3 层 CNN 感知层映射 | 1 周 | 🔴 必须 | SADKO 多模态基础 |
| P.0.3 | 数据 pipeline + tokenizer | 3 天 | 🟠 必需 | 训练效率 |
| P.0.4 | 训练基础设施 | 3 天 | 🟠 必需 | 训练稳定性 |
| P.0.5 | 评估框架（统一可比） | 1 周 | 🔴 必须 | 决定跨路线评估有效性 |
| P.0.6 | 端侧并行化基础设施 | 1 周 | 🟢 Logos 专用 | SADKO 不直接依赖 |

**Phase 0 失败后果**：
- P.0.1 失败 → SADKO / Logos 都需重新选择基座 → **延后 +1 月**
- P.0.2 失败 → 多模态基础不可行 → **重做 SADKO 多模态部分**
- P.0.5 失败 → SADKO / Logos 评估结果不可比 → **双轨决策失效**

### Phase 0 与 SADKO 三阶段的关系

```
Phase 0（共享，~5 周）
├─ P.0.1 - P.0.6 全部通过
└─ SADKO 64M 启动（L1/L2/L3 三阶段）
   ├─ v1.0：Split-GQA + Dual-Path FFN（左脑改造）
   ├─ v2.0：MemPool + 192 维压缩读取
   └─ v3.0：ELF-Lite + FSQ + 扩散对齐 + 四大实验

Phase 0 是 SADKO 64M v1.0 / v2.0 / v3.0 的**前置许可证**。
```

### SADKO 64M 的特殊需求

**P.0.2 CNN 感知层**对 SADKO 多模态至关重要：
- SADKO 的 ELF 流形压缩需要 3 层 CNN 作为感知入口
- 若 P.0.2 失败，**多模态**作为 SADKO 优势会大打折扣

**P.0.5 评估框架**对 SADKO 决策关键：
- "检索 Recall@64"是 SADKO 关键评估指标
- 必须用统一脚本评估 SADKO 与 Logos 对比

---

## 〇++、交叉验证（Phase 2，Month +5 末 ~2 周）

> 📌 **2026-07-29 v1.5 新增**：SADKO 64M 训练完成后，需在统一评估框架下与 Logos 对比。

详见 [logos-sadko-64m-coordination.md §4](../logos/sadko-64m-coordination.md#4-phase-2交叉验证-2-周month-5-末)。

**六类对比任务与 SADKO 预期表现**：

| 类别 | SADKO 预期表现 |
|------|-------------|
| **推理** (GSM8K mini) | ⚠️ 中等（非 SADKO 优势域）|
| **代码** (MultiPL-E mini) | ⚠️ 中等 |
| **决策** (nuScenes multi-path) | ✅ **强**：MemPool + 多轨迹 |
| **记忆** (Recall@64) | ✅ **强**：MemPool + FSQ 码本 |
| **端侧** (单 token 延迟) | ⚠️ 中等（依赖融合接口）|
| **多模态** (CIFAR-10) | ✅ **强**：双向 + Flow Matching |

**决策依据**（详见协调文档 §5）：
- Logos 推理强 + SADKO 记忆强 + 融合可行 → **双轨并行**（最可能）
- SADKO 在多任务胜出 → **SADKO 升格主线**
- 双轨都失败 → **退回通用 Transformer**

**禁止各路线私自评估**——必须用 P.0.5 统一脚本。

---

## 〇、MiniMind3 64M Dense 基座调研

### 原始架构

```text
MiniMind3 Dense 64M:
├── Tokenizer: BPE + ByteLevel, vocab=6400
├── Embedding: dim=768
├── Transformer Blocks × 8:
│   ├── RMSNorm
│   ├── Multi-Head Attention (GQA, 16 heads, 8 KV heads, head_dim=48)
│   ├── RMSNorm
│   └── FFN (SwiGLU, intermediate=2048)
├── LM Head: 768 → 6400
└── 总参数: ~64M
```

### 原始训练流程

```text
Pretrain (1.2GB 通用语料, ~2h/单卡3090)
  → SFT (指令微调)
    → DPO (偏好对齐)
      → LoRA (领域适配)
```

### 关键约束

| 维度 | 值 | 说明 |
| :--- | :--- | :--- |
| hidden_size | 768 | 所有改造必须兼容此维度 |
| num_layers | 8 | 锚点层选择空间有限 |
| num_kv_heads | 8 | Split-GQA 最大 4:4 分割 |
| head_dim | 48 | 非标准 64，RoPE/Attention 需适配 |
| vocab_size | 6400 | 小词表，Memory Token 需特殊处理 |
| 训练硬件 | 单卡 3090 / 4060 8GB | 所有阶段必须在此约束内 |

---

## 一、三阶段总览

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  v1.0 (基座适配)          v2.0 (记忆压缩)          v3.0 (灵魂注入)      │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐       │
│  │ Split-GQA    │       │ Shared       │       │ ELF-Lite     │       │
│  │ 异构RoPE     │──────▶│ MemPool      │──────▶│ (Flow Match) │       │
│  │ Dual-Path FFN│       │ MLP压缩      │       │ FSQ离散化    │       │
│  │ CA骨架(零初始化)│     │ 192维读取    │       │ 扩散对齐     │       │
│  └──────────────┘       │ Position Bias│       │ 动态门控     │       │
│                         └──────────────┘       │ 左脑校验     │       │
│  验证: 左脑改造         验证: 压缩-读取         └──────────────┘       │
│  不破坏语言能力         机制物理可行                                   │
│                                                验证: SADKO四大实验     │
│                                                完整机制正确性          │
│                                                                         │
│  周期: 1-2周            周期: 2-3周             周期: 3-4周            │
│  风险: 低               风险: 中                 风险: 中高             │
└─────────────────────────────────────────────────────────────────────────┘
```

| 阶段 | 核心命题 | 一句话目标 | 失败判定 |
| :--- | :--- | :--- | :--- |
| **v1.0** | 左脑能否被改造？ | 改造后 PPL 不退化，Static/Dynamic 自然分流 | PPL 退化 >0.2 或熵无分离 |
| **v2.0** | 压缩-读取是否可行？ | 192 维潜空间能承载远程记忆并正确召回 | 检索 Recall@64 < 50% |
| **v3.0** | SADKO 灵魂是否存在？ | 四大实验通过，零样本联想涌现 | 任一核心实验被证伪 |

---

## 二、v1.0：基座适配层（左脑改造）

### 2.1 目标

> **在不破坏 MiniMind3 原始语言能力的前提下，完成 SADKO 左脑的结构性改造，为后续右脑接入预留所有接口。**

### 2.2 架构设计

```text
SADKO-Native-64M v1.0:
├── [保留] Tokenizer: vocab=6400
├── [保留] Embedding: dim=768
├── [新增] Memory Embedding: 768→768 (零初始化, 暂不激活)
├── Transformer Blocks × 8:
│   ├── RMSNorm
│   ├── [改造] SADKOSplitGQA:
│   │   ├── 4 Static KV Heads (RoPE base=500k, 长程)
│   │   └── 4 Dynamic KV Heads (RoPE base=10k, 短程)
│   ├── [新增] Cross-Attention 骨架 (gate=0, 不参与计算)
│   ├── RMSNorm
│   └── [改造] 锚点层(2/4/6): SADKODualPathFFN
│       ├── memory_ffn (SwiGLU)
│       ├── reason_ffn (SwiGLU)
│       └── gate_proj (零初始化, 默认0.5混合)
├── [保留] LM Head: 768 → 6400
└── 总新增参数: ~15.5M (Dual-Path FFN × 3层 ~14.2M + CA骨架 ~0.89M + 门控 ~0.45M + Memory Emb)
    注: v1.0 总参数 = ~80M (基座 64M + 改造 15.5M), 不是 65.2M
```

### 2.3 关键设计决策

| 决策 | 选择 | 理由 |
| :--- | :--- | :--- |
| 锚点层 | L2, L4, L6（偶数层） | 与白皮书对齐；均匀分布确保记忆信号逐层渗透 |
| Split-GQA 比例 | 4:4（1:1） | 64M 仅 8 KV heads，1:1 是最小可行分割 |
| 异构 RoPE | **Static=500k（低频长程）, Dynamic=10k（高频短程）** | 对齐 NTK 长上下文外推实践：更大 base = 更慢旋转 = 长程平坦注意力 |
| Cross-Attention gate | 初始化为 0（sigmoid(-∞)） | 确保 v1.0 前向传播与原始 MiniMind3 完全一致 |
| Dual-Path gate | 初始化为 0.5 | 无先验偏好，让训练自然学习分流 |
| 层类型 | **全部标准 GQA**（不引入 SWA/GLA） | 控制变量，消融隔离原则 |

### 2.4 训练方案

```text
数据: MiniMind3 原始 pretrain_t2t_mini.jsonl (1.2GB)
目标: 标准 Next Token Prediction
策略: 从 MiniMind3 预训练权重初始化，继续训练 2 epochs
LR: 3e-4 (主干), 1e-4 (新增模块)
硬件: 单卡 3090, ~3-4h
```

### 2.5 验收标准与实验

| 实验 | 指标 | 通过标准 | 失败处理 |
| :--- | :--- | :--- | :--- |
| **PPL 保持** | C4 验证集 PPL | ≤ 2.75（原始 ~2.7） | 检查初始化/梯度问题 |
| **熵分离** | Static vs Dynamic KV Heads 注意力熵 | **Static 熵 > Dynamic 熵**（Static 长程平坦，Dynamic 短程锐利）且差值 >0.3 | 检查 RoPE base 是否正确加载（500k/10k） |
| **速度开销** | 训练 tokens/s | 下降 ≤ 15% | 检查 Split-GQA 实现效率 |
| **门控行为** | Dual-Path gate 值分布 | 非全 0.5，出现自然分化 | 延长训练或调整 gate 初始化 |
| **消融：去 RoPE** | 移除异构 RoPE 后 PPL | 退化 > 0.1（证明 RoPE 有效） | — |

### 2.6 产出物

```text
├── sadko_native_v1/
│   ├── model.py          # SADKOSplitGQA + SADKODualPathFFN
│   ├── config.yaml       # v1.0 配置
│   ├── train_v1.py       # 训练脚本
│   └── eval_v1.py        # PPL + 熵分析
├── checkpoints/
│   └── sadko_v1_base.pt  # 初始化权重（v2.0/v3.0 的起点）
└── reports/
    └── v1_validation.md  # 验收报告
```

---

## 三、v2.0：记忆压缩层（潜空间读写验证）

### 3.1 目标

> **验证"将远程 KV 压缩到 192 维潜空间并通过 Cross-Attention 正确读取"这一机制的物理可行性。此阶段用 MLP 压缩器代替 Flow Matching，隔离验证"压缩-存储-检索-读取"管线。**

### 3.2 架构设计（在 v1.0 基础上增量）

```text
SADKO-Native-64M v2.0 (在 v1.0 基础上新增):
├── [激活] Cross-Attention (L3, L5):
│   ├── gate: 标量可学习参数 (初始化 -2.0, sigmoid≈0.12)
│   ├── Q: 来自当前层 hidden_state (768→192 投影)
│   ├── K/V: 来自 Shared MemPool (192维)
│   └── 层专属投影: W_k_L3, W_v_L3, W_k_L5, W_v_L5 (192→192)
│
├── [新增] Shared MemPool:
│   ├── Compressor: MLP (768→192) × 2 (K/V各一个)
│   ├── Indexer: MLP (192→64) + L2_normalize
│   ├── Position Embedding: 相对位置分桶 (16 buckets)
│   └── Store: mem_k[N×192], mem_v[N×192], index[N×64]
│
├── [新增] 压缩触发器:
│   ├── hot_window = 4096 (锁定)
│   ├── chunk_size = 64
│   └── 触发: token 离开 hot_window 时，拼接 L3+L5 KV → 压缩
│
└── 总新增参数 (vs v1.0): ~2.3M
    ├── Compressor: 0.771M
    ├── Read Projections (L3+L5): 0.295M
    ├── Position Embedding: 0.01M
    └── Cross-Attention Q/O proj: ~1.2M
```

### 3.3 关键设计决策

| 决策 | 选择 | 理由 |
| :--- | :--- | :--- |
| 压缩器 | MLP（非 Flow Matching） | 隔离变量：先验证"压缩-读取"管线，再验证"压缩质量" |
| 潜空间维度 | 192 | 768/4，压缩比 4:1，64M 容量下的平衡点 |
| 压缩数据源 | L3+L5 KV **拼接**（64×768 匹配 kv_input_dim） | L3/L5 是 Cross-Attention 读取层，其 KV 含跨段全局信号；拼接保留完整语义，避免均值池化的维度损失 |
| 检索方式 | Cosine 相似度 + Position Bias | 语义 + 顺序双信号 |
| Top-K | 64 chunks | 64×64=4096 tokens 远程上下文 |
| 门控 | 标量 sigmoid（可学习） | 最简门控，验证"是否需要动态切换" |
| Cross-Attention 空间 | 严格 192 维 | 拒绝膨胀到 768，保持低维正交 |
| max_position_embeddings | v2.0 起扩到 **16384**（v1.0 的 4096 不支持远程记忆实验） | NTK-aware 动态缩放 effective base = 40000；否则 8K+ 长上下文训练无合法 position_id |

### 3.4 训练方案

```text
阶段 2a: 长上下文预训练
  数据: 8K-16K 长文本 (书籍/论文, ~200MB)
  目标: 标准 LM Loss + 远程记忆辅助 Loss
  策略: 从 v1.0 权重初始化
  LR: 1e-4 (主干), 5e-5 (MemPool 模块)
  硬件: 单卡 3090, ~6-8h

阶段 2b: 记忆检索微调
  数据: 构造 "远程事实召回" QA 对 (~50K 条)
  目标: 给定问题，从 MemPool 中检索正确 chunk 并生成答案
  指标: Recall@64, EM, F1
```

### 3.5 验收标准与实验

| 实验 | 指标 | 通过标准 | 失败处理 |
| :--- | :--- | :--- | :--- |
| **压缩重构** | 压缩后 KV 重构 MSE | 重构后 PPL 增加 < 10% | 增大潜空间维度或压缩器容量 |
| **检索准确率** | Recall@64 (精确查询) | > 70% | 检查 Indexer 训练 |
| **模糊检索** | Recall@64 (语义模糊查询) | > 40% | 引入内容寻址（v3.0 前置） |
| **长程 QA** | 8K+ 文本 QA 准确率 | > 纯 hot_window 基线 +15% | 检查 Position Bias 有效性 |
| **门控行为** | gate 值在不同任务下的分布 | 背诵时 gate↑，推理时 gate↓ | 若始终≈0.5，v3.0 需重新设计门控 |
| **显存** | 16K 上下文峰值显存 | < 原始全量 KV 的 60% | 验证压缩收益 |

### 3.6 产出物

```text
├── sadko_native_v2/
│   ├── mempool.py        # Shared MemPool + Compressor + Indexer
│   ├── cross_attn.py     # 层专属 Cross-Attention
│   ├── config_v2.yaml
│   ├── train_v2.py
│   └── eval_v2.py
├── checkpoints/
│   └── sadko_v2_mem.pt
└── reports/
    ├── v2_validation.md
    └── v2_falsified.md   # 已证伪/已验证机制清单
```

---

## 四、v3.0：灵魂注入层（SADKO 完整机制验证）

### 4.1 目标

> **在 v2.0 验证"压缩-读取管线可行"的基础上，引入 SADKO 白皮书的完整右脑（ELF + FSQ）和扩散对齐桥梁，执行五大核心实验（含简化版 Latent MoE），产出《已验证/已证伪机制清单》。**

### 4.2 架构设计（在 v2.0 基础上替换/新增）

```text
SADKO-Native-64M v3.0 (在 v2.0 基础上改造):
│
├── [替换] Compressor → 右脑 ELF-Lite (~5M):
│   ├── 2层 Transformer Encoder (双向注意力, 无因果Mask)
│   ├── Flow Matching Head (速度场预测)
│   ├── 输入: 完整 chunk 的 L3+L5 KV (64×768)
│   ├── 输出: 连续潜向量 z (64×192) → 池化为 chunk 级 z (192)
│   └── 训练: Flow Matching Loss + 重构 Loss
│
├── [新增] FSQ 量化器:
│   ├── codebook_size = 256 (64M 验证用)
│   ├── levels = [8,8,8,8] (4维标量量化)
│   ├── 输入: z_continuous (192)
│   ├── 输出: z_quantized (192) + code_index (int)
│   └── 码本冻结策略: Phase 3 后冻结，验证阶段不更新
│
├── [替换] Indexer → 内容寻址 Router:
│   ├── 固定码本嵌入 (256×64, 来自 FSQ)
│   ├── Query: z_block (192) → 投影到 64 维
│   ├── 相似度: Cosine(query, codebook_embeddings)
│   └── + Position Bias (保留 v2.0)
│
├── [替换] 标量门控 → 动态门控网络:
│   ├── 输入: hidden_state (768) + semantic_tag (16)
│   ├── 网络: Linear(784→192) → SiLU → Linear(192→1) → Sigmoid
│   └── 输出: gate ∈ [0,1] (0=纯右脑, 1=纯左脑)
│
├── [新增] 左脑校验模块:
│   ├── 置信度分数: LM Head logits 的 entropy
│   ├── 阈值 τ: 可学习 (初始化 0.7)
│   └── 触发: entropy > τ → 二次检索 MemPool
│
├── [新增] 扩散对齐训练接口:
│   ├── Teacher: ELF-Lite (冻结)
│   ├── Student: 左脑 code_embed 层 (FSQ码字→KV嵌入, 256×768, 与 v1.0 semantic_tag 分离)
│   ├── Loss: 0.3×KL + 0.7×CE
│   └── 步数: 2000-3000 steps
│
├── [新增] 实验 E 简化版 MoE:
│   ├── 2 experts × 锚点层 3 (L2/L4/L6) = 6 SwiGLU expert FFN
│   ├── 路由: router_idx = codes % n_experts (FSQ 码字 hash)
│   └── 训练: 与 Dense 基线对比 per-domain EM 保持
│
└── 总参数变化 (vs v2.0):
    ├── 移除 MLP Compressor: -0.771M
    ├── 新增 ELF-Lite: +5.0M
    ├── 新增 FSQ: +0.05M
    ├── 新增 code_embed: +0.196M
    ├── 新增动态门控: +0.15M
    ├── 新增左脑校验: +0.01M
    ├── 新增简化版 MoE: +9.4M
    └── 净增: ~14.0M (总模型 ~93M, 含 v1.0 累计)
    注：v1.0 已含 Dual-Path FFN 改造 ~15.5M, 总参数 ~80M (基座 64M + 改造 15.5M)
```

### 4.3 训练方案（Phase 0-5 完整流程）

```text
Phase 1: 右脑 ELF 预训练 (独立, ~4h)
  数据: 知识库文本 → MiniMind Prefill → KV Cache
  目标: Flow Matching Loss + 重构 Loss
  验收: 重构 KV 后 PPL 增加 < 5%

Phase 2: FSQ 离散化 (~2h)
  数据: ELF 编码后的连续潜向量
  目标: 量化误差最小化
  验收: 码字插值语义合理率 ≥ 60%

Phase 3: 机械对齐 (~3h)
  冻结: 左脑底层 6 层
  训练: Memory Embedding + Cross-Attention
  Loss: MSE(左脑生成KV, 右脑解码KV) + CE
  验收: 给定 FSQ 码字，左脑复述 EM > 80%

Phase 4: 扩散对齐 (~2h) ← 注入灵魂
  Teacher: ELF-Lite (冻结)
  Student: 左脑顶层 + Memory Embedding
  Loss: 0.3×KL + 0.7×CE
  步数: 2000-3000
  验收: 语义理解提升 > 8%, 零样本联想涌现

Phase 5: 全链路 SFT (~4h)
  解冻: 全参数
  数据: 带检索的 QA 对 + 混合任务
  验收: SADKO-MiniMind > 纯 MiniMind SFT
```

### 4.4 五大核心实验（白皮书要求）

| 实验 | 对照组 | 实验组 | 通过标准 | 证伪标准 |
| :--- | :--- | :--- | :--- | :--- |
| **A. 内容寻址** | 纯符号路由（地址标签） | FSQ 码本 Cosine 相似度 | 模糊查询 Recall 提升 >20% | 提升 <10% 或精确查询下降 >5% |
| **B. 拓扑缓存** | 线性 Top-K 独立检索 | FSQ 码字排序 + 滑动窗口 | 关联查询 F1 显著提升 | F1 无提升或延迟增加 >20% |
| **C. 动态门控** | 固定 0.5:0.5 混合 | 可学习门控网络 | 背诵 g→1 (>0.6), 推理 g→0 (<0.4) | 门控始终≈0.5 或震荡 |
| **D. 左脑校验** | 无校验直接输出 | 置信度阈值过滤 | 干扰项过滤率 >70%, 误杀 <10% | 过滤率 <50% 或误杀 >20% |
| **E. 简化版 Latent MoE** | Dense 全 expert 并行 | FSQ 码字 hash 路由 (2 experts) | 路由一致性 >70% 且 per-expert EM 不退化 >3% | 路由随机或单 expert 主导 >90% |

### 4.5 训练约束（64M 专属）

| 约束 | 值 | 理由 |
| :--- | :--- | :--- |
| FSQ 码本 | Phase 3 后**完全冻结** | 避免码本更新掩盖架构问题 |
| 右脑组件 LR | ≤ 1e-4 | 64M 对梯度噪声极度敏感 |
| 主干 LR | ≤ 3e-4 | 防止破坏预训练语言能力 |
| 每实验最大步数 | 2000 steps | 强制早停，1000 步无改善即终止 |
| 消融隔离 | 每次仅开启一个组件 | 严禁联合验证 |
| 数据复用 | 四实验同一份混合数据集 | 避免混淆变量 |

### 4.6 验收标准与决策树

```text
Phase 3.5 Rote 基线: EM > 95% ?        ← Rote 是"查字典"严格复述, 与 Phase 3.5 训练 EM>80% 不同
  ├── No → 停止，修复机械对齐
  └── Yes → 启动五大实验 (消融隔离, 每次仅一个)

实验 A (内容寻址): 模糊 Recall 提升 >20%, 精确 EM 下降 <5% 通过 / <10% 证伪
  ├── 通过 → 进入 B
  └── 失败 → 调整融合方式 / 放弃内容寻址，记录原因，进入 B

实验 B (拓扑缓存): 关联 F1 显著提升, 延迟增加 <20%
  ├── 通过 → 进入 C
  └── 失败 → 退回线性缓存，记录原因，进入 C

实验 C (动态门控): 背诵 g>0.6, 推理 g<0.4, 且相对固定 0.5 不退化
  ├── 通过 → 进入 D
  └── 失败 → 固定混合比例，记录原因，进入 D

实验 D (左脑校验): 过滤率 >70%, 误杀 <10% 通过 / <50% 或误杀 >20% 证伪
  ├── 通过 → 进入 E
  └── 失败 → 简化为启发式校验，记录原因，进入 E

实验 E (简化版 Latent MoE): 路由一致性 >70%, per-expert EM 不退化 >3%, 无单 expert 主导 >90%
  ├── 通过 → ✅ 64M 全架构验证通过 → 迁移至 300M
  └── 失败 → 64M 阶段放弃 MoE, 300M 再设计
```

### 4.7 产出物

```text
├── sadko_native_v3/
│   ├── elf_lite.py       # 右脑 ELF-Lite (Flow Matching)
│   ├── fsq.py            # FSQ 量化器
│   ├── router.py         # 内容寻址 Router
│   ├── gate.py           # 动态门控网络
│   ├── verifier.py       # 左脑校验模块
│   ├── diffusion_align.py # 扩散对齐训练脚本
│   ├── config_v3.yaml
│   ├── train_phase1_5.py # Phase 1-5 训练流水线
│   └── eval_experiments.py # 四大实验评估
├── checkpoints/
│   ├── elf_lite.pt
│   ├── fsq_codebook.json
│   ├── sadko_v3_aligned.pt
│   └── sadko_v3_sft.pt
└── reports/
    ├── v3_validation.md
    ├── verified_mechanisms.md    # 已验证机制清单
    ├── falsified_mechanisms.md   # 已证伪机制清单 ← 同等重要
    └── migration_to_300m.md     # 300M 迁移指南
```

---

## 五、三阶段依赖关系与风险控制

### 5.1 严格依赖链

```text
v1.0 通过 (PPL保持 + 熵分离)
  │
  ├── 失败 → 修复 Split-GQA/RoPE 实现，不进入 v2.0
  │
  ▼
v2.0 通过 (压缩-读取管线可行)
  │
  ├── 失败 → 调整潜空间维度/压缩器，不进入 v3.0
  │
  ▼
v3.0 四大实验
  │
  ├── 全部通过 → 迁移至 300M
  ├── 部分通过 → 记录清单，300M 仅保留已验证机制
  └── 全部失败 → 架构重新设计
```

### 5.2 每阶段的"不可逾越"原则

| 原则 | 说明 |
| :--- | :--- |
| **v1.0 不接入任何外部记忆** | Cross-Attention gate=0，确保纯左脑改造无副作用 |
| **v2.0 不引入 Flow Matching** | 用 MLP 隔离"压缩质量"与"读取管线"两个变量 |
| **v3.0 不改变 v2.0 的读取管线** | 仅替换压缩器（MLP→ELF）和检索器（Cosine→FSQ Router） |
| **每阶段仅改变一个核心变量** | 消融隔离，严禁多变量联合验证 |

### 5.3 时间线

```text
Week 1-2:  v1.0 (基座适配 + PPL验证)
Week 3-5:  v2.0 (MemPool + 压缩读取验证)
Week 6-9:  v3.0 (ELF + FSQ + 扩散对齐 + 四大实验)
Week 10:   产出《验证报告》+ 300M 迁移决策
```

---

## 六、v2.0 → v3.0 的具体替换清单

为确保 v2.0 的代码资产在 v3.0 中最大化复用，明确以下替换边界：

| 模块 | v2.0 实现 | v3.0 替换为 | 复用部分 |
| :--- | :--- | :--- | :--- |
| 压缩器 | MLP (768→192) | ELF-Lite (双向 Transformer + Flow Matching) | 输入/输出接口不变 (768→192) |
| 存储 | 连续 mem_k/mem_v (192) | FSQ 码字索引 + 连续潜向量双存储 | Store 结构扩展，不重建 |
| 检索 | Cosine + Position Bias | FSQ 码本嵌入 Cosine + Position Bias | Position Bias 完全复用 |
| 读取 | 层专属 192→192 投影 + CA | **完全保留** | 100% 复用 |
| 门控 | 标量 sigmoid | 动态门控网络 (MLP) | 接口不变，内部替换 |
| 校验 | 无 | 置信度阈值过滤 | 纯新增 |
| 训练 | 标准 LM Loss | Phase 1-5 流水线 | 数据管线复用 |

---

## 七、立即执行清单

| # | 任务 | 阶段 | 时间 | 产出 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Fork MiniMind3，跑通原始 Pretrain + SFT | 前置 | Day 1-2 | 环境确认 |
| 2 | 实现 `SADKOSplitGQA` + `SADKODualPathFFN` | v1.0 | Day 3-5 | model_v1.py |
| 3 | 从 MiniMind3 权重初始化，训练 2 epochs | v1.0 | Day 5-6 | sadko_v1_base.pt |
| 4 | PPL + 熵分离验收 | v1.0 | Day 7 | v1_validation.md |
| 5 | 实现 Shared MemPool + Compressor + Indexer | v2.0 | Day 8-12 | mempool.py |
| 6 | 长上下文训练 + 记忆检索微调 | v2.0 | Day 12-18 | sadko_v2_mem.pt |
| 7 | 压缩/检索/QA 验收 | v2.0 | Day 19-20 | v2_validation.md |
| 8 | 实现 ELF-Lite + FSQ | v3.0 | Day 21-25 | elf_lite.py, fsq.py |
| 9 | Phase 1-4 训练（ELF→FSQ→机械对齐→扩散对齐） | v3.0 | Day 25-35 | sadko_v3_aligned.pt |
| 10 | 四大实验（A→B→C→D，消融隔离） | v3.0 | Day 35-45 | 实验报告 |
| 11 | 产出《验证/证伪清单》+ 300M 迁移决策 | v3.0 | Day 46-50 | 最终报告 |
