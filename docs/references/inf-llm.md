# InfLLM 参考资料

> **论文**：*InfLLM: Unveiling the Intrinsic Capacity of LLMs for Understanding Extremely Long Sequences with Training-Free Memory*
> **作者**：Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Song Han, Maosong Sun
> **机构**：THU（清华）+ MIT + Meta AI
> **arXiv**：2402.04617v2（2024-02-07）
> **会议**：NeurIPS 2024
> **代码**：[github.com/thunlp/InfLLM](https://github.com/thunlp/InfLLM)
> **性质**：**Loop B（inter-decode 记忆压缩）当前 SOTA**——结合 sliding window + 块级 memory lookup，可处理 1M+ tokens 的训练无关长上下文

---

## 1. 一句话定位

**将 StreamingLLM 的滑动窗口与块级 memory 检索结合——把远距离上下文按 block 切分，每个 block 选 top-k 代表 token 作为"单元表示"，每步检索 top-k 相关 block 进 attention**。**完全训练无关**（与 StreamingLLM 一样），**但支持真正的 1,024K tokens 远距离依赖捕获**。在 Mistral-7B / Llama-3-8B 上验证有效。

> 📌 **归属**：详见 [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) — 属于 Loop B 维度（inter-decode 记忆压缩），是**当前最通用**的方案（兼顾 StreamingLLM 效率和 Memorizing Transformers 远距离能力）

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | https://arxiv.org/abs/2402.04617 |
| GitHub | https://github.com/thunlp/InfLLM |
| NeurIPS 2024 PDF | https://proceedings.neurips.cc/paper_files/paper/2024/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf |
| OpenReview | https://openreview.net/forum?id=bTHFrqhASY |

---

## 3. 核心创新

### 3.1 块级 Context Memory（核心机制）

把远距离 KV 按**块**切分，每块选 top-k 代表 token：

```
┌─────────────────────────────────────┐
│  Local Window (n_local=4096)         │  ← dense attention
│  (most recent tokens)                │
└─────────────────────────────────────┘
            + Attention
┌─────────────────────────────────────┐
│  Context Memory (块级)                │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐    │
│  │M_1  │ │M_2  │ │M_3  │ │M_4  │    │  ← 每块 size=128
│  │rep  │ │rep  │ │rep  │ │rep  │    │  ← rep=4 代表 token
│  └─────┘ └─────┘ └─────┘ └─────┘    │
└─────────────────────────────────────┘
            ↑
        每步检索 top-k=16 相关 block
```

**关键参数**（参考实现）：
- `n_init = 128`：初始 token（attention sinks）
- `n_local = 4096`：局部滑动窗口
- `topk = 16`：每步检索的 block 数
- `repr_topk = 4`：每 block 的代表 token 数
- `max_cached_block = 32`：GPU 内存最大 block 数
- `exc_block_size = 512`：每批 query 块大小
- `cache_strategy = lru`：LRU 替换

### 3.2 三层架构

1. **Local Window**：dense attention 完整覆盖最近 4K tokens
2. **Block-level Context Memory**：远距离 token 按 128 块切分，每块选 4 个"代表 token"（按历史 attention 分数）作为单元表示
3. **Memory Lookup**：每步 query 与所有 block 的"代表 token"算相似度，选 top-k block 做 attention

**优势**：
- **有效**：block 内部语义连贯，比 token-level 更准
- **高效**：block-level 计算量远小于 token-level
- **GPU 友好**：连续内存访问，无需 per-token 计算

### 3.3 关键技术点

#### 代表 token 选择
```python
# m-th block 的第 i 个 token 的代表分数
score(m, i) = sum over past queries of attention(m, i)
# 选 top-4 token 作为 block 的"代表"
```

#### 位置编码策略
- 局部 window 内用正常位置
- 远距离 memory 的位置全部设为 lL（与局部窗口末尾同一位置）—— 避免 OOD

#### 缓存管理
- 大多数 memory block 放在 **CPU 内存**
- 仅 top-use 的 block 保留在 **GPU 显存**
- 显著降低 GPU 显存占用

---

## 4. 关键数字

| 指标 | 值 | 备注 |
|------|:---:|------|
| 最大序列长度 | **1,024K tokens** | 论文验证 |
| 模型族 | Mistral-7B-inst-v0.2, Llama-3-8B-Instruct | 训练在 32K / 8K |
| 训练开销 | **0** | 完全训练无关 |
| 局部窗口 | 4096 tokens | 训练时是 32K / 8K |
| Memory 单元大小 | 128 tokens | 块大小 |
| 单元代表数 | 4 | repr_topk |
| 检索 block 数 | 16-32 | topk |
| GPU 缓存 blocks | 32 (Llama-3), 96 (Mistral) | LRU 替换 |
| 评估基准 | ∞-Bench, LongBench | 显著优于 sliding window |
| 与 continual-training 基线对比 | **相当或更好** | 无需重新训练 |

---

## 5. 关键发现

### 5.1 训练无关即可达到 continual-training 性能

> "Without any training, InfLLM enables LLMs that are pre-trained on the sequences consisting of a few thousand tokens to achieve comparable performance with competitive baselines that continually train these LLMs on long sequences."

**意义**：**长上下文能力部分已存在于 LLM 内部**（通过 attention 模式可挖掘），无需重新训练——这是**信息几何层面**的发现。

### 5.2 注意力稀疏性的关键证据

> "The attention score matrices of LLMs are sparse, and we can generate the same outputs with only a small portion of key-value vectors preserved."

**意义**：远距离 token 中只有少量"重要"——这与 Memorizing Transformers 的发现一致，但 InfLLM 通过 block-level 设计避免了 kNN 检索的开销。

### 5.3 内存与计算的解耦

- GPU 显存：固定（仅保留 top-use block）
- CPU 内存：可扩展（最大 1M tokens 测试）
- 计算：每步只增加 top-k block attention（远小于全 dense）

---

## 6. 与本项目架构的对照

| 维度 | InfLLM | 本项目 Hippo | 关键差异 |
|------|--------|-------------|----------|
| **存储单元** | KV block（连续 128 tokens）| FSQ 256 码本 | Hippo 显式离散化 |
| **压缩** | ❌ 无压缩（仅 block 化）| ✅ FM 压缩 | Hippo 信息密度更高 |
| **检索粒度** | 块级（4 代表 token / 128 块）| 码字级（256 码字）| 不同粒度 |
| **代表 token 选择** | 按历史 attention 分数 | FSQ 量化 | 不同表示 |
| **可微** | 不适用（训练无关）| ✅ 端到端 | 范式不同 |
| **端侧友好** | ⚠️ CPU/GPU 协同 | ⚠️ | 都需评估 |
| **目标** | 长上下文（1M+ tokens）| 知识召回 + 多模态 | Hippo 更丰富 |
| **架构归属** | Loop B（inter-decode）| Loop B（inter-decode）| ✅ 同一类别 |

**关键启示**：
- ✅ **Block-level memory** 思想可借鉴到 Hippo 内部——把 FSQ 256 码本视为"256 个 block 的代表"
- ✅ **代表 token 选择** 的"按历史 attention" 思想可作为 Hippo Router 优化的候选方案
- ✅ **CPU/GPU 协同** 思想对端侧部署关键——Hippo 64M 阶段可先不评估
- ⚠️ InfLLM **不解决端到端训练**问题——这是 Hippo 的优势（可与 SADKO 联合训练）

---

## 7. 局限性

| 限制 | 严重性 | 说明 |
|------|:---:|------|
| Block 大小固定（128）| 🟠 中 | 不同任务最优值不同（论文扫了 32/64/128/256）|
| 代表 token 数固定（4）| 🟡 低 | 论文扫了 1/2/4/8 |
| 仅 autoregressive LLM | 🟠 中 | encoder-decoder 未测 |
| CPU-GPU 协同 | 🟠 中 | 端侧不可行，需简化 |
| Block 边界启发式 | 🟡 低 | 论文承认"如何动态切分"是开放问题 |
| 与 Mamba/SSM 比较 | 🟡 低 | 论文未深入对比 |

---

## 8. 关键引用块

> **核心定位**：
> "InfLLM stores distant contexts into additional memory units and employs an efficient mechanism to lookup token-relevant units for attention computation."

> **核心成果**：
> "Even when the sequence length is scaled to 1,024K, InfLLM still effectively captures long-distance dependencies."

> **设计哲学**：
> "Block-level memory units can save computation costs compared to token-level ones. It also poses new challenges for unit representations, which are supposed to contain the semantics of the entire unit for effective relevance score computation."

---

## 9. 相关工作

| 名称 | 与 InfLLM 关系 |
|------|---------------|
| [StreamingLLM](./streaming-llm.md) | 前作：滑动窗口 + attention sink（无 memory 检索）|
| [Memorizing Transformers](./memorizing-transformers.md) | 同期：kNN 检索 vs 块级检索 |
| [Compressive Transformers](./compressive-transformers.md) | 早期：压缩路径 |
| [discoloop.md](./discoloop.md) | 互补：Loop A（intra-decode）|
| [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) | Loop B 完整谱系 |
| [../research/sadko/whitepaper.md §2.1](../research/sadko/whitepaper.md) | Hippo 设计的对比参考 |

---

## 10. 对 LatentMind 的启示

### 10.1 短期（64M 阶段）

- ✅ 在 Hippo 实验 A 中加入 **block-level 思想**——把 FSQ 256 码本视为 256 个 block，每个 block 选 top-k 代表 token
- ✅ Router 设计可参考 InfLLM 的"按历史 attention 分数选代表"思想

### 10.2 中期（300M 阶段）

- 🔍 评估 **InfLLM 风格的 block-level memory** 作为 Hippo 的轻量级替代方案（避免 FM 压缩的端侧成本）
- 🔍 调研 **CPU/GPU 协同** 在 ChipForge APU 端侧的可行性

### 10.3 长期（1B+ 阶段）

- 🎯 **三种记忆方案可选**：
  - **Hippo FM + FSQ**（完整版，端到端可微）
  - **InfLLM block-level**（训练无关，中等端侧友好）
  - **StreamingLLM 滑动窗口**（最简，端侧最友好）
- 🎯 **可分阶段实施**：1B 训练时用 Hippo 完整版；推理时根据端侧能力选 InfLLM 或 StreamingLLM

---

**最后更新**：2026-07-31
**信息源**：arXiv 2402.04617v2 + NeurIPS 2024 PDF + GitHub 仓库
**作者**：来自 Loop B 文献谱系调研