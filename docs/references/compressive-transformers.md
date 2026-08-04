# Compressive Transformers 参考资料

> **论文**：*Compressive Transformers for Long-Range Sequence Modelling*
> **作者**：Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Timothy Lillicrap
> **机构**：DeepMind
> **arXiv**：1911.05507v2（2019-11-13，最后修订 2020）
> **会议**：ICLR 2020
> **性质**：**Loop B（inter-decode 记忆压缩）先驱**——首次提出"压缩旧记忆"概念，是 Memorizing Transformers 的同期作品

---

## 1. 一句话定位

**在 Transformer-XL 的 memory 基础上，把即将淘汰的旧 KV 通过卷积压缩函数 f_c 压缩成 secondary "compressed memory"**——**细粒度 memory** + **粗粒度 compressed memory** 双缓冲，attention 同时查询两者。**首次将"睡眠巩固记忆"的认知科学思想引入 Transformer 架构**。

> 📌 **归属**：详见 [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) — 属于 Loop B 维度（inter-decode 记忆压缩），是"压缩"路径的先驱（Memorizing Transformers 是"检索"路径的先驱）

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | https://arxiv.org/abs/1911.05507 |
| OpenReview | https://openreview.net/forum?id=SylKikSYDH |
| DeepMind Blog | https://deepmind.google/blog/a-new-model-and-dataset-for-long-range-memory/ |
| ar5iv HTML | https://ar5iv.labs.arxiv.org/html/1911.05507 |

---

## 3. 核心创新

### 3.1 双缓冲记忆架构（核心机制）

```
Sequence (n_s) ──> 主模型 ──> Memory (n_m, FIFO) ──> 旧 memory ──> f_c ──> Compressed Memory (n_cm)
        ↑                                                                │
        └────────────── Attention over both ──────────────────────────────┘
```

- **Sequence**：当前处理的 ns 个 token
- **Memory**：FIFO 缓存最近的 nm 个 hidden activations
- **Compressed Memory**：旧的 memory 经压缩函数 f_c 压缩后的 ncm 个粗粒度表示

### 3.2 压缩函数 f_c（三种实现）

论文实验了三种压缩函数：

| 函数 | 描述 | 特点 |
|------|------|------|
| **Max / Mean pooling** | 最简单 | 无学习 |
| **1D Convolution** (最优) | stride=c, kernel=k≥c | 局部线性组合 |
| **Dilated Convolution** | 多层空洞卷积 | 扩大感受野 |

**关键设计**：
- 压缩率 c：nm / ncm
- 论文最佳 c=3 或 4
- 1D Conv（kernel=c）= 邻接不重叠压缩
- Dilated Conv = 跨范围压缩

### 3.3 注意力机制

Transformer-XL 的 attention 机制扩展为同时查询 memory + compressed memory：

```python
# 每个 step 的 attention 输入
queries = current_sequence
keys = concat([memory_keys, compressed_memory_keys])
values = concat([memory_values, compressed_memory_values])
output = attention(queries, keys, values)
```

**时间范围扩展**（与 Transformer-XL 对比）：
- Transformer-XL：max range = l × n（l 层 × n memory 大小）
- Compressive：max range = l × (n_m + c × n_cm)，**相同 attention cost**

### 3.4 辅助压缩损失（auxiliary loss）

为训练压缩函数 f_c，引入辅助损失（类似记忆回放）：

```python
# 在压缩前后的 hidden states 上计算重建损失
loss_aux = MSE(memory_old, f_c.inverse(memory_compressed))
# 总损失 = 主 LM loss + α * loss_aux
```

---

## 4. 关键数字

| 指标 | 值 | 备注 |
|------|:---:|------|
| Enwik8 BPC | **0.97** | 当时 SOTA，超过 Transformer-XL |
| WikiText-103 PPL | **17.1** | 当时 SOTA，超过 Transformer-XL 1.2 PPL |
| PG-19 PPL | 36.5 (24-layer) | 论文新提出的 book-level benchmark |
| 压缩率 c | 3 或 4 (1D conv 最优) | 论文扫了 c ∈ {2, 3, 4} |
| Memory 大小 (训练) | 768 (Enwik8), 512 (WikiText-103) | |
| Compressed Memory 大小 | 1152 (Enwik8), 1500 (WikiText-103) | 评估时 3072/1536 |
| 总 attention cost | `O(n_s² + n_s × (n_m + n_cm))` | 与 Transformer-XL 相同 |
| 时间范围扩展 | 2× Transformer-XL（c=3, n_m=n_cm=n/2）| 显著优势 |
| 罕见词改善 | 显著 | "improve rare words modeling" |

---

## 5. 关键发现与设计哲学

### 5.1 睡眠巩固隐喻

> "The Compressive Transformer takes inspiration from the role of sleep in the formation of consolidated episodic memories. Sleep is known to be crucial for memory, and it's thought that sleep serves to compress and consolidate memories."

**意义**：这是**首次**将认知科学中的"记忆巩固"思想引入 Transformer 架构，为后续工作提供哲学基础。

### 5.2 注意力稀疏性观察

> "We also show that the Compressive Transformer works not only for language, but can also model the waveform of high-frequency speech with a trend of lower likelihood than the TransformerXL and Wavenet when trained over 400,000 steps."

**意义**：压缩机制对**多模态**也有效（语音），预示后续多模态记忆系统可能。

### 5.3 与"真实记忆 vs 检索"的对比

| 维度 | Compressive Transformer | Memorizing Transformers |
|------|-------------------------|--------------------------|
| 哲学 | 压缩（睡眠巩固）| 检索（数据库查找）|
| 存储形式 | 压缩后的向量 | 原始 (K, V) 对 |
| 训练 | 端到端（含辅助损失）| 直接训练（kNN 不可微）|
| 端侧友好 | ✅ 简单 | ❌ kNN 检索重 |
| 信息丢失 | ⚠️ 有（压缩过程）| ✅ 无 |

---

## 6. 与本项目架构的对照

| 维度 | Compressive Transformer | 本项目 Hippo | 关键差异 |
|------|------------------------|-------------|----------|
| **压缩方式** | 1D Conv（无学习任务语义）| Flow Matching（学习流形）| Hippo 语义感知 |
| **压缩率** | 2-4× | FSQ 256 码本（视作高压缩比）| 不同表示空间 |
| **注意力查询** | 同一 attention over both | Cross-Attention 单独模块 | Hippo 跨模块 |
| **训练损失** | LM loss + 辅助重建损失 | LM loss + 扩散对齐损失 | Hippo 对齐更复杂 |
| **端到端可微** | ✅ | ✅ | 相同 |
| **架构归属** | Loop B（inter-decode）| Loop B（inter-decode）| ✅ 同一类别 |

**关键启示**：
- ✅ **1D Conv 压缩**可作为 Hippo FM 压缩的**轻量级替代方案**——若 64M 阶段 FM 太重，可考虑 Conv 兜底
- ✅ **辅助压缩损失**思想可借鉴——给 FSQ 训练加类似的辅助目标（如"重建原始 KV"）
- ✅ **睡眠巩固隐喻**可写入 Hippo README 作为哲学背书
- ⚠️ Compressive Transformer 的压缩是"无目的"的，Hippo 的 FM 是"目的性"的（保留流形）—— 不可直接混用

---

## 7. 局限性

| 限制 | 严重性 | 说明 |
|------|:---:|------|
| 压缩信息丢失 | 🟠 中 | 1D Conv 是局部线性组合，远距离语义可能丢失 |
| 端到端 BPTT 长 | 🟠 中 | 训练需 unroll 长时间窗口 |
| Memory 大小有限 | 🟡 低 | 论文最大 3072 压缩 memory |
| 无跨序列检索 | 🟡 低 | 不像 Memorizing Transformers 的 kNN |
| 未在循环 Transformer 中测试 | 🟠 中 | 论文是标准 Transformer-XL 基底，与 Logos H/L 集成未验证 |
| 无 attention 稀疏性优化 | 🟡 低 | dense attention over all memories |

---

## 8. 关键引用块

> **核心贡献**：
> "The Compressive Transformer, a simple extension to the Transformer which maps past hidden activations (memories) to a smaller set of compressed representations (compressed memories). The Compressive Transformer uses the same attention mechanism over its set of memories and compressed memories, learning to query both its short-term granular memory and longer-term coarse memory."

> **哲学隐喻**：
> "Sleep is known to be crucial for memory, and it's thought that sleep serves to compress and consolidate memories, thereby improving reasoning abilities for memory tasks."

> **时间范围扩展**：
> "We obtain a maximum temporal range that is two times greater than the TransformerXL with an identical attention cost."

---

## 9. 相关工作

| 名称 | 与 Compressive Transformers 关系 |
|------|-----------------------------------|
| [Transformer-XL](../research/loop-memory-survey.md) | 直接前置：仅维护 memory，不压缩 |
| [Memorizing Transformers](./memorizing-transformers.md) | 同期另一种思路：检索 vs 压缩 |
| [StreamingLLM](./streaming-llm.md) | 后作：更激进的滑动窗口 + attention sink |
| [InfLLM](./inf-llm.md) | 后作：训练无关的块级 memory |
| [AutoCompressors](#) | 后作：让 LLM 自身生成摘要向量 |
| [discoloop.md](./discoloop.md) | 互补：Loop A（intra-decode）vs 本篇 Loop B |
| [../research/sadko/whitepaper.md §2.1](../research/sadko/whitepaper.md) | Hippo Flow Matching 压缩的对比参考 |
| [../research/hippo/README.md §2.2](../research/hippo/README.md) | FSQ 码本（vs Conv 压缩） |

---

## 10. 对 LatentMind 的启示

### 10.1 短期（64M 阶段）

- ✅ 在 Hippo 实验 A 中加入**"压缩信息保留率"指标**——这是 Compressive Transformer 启发的核心测量
- ✅ 评估 **FSQ 256 码本 vs Conv 压缩 vs FM 压缩**的端到端性能对比

### 10.2 中期（300M 阶段）

- 🔍 若 FM 压缩端侧成本过高，**1D Conv 压缩**作为轻量级备选方案
- 🔍 评估 **辅助压缩损失**（重建损失）vs **扩散对齐损失**在 SADKO 训练中的相对重要性

### 10.3 长期（1B+ 阶段）

- 🎯 "**睡眠巩固**" 隐喻可作为长期架构演进的目标——阶段性重压缩
- 🎯 **分形码本**（SADKO 已有设想）可借鉴 Compressive Transformer 的"双粒度"思想

---

**最后更新**：2026-07-31
**信息源**：arXiv 1911.05507v2 + OpenReview + DeepMind Blog
**作者**：来自 Loop B 文献谱系调研