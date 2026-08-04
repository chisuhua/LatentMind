# Memorizing Transformers 参考资料

> **论文**：*Memorizing Transformers*
> **作者**：Yuhuai Wu, Markus Rabe, DeLesley Hutchins, Christian Szegedy
> **机构**：Google Research
> **arXiv**：2203.08913v2（2022-03-16）
> **会议**：ICLR 2022 Spotlight
> **性质**：**Loop B（inter-decode 记忆压缩）代表架构**——最接近"持久记忆系统"流派的奠基作

---

## 1. 一句话定位

**在 Transformer 的某层（倒数第二层）插入 kNN 增强的注意力——同时维护**局部 dense attention 上下文**和**外部非可微的 (key, value) 记忆池**。**记忆大小可达 262K tokens**，性能随记忆增大稳步提升，**比参数 × 5 的基线 Transformer 效果更好**。

> 📌 **归属**：详见 [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) — 属于 Loop B 维度（inter-decode 记忆压缩），**不是** Loop A（intra-decode 表征对齐，如 DiscoLoop）

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | https://arxiv.org/abs/2203.08913 |
| OpenReview | https://openreview.net/forum?id=TrjbxzRcnf- |
| Google Research | https://research.google/pubs/memorizing-transformers/ |
| OpenReview PDF | https://openreview.net/pdf?id=TrjbxzRcnf- |

---

## 3. 核心创新

### 3.1 kNN 增强注意力层（核心机制）

在 Transformer 的**倒数第二层**插入一个 kNN 增强注意力层，**仅这一层使用 kNN 检索**，其余层保持标准 dense self-attention。

```python
# 每个 head 的混合公式
V_a = V_m * g + V_c * (1 - g)     # 公式 (1)(2)
g = σ(b_g)                          # g 是 per-head 标量参数（b_g 是 learned bias）
# V_m = kNN 检索外部 memory 的结果
# V_c = 局部 dense self-attention 的结果
```

**关键设计**：
- **kNN 检索**对每个 query 在外部 memory 中找 top-k (key, value)
- **g 是 learned 标量**：模型可学习何时信任 memory，何时信任 local context
- **位置编码**：局部 attention 用 T5 relative bias；kNN 检索的 memory 不加位置 bias（实验证明长程位置无影响）

### 3.2 外部记忆管理

- **每 step**：(key, value) 对从 local context 追加到 memory 末尾
- **FIFO 淘汰**：当文档过长，旧的 (key, value) 会被淘汰
- **每个 head 独立 memory**：per-head 的 cache of prior M (key, value) pairs
- **Non-differentiable**：kNN 查找是不可微的（论文作者证明这**不影响**效果——"lower layers don't need long-range context"）

### 3.3 架构与训练

- **Backbone**：标准 decoder-only Transformer
- **序列切片**：长文档分成 512 tokens 子序列
- **不 shuffle**：从前往后顺序处理（类似 Transformer-XL）
- **Memory 写入**：每步（每个 subsequence 后）追加 local context 的 (K, V)
- **Memory 读取**：每步每个 query 通过 kNN 检索 top-k

---

## 4. 关键数字

| 指标 | 值 | 备注 |
|------|:---:|------|
| 记忆大小（M）| 1,536 → 262,144 tokens | 最大 262K，单 TPU 可行 |
| 检索 k 值 | 默认未明确 | 每 query 独立检索 |
| Long-text 训练集 | C4(4K+), PG-19, arXiv, GitHub, Isabelle | 5 个长文档基准 |
| 序列窗口大小 | 512 tokens | 子序列 |
| kNN 层位置 | 倒数第二层 | 仅一层使用 kNN |
| PPL 提升（C4 4K+, 8K 记忆）| 17.20 → 14.42 | 显著 |
| PPL 提升（Transformer-XL 15.38 → 14.04）| 同上数据集 | 进一步改善 |
| 训练稳定性 | ⚠️ 大记忆时需 finetune | 131K+ 需先 pretrain 小记忆再 finetune |
| 速度 | 131K-262K tokens 可在单 TPU 运行 | 保持合理 step 时间 |
| **最大意义** | 8B 模型 + 记忆 > 5× 参数的 dense 模型 | 等效扩张上下文 |

---

## 5. 关键发现

### 5.1 记忆大小与性能的关系

> "We experimented with various sizes of external memory, from 1536 to as high as 262K. On most of the datasets, there was an initial sharp gain from adding a small external memory, followed by smaller but steadily increasing gains as the size of the memory was increased."

- **小记忆（2K-8K）即有显著增益**
- **增益随记忆大小稳步提升，无明显饱和**（到 262K 仍在涨）
- **训练-测试记忆大小可解耦**：训练用 8K 记忆，推理可用 262K

### 5.2 关于"小记忆 ≈ 长 dense context"

> "Using even a small external memory of size 2048 provides a gain in perplexity which is almost as good as using a local context of size 2048 but no memory."

**意义**：低层 Transformer 并不真正需要 long-range context —— 这是"循环记忆系统"流派的**核心论据**之一。

### 5.3 检索内容的分析

> "We found that the model gained the most when looking up rare words, such as proper names, references, citations, and function names, where the first use of a name is too far away from subsequent uses to fit in the local context."

**意义**：kNN 记忆的**主要价值**是补全**远距离实体的重复出现**——这与本项目 Hippo 的"知识召回"目标高度同源。

---

## 6. 与本项目架构的对照

| 维度 | Memorizing Transformers | 本项目 Hippo | 关键差异 |
|------|------------------------|-------------|----------|
| **存储形式** | (K, V) 对直接存储 | FSQ 离散码本 | Hippo 显式离散化 |
| **压缩** | ❌ 无压缩 | ✅ Flow Matching 压缩 | Hippo 信息密度更高 |
| **检索** | kNN（per-head）| Top-K 码字 + Cross-Attention | Hippo 有显式 Router |
| **可微** | ❌ 非可微 | 通过扩散蒸馏 | Hippo 端到端可微 |
| **记忆位置** | 倒数第二层 | 独立模块 | Hippo 跨模块 |
| **目标** | 长文档 PPL 改善 | 多模态记忆 + 知识召回 | Hippo 范围更广 |
| **架构归属** | Loop B（inter-decode）| Loop B（inter-decode）| ✅ 同一类别 |

**关键启示**：
- ✅ Hippo 的"知识召回"目标与 Memorizing Transformers 的"远距离实体"目标**同源**
- ✅ Hippo 的 FSQ 离散化可视为"Memorizing Transformers 的 kNN 检索的更结构化替代"
- ⚠️ Memorizing Transformers 的**可训练 gate g**（per-head 标量）可借鉴到 Hippo-Logos 集成
- ❌ **不**直接套用 kNN——Hippo 的 FSQ 码本已经是更结构化的检索空间

---

## 7. 局限性（对项目决策的边界）

| 限制 | 严重性 | 说明 |
|------|:---:|------|
| 1 个 layer 用 kNN | 🟠 中 | 多层都用是否有帮助未测 |
| Non-differentiable memory | 🟡 低 | 论文证明有效，但对训练稳定性有要求 |
| 512 tokens 子序列切分 | 🟡 低 | 与 Logos 的 K 循环不直接对齐 |
| 训练与推理的 memory 分布 | 🟠 中 | 大记忆需 finetune 阶段 |
| 仅 decoder-only 测试 | 🟡 低 | encoder-decoder 未测 |
| kNN 检索延迟 | 🔴 高 | 端侧不可行——论文在 TPU 上 |
| **核心发现："非可微 memory 也有效"** | 🟢 利好 | **支持 Hippo 端到端可微方向** |

---

## 8. 关键引用块

> **核心结论**：
> "A Memorizing Transformer does not need to be pre-trained from scratch; it is possible to obtain large gains from adding memory to an existing pre-trained model, and then fine-tuning it."

> **架构简洁性**：
> "The simplicity of the changes to the Transformer architecture allows us to easily integrate this approach into existing code bases, including extremely large language models."

> **长程语义机制**：
> "External memory continues to provide benefits even as the transformer is scaled up from 200M to 8B parameters."

---

## 9. 相关工作

| 名称 | 与 Memorizing Transformers 关系 |
|------|----------------------------------|
| [Transformer-XL](../research/loop-memory-survey.md) | 前作：仅维护 KV cache，不压缩不检索 |
| [Compressive Transformers](./compressive-transformers.md) | 同期：维护 memory + 压缩 memory |
| [StreamingLLM](./streaming-llm.md) | 后作：滑动窗口 + attention sink（更轻量）|
| [InfLLM](./inf-llm.md) | 后作：块级 memory + 训练无关 |
| [discoloop.md](./discoloop.md) | 互补：Loop A（intra-decode 表征对齐）|
| [loop-memory-survey.md §2.2](../research/loop-memory-survey.md) | Loop B 完整谱系位置 |
| [../research/sadko/whitepaper.md §2.1](../research/sadko/whitepaper.md) | Hippo 设计的同源参考 |
| [../research/hippo/README.md §2.2 I3](../research/hippo/README.md) | 胼胝体契约的 I3 Retrieval API |

---

## 10. 对 LatentMind 的启示

### 10.1 短期（64M 阶段）

- ❌ **不**引入 kNN 检索——64M 强约束下不验证复杂机制
- ✅ 关注 Hippo 实验 A 的"码本使用率 > 80%"目标——这是记忆系统有效性的关键指标

### 10.2 中期（300M 阶段）

- 🔍 调研 **Memorizing Transformers + 循环 Transformer 的融合**——是否在 Logos H/L 循环的某层加入 kNN 增强？
- 🔍 评估 **per-head gate g** 是否可借鉴到 Nano-WM Gate 与 FSQ 通道 gate

### 10.3 长期（1B+ 阶段）

- 🎯 若 Hippo FSQ 验证通过，可考虑 **FSQ 码本 vs kNN** 的"双轨记忆"方案
- 🎯 **可训练 gate g** 的设计哲学可延伸到"何时信任记忆 vs 何时依赖循环计算"

---

**最后更新**：2026-07-31
**信息源**：arXiv 2203.08913v2 + OpenReview + 微信文章调研工作流
**作者**：来自 Loop B 文献谱系调研（用户委托 → 概念澄清 → 论文笔记落地）