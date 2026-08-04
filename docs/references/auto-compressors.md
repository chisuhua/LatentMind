# AutoCompressors 参考资料

> **论文**：*Adapting Language Models to Compress Contexts*
> **作者**：Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen
> **机构**：Princeton NLP
> **arXiv**：2305.14788v2（2023-05-24）
> **会议**：EMNLP 2023 Main
> **代码**：[github.com/princeton-nlp/AutoCompressors](https://github.com/princeton-nlp/AutoCompressors)
> **模型**：[princeton-nlp/AutoCompressor-Llama-2-7b-6k](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k) 等
> **性质**：**Loop B（inter-decode 记忆压缩）的 LLM 自压缩流派代表**——让 LLM 自身学习生成"summary vectors"作为软提示

---

## 1. 一句话定位

**在预训练 LM 的词表中加入 κ 个 `<Sum>` 特殊 token，让 LM 把长文档分段处理并自动输出 summary vectors（κ 个连续向量）作为下段的软提示**。**Summary accumulation**（所有段向量拼接）+ **randomized segmenting** 是关键创新。在 OPT-2.7B / Llama-2-7B 上验证，最长处理 30,720 tokens。

> 📌 **归属**：详见 [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) — 属于 Loop B 维度（inter-decode 记忆压缩），**第 4 种路径**：让 LLM **自身**生成压缩表示（区别于 Memorizing 的 kNN 检索 / Compressive 的 1D Conv / StreamingLLM 的滑动窗口）

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | https://arxiv.org/abs/2305.14788 |
| ACL Anthology | https://aclanthology.org/2023.emnlp-main.232/ |
| GitHub | https://github.com/princeton-nlp/AutoCompressors |
| HuggingFace | [princeton-nlp/AutoCompressor-Llama-2-7b-6k](https://huggingface.co/princeton-nlp) |

---

## 3. 核心创新

### 3.1 Summary Vectors（核心机制）

在 LM 词表中加入 κ 个特殊 token `<Sum>_1, ..., <Sum>_κ`，当输入末尾追加这些 token 时，模型**输出对应的 hidden states** 作为 summary vectors。

```python
# 输入：原文 segment + <Sum> tokens
input_segment = "...long document..."
input_with_sum_tokens = input_segment + "<Sum>_1 <Sum>_2 ... <Sum>_κ"

# 前向：得到 summary vectors
outputs = model(input_with_sum_tokens)
summary_vectors = outputs.hidden_states[:, -κ:, :]  # shape: [B, κ, d_model]
```

**关键特性**：
- 软提示（soft prompt）：连续向量，可表达比离散 token 更抽象的概念
- κ 默认 50（每段 50 个 summary vectors）
- 单段压缩比：2048 tokens → 50 vectors（≈ 40:1 压缩）

### 3.2 Summary Accumulation（关键改进）

把所有段的 summary vectors **拼接**而非只保留最新：

```python
# 第 i 段处理后
σ_i = model_summary(segment_i)  # 当前段 summary
σ_<i = concat([σ_1, σ_2, ..., σ_{i-1}])  # 累积所有历史 summary
# 第 i+1 段处理时
input_{i+1} = concat([σ_<i, segment_{i+1}])  # 把累积的软提示拼到段首
```

**优势**：每个段都能直接访问**所有**前序段的信息（非仅前一段），解决 RMT 的"前序信息衰减"问题。

### 3.3 Randomized Segmenting

训练时**随机化**每段长度（subject to context window 限制），让模型对**变长**输入都鲁棒。

### 3.4 BPTT with stop-gradients

- 训练用 BPTT（截断到 2 段）
- 段间 summary gradients stop（类似 Transformer-XL 缓存）

---

## 4. 关键数字

| 指标 | 值 | 备注 |
|------|:---:|------|
| **最大处理序列** | **30,720 tokens** | OPT-2.7B finetune |
| **Summary vectors 数 κ** | 50（默认）| 可调 |
| **单段压缩比** | 2048 tokens → 50 vectors (40:1) | |
| **总压缩比** | 6144 tokens → 50 vectors (123:1) | 4 段累积 |
| **单 GPU** | NVIDIA A100 80GB | 完整训练 |
| **基座模型** | OPT-2.7B / Llama-2-7B | 完整微调 |
| **训练数据** | 2B tokens（OPT） / 15B tokens（Llama-2）| RedPajama / Books3 |
| **下游任务** | 11 SuperGLUE 任务，8/11 优于 plain-text ICL | |

### 关键发现

> "Compressing a 4,096-token context into 100 summary vectors achieves similar perplexity to the Extended Full Attention baseline with 512 plain text tokens, and compressing a 6,144-token context into 150 summary vectors further improves perplexity slightly."

**意义**：summary vectors 编码的信息密度**远高于** plain-text tokens（30-50× 信息密度）。

---

## 5. 与其他 Loop B 论文的关系

### 5.1 谱系定位

```
Loop B 记忆系统谱系：
├── RMT (Bulatov 2022)         ← AutoCompressors 的前置
├── Memorizing Transformers (2022)  ← 检索路径
├── Compressive Transformers (2020)  ← 压缩路径
├── AutoCompressors (2023)     ← LLM 自压缩路径（本文）
├── StreamingLLM (2024)         ← 滑动窗口 + anchor
└── InfLLM (2024)              ← 块级检索
```

**关键澄清**：
- AutoCompressors 是 RMT 的**直接后继**（Chevalier et al. 在论文中明确"our work builds on RMT"）
- 关键改进：summary accumulation（vs RMT 只保留最新段 summary）

### 5.2 与本项目 Hippo 的对照

| 维度 | AutoCompressors | 本项目 Hippo | 关键差异 |
|------|-----------------|-------------|----------|
| **压缩方式** | LM 自身生成 summary vectors | Flow Matching 显式压缩 | Hippo 更可控 |
| **存储单元** | κ=50 连续向量 | FSQ 256 离散码字 | Hippo 离散化更结构化 |
| **训练成本** | 中（finetune 2-15B tokens）| 高（端到端 FM）| AutoCompressors 较低 |
| **可微** | ✅ 端到端 | ✅ 端到端 | 相同 |
| **端侧友好** | ✅ 软提示机制简单 | ⚠️ FM 计算重 | AutoCompressors 更友好 |
| **累积机制** | ✅ Summary accumulation | ❌ Hippo 用 Router 选择 | 不同的多段信息访问方式 |
| **多模态** | ❌ 仅文本 | ✅ 多模态 | Hippo 范围更广 |
| **架构归属** | Loop B（独立模块）| Loop B（独立模块）| ✅ 同一类别 |

**关键启示**：
- ✅ **Summary accumulation 思想**可借鉴到 Hippo 的多段检索——不要只保留最新段，要累积所有历史
- ✅ **特殊 token 嵌入**思想可借鉴到 SADKO——为 Hippo 增设 `<MemRead>` `<MemWrite>` 特殊 token
- ✅ **软提示（连续向量）vs 离散码字**的对比——Hippo 的 FSQ 已做离散化，但若端侧成本过高，**回退**到软提示版本
- ❌ **不**直接套用——AutoCompressors 仅文本，Hippo 多模态目标要求 FSQ 离散化

---

## 6. 局限性

| 限制 | 严重性 | 说明 |
|------|:---:|------|
| 仅文本 | 🔴 高 | 论文未涉及多模态 |
| 训练数据需 2-15B tokens | 🟠 中 | 64M 阶段无法独立 finetune |
| 压缩比上限 | 🟡 低 | κ=50 时 40:1，再大可能信息丢失 |
| 累积信息线性增长 | 🟠 中 | σ_<i 长度 = (i-1) × κ，无界增长 |
| 段长度固定 | 🟡 低 | 推理时需预设段大小 |
| 端到端 fine-tune 成本 | 🟠 中 | 需完整 LM finetune，不能 add-on |

---

## 7. 关键引用块

> **核心定位**：
> "These language models are capable of compressing long contexts into compact summary vectors, which are then accessible to the model as soft prompts."

> **Summary accumulation 创新**：
> "We propose summary accumulation, in which summary vectors from all segments are concatenated to produce the summary of the entire document."

> **下游应用**：
> "Re-ranking passages based on their summary vectors achieves the best trade-off between re-ranking performance and inference throughput."

> **相对 RMT 改进**：
> "Our work builds on the recently proposed RMT architecture (Bulatov et al., 2022) with a crucial difference: we introduce summary accumulation..."

---

## 8. 相关工作

| 名称 | 与 AutoCompressors 关系 |
|------|------------------------|
| [RMT (Bulatov 2022)](./rmt.md) | **直接前置**——特殊 memory tokens 思想 |
| [Memorizing Transformers](./memorizing-transformers.md) | 同期：检索路径 |
| [Compressive Transformers](./compressive-transformers.md) | 早期：压缩路径（被 AutoCompressors 间接借鉴）|
| [StreamingLLM](./streaming-llm.md) | 后作：滑动窗口路径 |
| [InfLLM](./inf-llm.md) | 后作：块级检索路径 |
| [discoloop.md](./discoloop.md) | 互补：Loop A（intra-decode）|
| [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) | Loop B 完整谱系 |
| [../research/sadko/whitepaper.md §2.1](../research/sadko/whitepaper.md) | Hippo 设计的对比参考 |

---

## 9. 对 LatentMind 的启示

### 9.1 短期（64M 阶段）

- ❌ **不**引入 summary vectors——64M 强约束下无法 finetune 完整 LM
- ✅ 关注 **"累积" 思想**——Hippo 多段检索时考虑累积所有历史段

### 9.2 中期（300M 阶段）

- 🔍 评估 **Hippo FSQ 是否需要软提示回退**——若 FSQ 256 码本利用率 < 80%，可考虑 κ=50 软提示 + summary accumulation 替代
- 🔍 调研 **特殊 token 嵌入**（`<MemRead>` `<MemWrite>`）作为 SADKO 扩展接口

### 9.3 长期（1B+ 阶段）

- 🎯 **"内化压缩能力"** 思想可作为 Logos 主干扩展——不只外部 Hippo，模型自身也能压缩
- 🎯 **Summary accumulation** 可作为多段对话记忆的物理实现

---

**最后更新**：2026-07-31
**信息源**：arXiv 2305.14788v2 + EMNLP 2023 + GitHub 仓库
**作者**：来自 Loop B 文献谱系调研