# Recurrent Memory Transformer (RMT) 参考资料

> **论文**：*Recurrent Memory Transformer*
> **作者**：Aydar Bulatov, Yuri Kuratov, Mikhail Burtsev
> **机构**：Moscow Institute of Physics and Technology + AIRI
> **arXiv**：2207.06881v2（2022-07-14）
> **会议**：NeurIPS 2022
> **代码**：[github.com/booydar/recurrent-memory-transformer](https://github.com/booydar/recurrent-memory-transformer)
> **性质**：**Loop B（inter-decode 记忆压缩）的"特殊 memory tokens"流派**——RMT 是 AutoCompressors 的直接前置，但更接近"循环 Transformer + 外部 memory"流派

---

## 1. 一句话定位

**在输入序列两端各加 m 个 `[mem]` 特殊 token——前段 m 个作"读 memory"（让当前段 tokens attend 到上段信息），后段 m 个作"写 memory"（让当前段信息写入下段输入）。无需修改 Transformer 架构，仅通过输入输出 token 序列实现**。RMT 在 copy / reverse 等长序列任务上**完美击败 Transformer-XL**，且 memory size 小 10× 即可达到 Tr-XL 的 LM 性能。

> 📌 **归属**：详见 [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) — 属于 Loop B 维度（inter-decode 记忆压缩），是**特殊 memory tokens** 流派的奠基作，被 AutoCompressors 直接继承

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | https://arxiv.org/abs/2207.06881 |
| NeurIPS 2022 PDF | https://proceedings.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf |
| OpenReview | https://openreview.net/pdf?id=Uynr3iPhksa |

---

## 3. 核心创新

### 3.1 Memory Tokens（核心机制）

**输入序列构造**：

```python
# 每个 segment τ 的输入
H_τ = [H_read τ ◦ H_segment τ ◦ H_write τ]  # 拼接
# - H_read τ: m 个读 memory tokens（在段首）
# - H_segment τ: 原始 segment 的 hidden states
# - H_write τ: m 个写 memory tokens（在段尾）
```

**段间传递**：

```python
# 段 τ 处理后，写 memory tokens 的 hidden states 传到段 τ+1
H_mem_τ+1 := H_write_τ  # 写入 → 下段的读

# 段 τ+1 构造输入
H_τ+1 = [H_mem_τ+1 ◦ H_segment_τ+1 ◦ H_mem_τ+1]
```

**关键设计**：
- **无需修改 Transformer 架构**——仅在输入输出层操作
- **m memory tokens per side**（典型 10-25）—— 极轻量
- **写 memory attention** 允许全段 attention（不受 causal mask 限制）
- **读 memory causal attention** 仅看到前段 memory

### 3.2 BPTT 训练

- **Backpropagation Through Time** 跨段反向传播
- 训练时可调 BPTT unroll 深度（0-4 段）
- 与 Transformer-XL 相比：**memory gradients not stopped between segments**（RMT 的关键区别）

### 3.3 与 Transformer-XL 的关系

| 维度 | RMT | Transformer-XL |
|------|-----|-----------------|
| **存储量** | m × 1 段 | m × N 层 |
| **存储位置** | 特殊 memory tokens | hidden activations |
| **跨段信息深度** | 有效深度 = τ × N（更深）| 浅（受深度限制）|
| **修改架构** | ❌ 无需 | ❌ 无需 |
| **训练反向传播** | 不 stop gradients | stop gradients |

**关键洞察**（论文）：RMT 的 memory 性能更好，因为**有效深度更深**——memory tokens 经过每段的 N 层处理。

### 3.4 任务表现

| 任务 | 序列长度 | Transformer baseline | Transformer-XL | RMT |
|------|---------|---------------------|-----------------|-----|
| **Copy** | 50 segments | ❌ 失败 | ✅ 完美 | ✅ 完美 |
| **Reverse** | 50 segments | ❌ 失败 | ✅ 完美 | ✅ 完美 |
| **Associative retrieval** | 50 segments | ❌ 失败 | ✅ 完美 | ✅ 完美 |
| **LM WikiText-103** | 段长 150 | 差 | 中 | 中（memory size 10-25 ≈ Tr-XL memory 75）|
| **LM WikiText-103** | 段长 50 | 差 | 中 | RMT-1 ≈ Tr-XL |

**关键数字**：RMT 在 50 段以上长序列任务上**显著超过** Transformer-XL（Tr-XL 在更长序列上因 depth 限制性能下降）。

---

## 4. 关键数字

| 指标 | 值 | 备注 |
|------|:---:|------|
| Memory tokens 数 m | 1-25 per side | 默认 10 |
| 段长 | 50 / 150 tokens | 任务相关 |
| BPTT unroll 深度 | 0-4 段 | 训练时 |
| Copy/Reverse 任务序列长 | 50 segments | 完美 |
| LM 任务 | WikiText-103 | 性能 ≈ Tr-XL（memory 小 10×）|
| 与 Tr-XL 联合 | ✅ 可叠加 | 进一步改善 |
| **RoBERTa 集成** | ✅ 长期文本分类 SOTA | RMT + RoBERTa 组合 |

---

## 5. 与本项目架构的对照

| 维度 | RMT | 本项目 Hippo | 关键差异 |
|------|-----|-------------|----------|
| **Memory 形式** | 特殊 `[mem]` tokens | FSQ 256 码字 | Hippo 离散化更结构化 |
| **修改架构** | ❌ 无需 | ⚠️ 需 Hippo 模块 | Hippo 跨模块 |
| **累积** | ❌ 仅最新段 | ❌ Router 单选（但可改进为累积）| 当前一致 |
| **训练** | BPTT 跨段 | 端到端可微 | 相同 |
| **端侧友好** | ✅ 极轻量（m=10）| ⚠️ FSQ + FM | RMT 更轻 |
| **任务范围** | 通用 LM + 长序列任务 | 多模态 + 知识召回 | Hippo 更丰富 |
| **架构归属** | Loop B（inter-decode）| Loop B（inter-decode）| ✅ 同一类别 |

**关键启示**：
- ✅ **特殊 memory token 思想**可作为 SADKO 扩展接口——为 Hippo 增设 `<MemRead>` `<MemWrite>` 特殊 token
- ✅ **BPTT 不 stop gradient** 思想可借鉴——Hippo 多段累积训练
- ✅ **轻量级 memory** 思想可作为端侧 fallback——RMT m=10 远小于 Hippo FSQ 256 码本
- ❌ **不**直接套用——Hippo 多模态目标要求 FSQ 离散化，RMT 仅文本

---

## 6. 局限性

| 限制 | 严重性 | 说明 |
|------|:---:|------|
| 段长固定 | 🟠 中 | 推理时需预设段大小 |
| 训练需 BPTT 跨段 | 🟠 中 | 长序列训练成本 |
| Memory tokens 信息容量有限 | 🟡 低 | m=10 编码能力有限 |
| 仅文本 | 🔴 高 | 论文未涉及多模态 |
| 与本项目 SADKO 集成路径 | 🟡 低 | 需要架构级改造 |
| Memory tokens 解释性 | 🟢 利好 | 可解释（论文 §4） |

---

## 7. 关键引用块

> **核心机制**：
> "RMT uses a memory mechanism based on special memory tokens (Burtsev et al., 2020) added to the input sequence. Memory tokens provide additional reserved capacity to the model that could be used to process information which is not directly representing any element in the input sequence."

> **关键优势**：
> "RMT learns to use smaller memory more effectively than Transformer-XL. Additionally, the smaller memory size of RMT leads to reducing required GPU memory for running the model."

> **与 AutoCompressors 关系**：
> "Our work builds on the recently proposed RMT architecture (Bulatov et al., 2022) with a crucial difference: we introduce summary accumulation..."（AutoCompressors 论文原文）

---

## 8. 相关工作

| 名称 | 与 RMT 关系 |
|------|------------|
| [AutoCompressors](./auto-compressors.md) | **直接后继**——summary accumulation 改进 |
| [Memorizing Transformers](./memorizing-transformers.md) | 同期：检索路径（vs RMT 特殊 token 路径）|
| [Compressive Transformers](./compressive-transformers.md) | 早期：压缩路径 |
| [StreamingLLM](./streaming-llm.md) | 后作：滑动窗口 + attention sink |
| [InfLLM](./inf-llm.md) | 后作：块级检索 |
| [Transformer-XL](https://arxiv.org/abs/1901.02860) | 直接比较基线 |
| [discoloop.md](./discoloop.md) | 互补：Loop A（intra-decode）|
| [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) | Loop B 完整谱系 |
| [../research/sadko/whitepaper.md §2.1](../research/sadko/whitepaper.md) | Hippo 设计的对比参考 |

---

## 9. 对 LatentMind 的启示

### 9.1 短期（64M 阶段）

- ❌ **不**引入 memory tokens——64M 强约束下不验证复杂机制
- ✅ 关注 **特殊 token 嵌入** 思想——为后续 SADKO 扩展做准备

### 9.2 中期（300M 阶段）

- 🔍 评估 **RMT 风格 + Hippo FSQ 组合**——RMT 的特殊 tokens 思路可作为 Hippo 接口的轻量级扩展
- 🔍 调研 **summary accumulation 思想**——Hippo 多段检索时不要只保留最新段
- 🔍 若 Hippo 64M 验证失败，**回退**到 RMT 风格的特殊 memory tokens 路径

### 9.3 长期（1B+ 阶段）

- 🎯 **"特殊 token" 接口**可作为 Logos / SADKO / Hippo / Thumos 四线统一接口契约
- 🎯 **RMT + FSQ 融合**——特殊 token 路由到 FSQ 码本

---

**最后更新**：2026-07-31
**信息源**：arXiv 2207.06881v2 + NeurIPS 2022 PDF
**作者**：来自 Loop B 文献谱系调研