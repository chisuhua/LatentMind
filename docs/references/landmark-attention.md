# Landmark Attention 参考资料

> **论文**：*Landmark Attention: Random-Access Infinite Context Length for Transformers*
> **作者**：Amirkeivan Mohtashami, Martin Jaggi
> **机构**：EPFL（洛桑联邦理工）
> **arXiv**：2305.16300v2（2023-05-25）
> **会议**：NeurIPS 2023（ES-FoMo Workshop Poster）
> **代码**：[github.com/epfml/landmark-attention](https://github.com/epfml/landmark-attention/)
> **性质**：**Loop B（inter-decode 记忆压缩）的"landmark gating"流派**——每个 block 末尾插入 landmark token 作为 attention gate，可在推理时处理任意长度上下文（**已验证 LLaMA 7B 扩展到 32K+ tokens**）

---

## 1. 一句话定位

**每个 block 末尾插入 1 个 landmark token 训练 attention 学会用它做 block 检索 gate——通过 attention score 即可识别相关 block，整套机制**通过 attention 自身实现**而非外挂**。**fine-tuning LLaMA 7B 可扩展到 32K+ tokens**（GPT-4 等同上下文长度），无需重新训练。

> 📌 **归属**：详见 [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) — 属于 Loop B 维度（inter-decode 记忆压缩），是**最优雅的 attention 内生检索**方案（不依赖 kNN 或块级调度）

---

## 2. 链接速查

| 类型 | 链接 |
|---|---|
| arXiv | https://arxiv.org/abs/2305.16300v2 |
| GitHub | https://github.com/epfml/landmark-attention/ |
| OpenReview | https://openreview.net/forum?id=PkoGERXS1B |
| NeurIPS 2023 PDF | https://proceedings.neurips.cc/paper_files/paper/2023/file/ab05dc8bf36a9f66edbff6992ec86f56-Paper-Conference.pdf |

---

## 3. 核心创新

### 3.1 Landmark Token（核心机制）

**输入序列构造**：

```python
# 每个 block（ℓ tokens）末尾插入 1 个 landmark token
sequence = [tok_1, tok_2, ..., tok_ℓ, <landmark>, tok_ℓ+1, ..., tok_2ℓ, <landmark>, ...]
```

**训练目标**：让 model 学会用 landmark token 的 attention score 作为对应 block 的"代表向量"。

```python
# 修改后的 attention
# 关键：tokens in same block + landmark 共享 softmax group
# → model 必须 choose between attending to other blocks and current tokens
```

**核心机制**：
- block 内所有 tokens（包括 landmark）**共享一个 softmax group**
- block 间的 attention 通过**对 landmark 的 attention score**作为 gate
- 训练目标隐式学会："用 landmark score 判断 block 相关性"

### 3.2 推理时的随机访问

```python
# 推理时
# 1. 分块处理，每块末尾有 landmark
# 2. 维护 KV cache（含 landmark KV）
# 3. 每层：先计算当前 query 与所有 landmark 的 score
# 4. 选 top-k landmark → 加载对应 block 的所有 tokens
# 5. 与本地 attention 拼接 → 输出
```

**核心优势**：
- **无限上下文**——landmark attention 可处理任意长度的输入（无训练时长度限制）
- **随机访问**——可访问任何 block（vs Tr-XL 只能 access 前段）
- **可解释**——可看到 model 实际 retrieve 了哪些 block

### 3.3 显存优化（CPU/GPU 协同）

```python
# 仅 landmark 的 KV 保留在 GPU 显存
# block 普通 tokens 卸载到 CPU 内存或磁盘
# 仅当对应 block 被选中时再 swap in
```

**关键设计**：
- 仅 landmark 需常驻 GPU
- block tokens 按需 swap（类似 OS 虚拟内存）
- 显著降低端侧显存需求

---

## 4. 关键数字

| 指标 | 值 | 备注 |
|------|:---:|------|
| Block size ℓ | 250 tokens (典型) | |
| LLaMA 7B finetune 上下文 | 2048 tokens | 训练时 |
| LLaMA 7B 推理上下文 | **32K+ tokens** | 远超训练长度 |
| 检索 blocks k | 2-4 | top-k retrieval |
| LLaMA 7B 模型卡 | 32K+ tokens | GPT-4 等同 |
| 性能 vs Tr-XL | 相当（PPL 相当）| 显存少 |
| 关键实验 | PG-19, arXiv math | 长文档 LM |

### 关键发现

> "We demonstrate that using our method to fine-tune LLaMA 7B, a large language model, allows it to retrieve relevant information from contexts with over 32k tokens, which is the context length of GPT-4."

**意义**：**fine-tune 即可扩展到 32K+**——无需从头训练，与 64M 验证理念同源。

---

## 5. 与本项目架构的对照

| 维度 | Landmark Attention | 本项目 Hippo | 关键差异 |
|------|-------------------|-------------|----------|
| **检索机制** | Attention score 内生（无需 kNN）| Router + Top-K | Hippo 显式但需额外模块 |
| **Block 标记** | 1 landmark token per block | FSQ 256 码字 | 不同的"代表单元" |
| **训练开销** | LLaMA 7B finetune（低）| 端到端 FM（高）| Landmark 更轻 |
| **可微** | ✅ | ✅ | 相同 |
| **端侧友好** | ✅✅（landmark 极轻量）| ⚠️ | Landmark 更友好 |
| **可解释性** | ✅✅（attention 可视化）| ⚠️ | Landmark 优势 |
| **任务范围** | 文本 | 多模态 + 知识召回 | Hippo 更丰富 |
| **架构归属** | Loop B（inter-decode）| Loop B（inter-decode）| ✅ 同一类别 |

**关键启示**：
- ✅ **Attention 内生检索**思想可借鉴到 Hippo Router——让 attention 自身学会选 block
- ✅ **CPU/GPU 协同** + **按需 swap** 思想对端侧部署关键
- ✅ **Fine-tune 即可扩展上下文**思想支持 Logos 1B 端侧部署
- ❌ **不**直接套用——Hippo 多模态目标需要更结构化的检索

---

## 6. 局限性

| 限制 | 严重性 | 说明 |
|------|:---:|------|
| Block size 固定 | 🟠 中 | 推理时需预设 |
| 训练需要 landmark tokens | 🟡 低 | 模型需"学会"这种 token 含义 |
| Landmark token 必须规则插入 | 🟠 中 | 任意位置实现（high-level），但 fused 实现要求规则 |
| 仅文本 | 🔴 高 | 论文未涉及多模态 |
| 检索精度依赖 landmark 训练质量 | 🟠 中 | landmark 训练差则检索差 |
| 32K 测试上限 | 🟡 低 | 论文未测更长 |

---

## 7. 关键引用块

> **核心定位**：
> "Our method uses a landmark token to represent each block of the input and trains the attention to use it for selecting relevant blocks, enabling retrieval of blocks directly through the attention mechanism instead of by relying on a separate mechanism."

> **LLaMA 7B 成果**：
> "We show that fine-tuning LLaMA 7B with our method successfully extends its context length capacity to over 32k tokens, allowing for inference at the context lengths of GPT-4."

> **CPU/GPU 协同**：
> "Since the retrieval only requires access to the landmarks, we can reduce the memory usage significantly by swapping out (for example to CPU memory or even to disk) all regular tokens' cached key-value vectors, and then swapping them back in only if their corresponding block is retrieved by the attention."

---

## 8. 相关工作

| 名称 | 与 Landmark Attention 关系 |
|------|---------------------------|
| [Memorizing Transformers](./memorizing-transformers.md) | 同期：外部 kNN 检索（vs 内生 attention 检索）|
| [Compressive Transformers](./compressive-transformers.md) | 早期：压缩路径 |
| [StreamingLLM](./streaming-llm.md) | 同期：滑动窗口 + attention sink（vs landmark gate）|
| [InfLLM](./inf-llm.md) | 同期：块级检索（vs attention 内生检索）|
| [RMT](./rmt.md) | 同期：特殊 memory tokens（vs 特殊 landmark tokens）|
| [AutoCompressors](./auto-compressors.md) | 同期：summary vectors（不同压缩方式）|
| [discoloop.md](./discoloop.md) | 互补：Loop A（intra-decode）|
| [../research/loop-memory-survey.md §2.2](../research/loop-memory-survey.md) | Loop B 完整谱系 |
| [../research/sadko/whitepaper.md §2.1](../research/sadko/whitepaper.md) | Hippo 设计的对比参考 |

---

## 9. 对 LatentMind 的启示

### 9.1 短期（64M 阶段）

- ❌ **不**引入 landmark 机制——64M 强约束下不验证复杂机制
- ✅ 关注 **"attention 内生检索"** 思想——为 Hippo Router 设计提供新选项

### 9.2 中期（300M 阶段）

- 🔍 评估 **Landmark Attention 思想应用到 Hippo Router**——attention 自身选 block，而非显式 kNN
- 🔍 调研 **CPU/GPU 协同 + 按需 swap** 在 ChipForge APU 端侧的可行性
- 🔍 若需要"无限上下文" 能力——Landmark 是 fine-tune 即可的方案

### 9.3 长期（1B+ 阶段）

- 🎯 **fine-tune 即可扩展**支持 Logos 1B 端侧长上下文推理
- 🎯 **可解释 attention 检索**——增强 Hippo Router 的可解释性
- 🎯 **attention 内生检索** 是 Hippo 端侧部署的**最低成本方案**

---

**最后更新**：2026-07-31
**信息源**：arXiv 2305.16300v2 + NeurIPS 2023 PDF + GitHub 仓库
**作者**：来自 Loop B 文献谱系调研