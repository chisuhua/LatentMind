# TRM（Tiny Recursive Model）参考资料

> **论文**：*Less is More: Recursive Reasoning with Tiny Networks*
> **作者**：Alexia Jolicoeur-Martineau
> **机构**：Samsung SAIL Montréal
> **arXiv**：2510.04871（2025-10-07）
> **本地源**：[`papers/arxiv-2510.04871-trm.pdf`](./papers/arxiv-2510.04871-trm.pdf) (420 KB)

---

## 1. 一句话定位

**"少即是多"的递归推理**：**单网络 + 2 层 + 5M 参数**就在 ARC-AGI-1 达到 87.4%，**比 HRM 27M（55%）还高 32 个百分点**，比大多数 LLM（DeepSeek R1、o3-mini、Gemini 2.5 Pro）还好——只用其 0.01% 的参数。**直接挑战 HRM 的"双网络分层"必要性**。

---

## 2. 摘要原文翻译

> 分层推理模型（HRM）是一种使用**两个小神经网络以不同频率递归**的生物启发方法。这种方法在数独、迷宫、ARC-AGI 等难题上击败 LLM，同时用 27M 参数 / ~1000 样本训练。HRM 前景广阔，但**尚未被充分理解且可能次优**。我们提出 **TRM**（Tiny Recursive Model），一种**更简单**的递归推理方法，**用单个小网络 + 2 层**获得显著高于 HRM 的泛化能力。**仅 7M 参数**，TRM 在 **ARC-AGI-1 上达到 45% 测试准确率，ARC-AGI-2 上 8%**，高于多数 LLM（Deepseek R1、o3-mini、Gemini 2.5 Pro），参数不到其 0.01%。

---

## 3. 核心创新

### 3.1 简化 HRM

| 维度 | HRM 27M | TRM 5M |
|---|---|---|
| 网络数 | 2（H + L）| **1** |
| 每网层数 | — | **2** |
| 总参数 | 27M | **5M**（5× 更小）|
| 不动点假设 | 需要 | **不需要** |
| 生物启发解释 | 需要 | **不需要** |
| 分层结构 | 需要 | **不需要** |

### 3.2 自递归

- 单网络 f 在**自己的潜在推理特征**上递归
- **渐进式改进最终答案**：网络迭代地 refine 输出，而非一次性给答案

### 3.3 无需不动点

- HRM 假设 z_L, z_H 收敛到不动点 → 1 步梯度近似
- TRM **直接用多步 BPTT**（无不动点假设）
- 结果更简单、训练更稳定

### 3.4 简化停止机制

- HRM 用 Adaptive Computation Time（ACT）
- TRM 用**简单固定深度**（T=3, n=6）
- 少一个超参数，少一个不稳定性来源

---

## 4. 关键结果（论文 Table 1）

| 模型 | ARC-AGI 准确率 | 深度 | NFP | 参数 |
|---|---|---|---|---|
| **HRM** | 55.0% | 24 | 2 | 27M |
| **TRM (T=3, n=6)** | **87.4%** | 42 | 1 | **5M** |
| TRM + ACT | 86.1% | 42 | 2 | 5M |
| TRM + separate f_H, f_L | 82.4% | 42 | 1 | 10M |
| TRM 无 EMA | 79.9% | 42 | 1 | 5M |
| TRM 4 层, n=3 | 79.5% | 48 | 1 | 10M |
| TRM + self-attention | 74.7% | 42 | 1 | 7M |
| TRM (T=2, n=2) | 73.7% | 12 | 1 | 5M |
| TRM 1 步梯度 | 56.5% | 42 | 1 | 5M |

**关键观察**：
- 1 步梯度（HRM 风格）→ 56.5% → **多步 BPTT 至关重要**（→ 87.4%）
- 分 H/L 反而**降低**性能（82.4% vs 87.4%）→ **单网络更好**
- ACT 增加复杂度但**降低**性能（86.1% vs 87.4%）→ **固定深度更好**
- Self-attention 也**降低**性能（74.7%）→ **MLP 更适合**

---

## 5. 对 LatentMind 的启示

### 5.1 ⚠️ 关键警告

**TRM 直接质疑了 HRM 的核心设计**。如果 TRM 在语言任务上也能复现这种优势：
- HRM-Text 的 H/L 双栈设计可能是次优的
- HRM-Text 的 1 步梯度近似（DEQ 风格）应该改为多步 BPTT
- 1.15B 参数可能严重浪费

**但**：TRM 实验**仅在符号任务**（ARC-AGI、Sudoku、Maze）上验证，**未在自然语言上**测试。HRM-Text 团队在 1B 语言任务上仍然选择了 HRM 风格的双层 + 1 步梯度近似 + MagicNorm，可能有语言任务特有的原因。

### 5.2 可借鉴的具体设计

- **EMA 必要**：去掉 EMA → 79.9%（仍不错但下降）
- **多步 BPTT 必要**：1 步梯度 → 56.5%
- **T=3, n=6 的递归深度**作为基线
- **自递归**（网络 refines 自己的潜在特征）比 H/L 分离更简单

### 5.3 可能应用到 LatentMind 的方式

- **v1.5 阶段消融实验**：先用 TRM 风格的 2 层 + 自递归 跑 100M 规模语言任务，与 HRM-Text 风格的 H/L + MagicNorm 对比
- **若 TRM 风格胜出**：考虑重写 v1.5 latent engine，去掉双网络
- **若 HRM 风格胜出**：验证"语言任务的规模/特征需要双层"
- **架构选择依据**：1B 规模下，深 BPTT（n=6）的反向计算量可能成为瓶颈，需 gradient checkpointing

### 5.4 与 GRAM 的对比

| 维度 | TRM | GRAM |
|---|---|---|
| 目标 | 极小模型强泛化 | 多假设概率推理 |
| 架构 | 单网络 2 层 | HRM 风格 + 随机扰动 |
| 关键创新 | 简化 + 自递归 | 变分随机化 + 多轨迹 |
| 规模 | 5M | 10-11M（论文最大）|
| 适用 LatentMind | 简化 backbone 选择 | 增强推理（多假设）|

两者**可结合**：用 TRM 简化 backbone 结构，用 GRAM 添加多轨迹概率推理。

---

## 6. 关键引用块

> "We propose Tiny Recursive Model (TRM), a much simpler recursive reasoning approach that achieves significantly higher generalization than HRM, while using a single tiny network with only 2 layers."

> "Contrary to the Hierarchical Reasoning Model (HRM), TRM requires no fixed-point theorem, no complex biological justifications, and no hierarchy."

> "With only 7M parameters, TRM obtains 45% test-accuracy on ARC-AGI-1 and 8% on ARC-AGI-2, higher than most LLMs with less than 0.01% of the parameters."

---

## 7. 对项目决策的影响

| 决策 | 优先级 | 行动 |
|---|---|---|
| **是否重写 v1.0 backbone 为 TRM 风格** | 🟡 中 | 1B 规模语言任务上**未验证** TRM 优势 → 不建议改 v1.0 |
| **v1.5 消融实验** | 🔴 高 | 100M 语言规模上对比 HRM-Text 风格 vs TRM 风格 |
| **EMA 必用** | ✅ 确认 | HRM-Text 已用 EMA，TRM 也确认必要 |
| **BPTT 步数** | ✅ 确认 | HRM-Text 用 K=2→5，TRM 用 42（深监督） |
| **MagicNorm 是否仍需** | 🟡 中 | TRM 没用 MagicNorm 也行 → 可能 H/L 分层才需要 |

---

**最后更新**：2026-06-13
**信息源**：arXiv 2510.04871 全文、Table 1 消融数据、TRM 与 HRM/HRM-Text 的设计对比
